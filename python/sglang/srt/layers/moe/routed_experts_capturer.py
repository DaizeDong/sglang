import logging
from abc import ABC
from typing import Optional

import numpy as np
import pybase64
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_dp_local_info,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class _RoutedExpertsDeviceCache:
    def __init__(
        self,
        max_running_requests: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int,
        device: str,
    ) -> None:
        self.buffer = torch.zeros(
            (
                max(
                    get_global_server_args().chunked_prefill_size
                    * get_global_server_args().dp_size,
                    max_running_requests,
                ),
                num_hidden_layers,
                num_experts_per_tok + num_fused_shared_experts,
            ),
            dtype=torch.int32,
            device=device,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[:batch, layer_id, :] = topk_ids

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_MB = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Routing experts device buffer allocated. #shape: {tuple(self.buffer.shape)}, size: {buffer_size_MB:.2f} MB"
        )


class _RoutedExpertsHostCache:
    def __init__(
        self,
        num_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
    ) -> None:
        self.num_tokens = num_tokens
        self.buffer = torch.zeros(
            (
                num_tokens,
                num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self._request_token_experts: dict[int, dict[int, torch.Tensor]] = {}
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def set_experts_buffer(self, layer_id: int, loc: torch.Tensor, top_k: torch.Tensor):
        self.buffer[layer_id, loc, :] = top_k.to(device="cpu", non_blocking=True)

    def set_request_topk(
        self,
        req_pool_idx: int,
        token_positions: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        if topk_ids.numel() == 0:
            return
        assert token_positions.ndim == 1, (
            f"token_positions must be 1D, got shape={tuple(token_positions.shape)}"
        )
        assert token_positions.numel() == topk_ids.shape[0], (
            f"token_positions/topk_ids size mismatch: {token_positions.numel()} vs {topk_ids.shape[0]}"
        )
        if torch.any(token_positions == 0):
            self._request_token_experts.pop(int(req_pool_idx), None)
        request_states = self._request_token_experts.setdefault(int(req_pool_idx), {})
        for local_idx, token_position in enumerate(token_positions.tolist()):
            request_states[int(token_position)] = topk_ids[local_idx]

    def get_request_topk(self, req_pool_idx: int):
        request_states = self._request_token_experts.get(int(req_pool_idx))
        if not request_states:
            return None, None
        token_positions = sorted(request_states)
        topk_per_token = [request_states[token_position] for token_position in token_positions]
        return torch.stack(topk_per_token, dim=0), torch.tensor(token_positions, dtype=torch.int32)

    def clear_request(self, req_pool_idx: int) -> None:
        self._request_token_experts.pop(int(req_pool_idx), None)

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_GB = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"Routing experts host buffer allocated. #tokens: {self.num_tokens}, size: {buffer_size_GB:.2f} GB"
        )


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_fused_shared_experts: int,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                num_fused_shared_experts=num_fused_shared_experts,
                device=device,
            )
        else:
            return _RoutedExpertsCapturerNoop()

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        raise NotImplementedError

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        raise NotImplementedError

    def get_host_cache(self):
        raise NotImplementedError

    def get_device_cache(self):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok

        self.host_cache = _RoutedExpertsHostCache(
            num_tokens=num_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
        )

        self.device_cache = _RoutedExpertsDeviceCache(
            max_running_requests=max_running_requests,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            num_fused_shared_experts=self.num_fused_shared_experts,
            device=device,
        )

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        if is_dp_attention_enabled():
            local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
            # handle with cuda graph padding
            if can_run_graph:
                local_start_pos = get_attention_dp_rank() * cuda_graph_batch
                local_end_pos = local_start_pos + local_num_tokens
            else:
                local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos = 0
            local_end_pos = forward_batch.out_cache_loc.shape[0]

        # FIXME: sync explicitly here, overlap scheduler breaks here.
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        experts_cpu = self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.num_experts_per_tok
        ].cpu()
        self.host_cache.buffer[out_cache_loc_cpu] = experts_cpu

        if is_dp_attention_enabled():
            return

        if forward_batch.forward_mode.is_decode() or forward_batch.forward_mode.is_target_verify():
            token_counts = [1] * forward_batch.batch_size
        else:
            if forward_batch.extend_seq_lens_cpu is not None:
                token_counts = [int(x) for x in forward_batch.extend_seq_lens_cpu]
            elif forward_batch.extend_seq_lens is not None:
                token_counts = [int(x) for x in forward_batch.extend_seq_lens.cpu().tolist()]
            else:
                logger.warning(
                    "[RoutedExperts] Missing extend_seq_lens for forward_mode=%s. "
                    "Skipping explicit position-aware cache update.",
                    forward_batch.forward_mode,
                )
                return

        total_tokens = int(sum(token_counts))
        if total_tokens != int(experts_cpu.shape[0]):
            logger.warning(
                "[RoutedExperts] Token partition mismatch: total_tokens=%s experts_tokens=%s forward_mode=%s. "
                "Skipping explicit position-aware cache update.",
                total_tokens,
                int(experts_cpu.shape[0]),
                forward_batch.forward_mode,
            )
            return

        positions_cpu = forward_batch.positions.cpu()
        if int(positions_cpu.shape[0]) != total_tokens:
            logger.warning(
                "[RoutedExperts] positions/token mismatch: positions=%s total_tokens=%s forward_mode=%s. "
                "Skipping explicit position-aware cache update.",
                int(positions_cpu.shape[0]),
                total_tokens,
                forward_batch.forward_mode,
            )
            return

        req_pool_indices = [int(x) for x in forward_batch.req_pool_indices.cpu().tolist()]
        offset = 0
        for req_pool_idx, token_count in zip(req_pool_indices, token_counts):
            if token_count <= 0:
                continue
            next_offset = offset + token_count
            self.host_cache.set_request_topk(
                req_pool_idx=req_pool_idx,
                token_positions=positions_cpu[offset:next_offset].to(torch.int32),
                topk_ids=experts_cpu[offset:next_offset],
            )
            offset = next_offset

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        expected_tokens = max(seqlen - 1, 0)
        cache_pool_idx = req_to_token_pool.req_to_token[req_pool_idx][:expected_tokens].cpu().clone()
        implicit_routed = self.get_host_cache().buffer[cache_pool_idx]

        explicit_routed, explicit_positions = self.get_host_cache().get_request_topk(req_pool_idx)
        self.get_host_cache().clear_request(req_pool_idx)
        if explicit_routed is None or explicit_positions is None or expected_tokens == 0:
            return implicit_routed

        explicit_positions = explicit_positions.to(torch.long)
        valid_mask = (explicit_positions >= 0) & (explicit_positions < expected_tokens)
        if not bool(valid_mask.all()):
            logger.warning(
                "[RoutedExperts] Dropping %s out-of-range explicit positions for req_pool_idx=%s expected_tokens=%s",
                int((~valid_mask).sum().item()),
                req_pool_idx,
                expected_tokens,
            )
            explicit_routed = explicit_routed[valid_mask]
            explicit_positions = explicit_positions[valid_mask]
        if explicit_positions.numel() == 0:
            return implicit_routed

        blended_routed = implicit_routed.clone()
        implicit_at_positions = blended_routed[explicit_positions]
        mismatch_count = int((implicit_at_positions != explicit_routed).any(dim=-1).sum().item())
        if mismatch_count > 0:
            logger.warning(
                "[RoutedExperts] req_to_token reconstruction mismatched explicit position cache for req_pool_idx=%s "
                "expected_tokens=%s explicit_tokens=%s mismatched_tokens=%s head_positions=%s",
                req_pool_idx,
                expected_tokens,
                int(explicit_positions.numel()),
                mismatch_count,
                explicit_positions[:8].tolist(),
            )
        blended_routed[explicit_positions] = explicit_routed
        return blended_routed

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        self._sync_fwd_experts_buffer_DtoH(
            forward_batch=forward_batch,
            can_run_graph=can_run_graph,
            cuda_graph_batch=cuda_graph_batch,
        )

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        pass

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        pass

    def get_host_cache(self):
        pass

    def get_device_cache(self):
        pass


_global_expert_capturer: Optional[RoutedExpertsCapturer] = _RoutedExpertsCapturerNoop()


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer


def extract_routed_experts_from_meta_info(data):
    # To solve the performance issue, we return the experts_ids in base64
    # We left this function for user to change it back to normal int32
    # See detokenizer_manager::_extract_routed_experts
    routed_experts_base64 = data["meta_info"].get("routed_experts", None)
    routed_experts = np.frombuffer(
        pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=np.int32
    )
    return routed_experts
