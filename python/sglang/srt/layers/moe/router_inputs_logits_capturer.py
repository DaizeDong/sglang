"""
Capturer for router inputs and logits for predictive routing replay.
Similar to verl's merge_router_predictive_data logic with downsampling.
"""
import json
import logging
import os
import random
import socket
import zlib
from abc import ABC
from typing import Dict, Optional

import numpy as np
import pybase64
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def _debug_r3_trace_enabled() -> bool:
    return os.getenv("VERL_DEBUG_R3_TRACE", "").lower() in {"1", "true", "yes", "on"}


def _debug_r3_trace_save_dir() -> Optional[str]:
    value = os.getenv("VERL_DEBUG_R3_TRACE_SAVE_DIR", "").strip()
    return value or None


def _debug_router_states_enabled() -> bool:
    return os.getenv("VERL_DEBUG_ROUTER_STATES", "").lower() in {"1", "true", "yes", "on"}


def _normalize_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().view(-1).tolist()]
    return [int(x) for x in np.asarray(value).reshape(-1).tolist()]


def _checksum_array(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        array = tensor.numpy()
    else:
        array = np.ascontiguousarray(np.asarray(value))
        if str(array.dtype) == "bfloat16":
            array = array.astype(np.float32, copy=False)
    return f"{zlib.crc32(array.tobytes()) & 0xFFFFFFFF:08x}"


def _tensor_summary(value) -> Optional[dict]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous()
        shape = list(array.shape)
        dtype = str(array.dtype)
    else:
        array = np.asarray(value)
        shape = list(array.shape)
        dtype = str(array.dtype)
    return {
        "shape": shape,
        "dtype": dtype,
        "checksum": _checksum_array(array),
    }


def _append_r3_trace(source: str, payload: dict) -> None:
    if not _debug_r3_trace_enabled():
        return
    record = {"source": source, **payload}
    message = json.dumps(record, sort_keys=True)
    logger.warning("[R3Trace] %s", message)
    print(f"[R3Trace] {message}", flush=True)
    save_dir = _debug_r3_trace_save_dir()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        trace_path = os.path.join(save_dir, f"trace-{socket.gethostname()}-{os.getpid()}.jsonl")
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


def _resolve_num_experts(config) -> int:
    for attr_name in ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"):
        value = getattr(config, attr_name, None)
        if value is not None:
            return value
    raise AttributeError(
        f"Unable to resolve number of experts from config type={type(config).__name__}. "
        "Expected one of: num_local_experts, num_experts, n_routed_experts, moe_num_experts."
    )


class _RouterInputsLogitsDeviceCache:
    """Device (GPU) cache for router inputs and logits."""
    
    def __init__(
        self,
        max_running_requests: int,
        num_hidden_layers: int,
        hidden_size: int,
        num_experts: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        """
        Initialize device cache buffers for router inputs and logits.
        
        Args:
            max_running_requests: Maximum number of concurrent requests
            num_hidden_layers: Number of MoE layers in the model
            hidden_size: Hidden dimension size
            num_experts: Number of experts in each MoE layer
            device: Device to allocate tensors on
            dtype: Data type (use model's native dtype, no conversion)
        """
        max_tokens = max(
            get_global_server_args().chunked_prefill_size * get_global_server_args().dp_size,
            max_running_requests,
        )
        
        # Buffer for router inputs: [max_tokens, num_layers, hidden_size]
        self.inputs_buffer = torch.zeros(
            (max_tokens, num_hidden_layers, hidden_size),
            dtype=dtype,
            device=device,
        )
        
        # Buffer for router logits: [max_tokens, num_layers, num_experts]
        self.logits_buffer = torch.zeros(
            (max_tokens, num_hidden_layers, num_experts),
            dtype=dtype,
            device=device,
        )

        # Buffer for router bias (delta_logits from bias_predictor): [max_tokens, num_layers, num_experts]
        # Only allocated once; populated only when bias_predictor is used.
        self.bias_buffer = torch.zeros(
            (max_tokens, num_hidden_layers, num_experts),
            dtype=dtype,
            device=device,
        )
        # Track whether bias has been written this forward pass
        self.has_bias = False
        
        self._finalize_allocation_log()
    
    def get_buffer_size_bytes(self):
        """Calculate total buffer size in bytes."""
        inputs_size = get_tensor_size_bytes(self.inputs_buffer)
        logits_size = get_tensor_size_bytes(self.logits_buffer)
        bias_size = get_tensor_size_bytes(self.bias_buffer)
        return inputs_size + logits_size + bias_size
    
    def capture_fwd_router_states(
        self, 
        layer_id: int, 
        router_inputs: torch.Tensor, 
        router_logits: torch.Tensor,
        router_bias: Optional[torch.Tensor] = None,
    ):
        """
        Capture router inputs, logits, and optional bias for a specific layer during forward pass.
        NO dtype conversion - store as is.
        
        Args:
            layer_id: Layer index (0-indexed)
            router_inputs: Router input tensor [batch_tokens, hidden_size]
            router_logits: Router logits tensor [batch_tokens, num_experts]
            router_bias: Optional delta_logits from bias_predictor [batch_tokens, num_experts]
        """
        assert layer_id is not None, "capturing router states but get layer_id None"
        batch_tokens = router_inputs.shape[0]
        
        # Store in buffers WITHOUT dtype conversion
        self.inputs_buffer[:batch_tokens, layer_id, :] = router_inputs
        self.logits_buffer[:batch_tokens, layer_id, :] = router_logits
        if router_bias is not None:
            self.bias_buffer[:batch_tokens, layer_id, :] = router_bias
            self.has_bias = True
    
    def _finalize_allocation_log(self):
        """Log buffer allocation info."""
        total_size_MB = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Router inputs/logits device buffer allocated. "
            f"inputs_shape: {tuple(self.inputs_buffer.shape)}, "
            f"logits_shape: {tuple(self.logits_buffer.shape)}, "
            f"bias_shape: {tuple(self.bias_buffer.shape)}, "
            f"dtype: {self.inputs_buffer.dtype}, "
            f"total_size: {total_size_MB:.2f} MB"
        )


class _RouterInputsLogitsHostCache:
    """Host (CPU) cache for sparse per-request router states keyed by token position."""

    def __init__(
        self,
        num_tokens: int,
        num_hidden_layers: int,
        hidden_size: int,
        num_experts: int,
        dtype: torch.dtype,
    ) -> None:
        # Keep the original constructor signature for call-site compatibility.
        self.num_tokens = num_tokens
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.dtype = dtype
        self._request_token_states: Dict[
            int, Dict[int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
        ] = {}
        self._capture_decision: Dict[int, bool] = {}
        self._finalize_allocation_log()

    def _should_capture_request(self, req_pool_idx: int) -> bool:
        capture = self._capture_decision.get(req_pool_idx)
        if capture is None:
            sample_rate = get_global_server_args().router_states_sample_rate
            capture = sample_rate >= 1.0 or random.random() < sample_rate
            self._capture_decision[req_pool_idx] = capture
            _append_r3_trace(
                "sglang.host_cache.capture_decision",
                {
                    "req_pool_idx": int(req_pool_idx),
                    "capture_decision": bool(capture),
                    "sample_rate": float(sample_rate),
                },
            )
        return capture

    def get_capture_decision(self, req_pool_idx: int) -> Optional[bool]:
        return self._capture_decision.get(req_pool_idx)

    def clear_request(
        self,
        req_pool_idx: int,
        cache_locs: Optional[torch.Tensor] = None,
        reason: Optional[str] = None,
    ) -> None:
        self._capture_decision.pop(req_pool_idx, None)
        cache_loc_list = _normalize_int_list(cache_locs)
        request_states = self._request_token_states.pop(int(req_pool_idx), None)
        removed_states = 0 if request_states is None else len(request_states)
        skipped_foreign_owner = 0
        if cache_loc_list:
            _append_r3_trace(
                "sglang.host_cache.clear_request",
                {
                    "req_pool_idx": int(req_pool_idx),
                    "reason": reason,
                    "cache_locs_head": cache_loc_list[:8],
                    "cache_locs_count": len(cache_loc_list),
                    "cache_locs_checksum": _checksum_array(np.asarray(cache_loc_list, dtype=np.int64)),
                    "removed_token_states": removed_states,
                    "skipped_foreign_owner": skipped_foreign_owner,
                },
            )

    def set_token_states(
        self,
        req_pool_idx: int,
        cache_locs: torch.Tensor,
        token_positions: torch.Tensor,
        router_inputs: torch.Tensor,
        router_logits: torch.Tensor,
        router_bias: Optional[torch.Tensor] = None,
    ) -> None:
        if cache_locs.numel() == 0:
            return
        assert cache_locs.numel() == token_positions.numel(), (
            f"cache_locs/token_positions size mismatch: {cache_locs.numel()} vs {token_positions.numel()}"
        )

        has_bias = router_bias is not None
        request_states = self._request_token_states.setdefault(int(req_pool_idx), {})
        for local_idx, token_position in enumerate(token_positions.tolist()):
            request_states[int(token_position)] = (
                router_inputs[local_idx],
                router_logits[local_idx],
                router_bias[local_idx] if has_bias else None,
            )

    def get_request(
        self,
        req_pool_idx: int,
        cache_locs: Optional[torch.Tensor] = None,
    ):
        capture_decision = self._should_capture_request(req_pool_idx)
        if not capture_decision:
            return None, None, None, None, 0, False, "sample_rate", capture_decision

        request_states = self._request_token_states.get(int(req_pool_idx))
        if not request_states:
            return None, None, None, None, 0, False, "empty", capture_decision

        inputs_per_token = []
        logits_per_token = []
        bias_per_token = []
        token_positions = []
        missing_bias = False

        for token_pos in sorted(request_states):
            state = request_states[token_pos]
            router_inputs, router_logits, router_bias = state
            inputs_per_token.append(router_inputs)
            logits_per_token.append(router_logits)
            token_positions.append(token_pos)
            if router_bias is None:
                missing_bias = True
            else:
                bias_per_token.append(router_bias)

        missing_count = 0
        if missing_bias or not inputs_per_token:
            drop_reason = "missing_bias" if missing_bias else "empty"
            return (
                None,
                None,
                None,
                None,
                missing_count,
                missing_bias,
                drop_reason,
                capture_decision,
            )

        return (
            torch.stack(inputs_per_token, dim=0),
            torch.stack(logits_per_token, dim=0),
            torch.stack(bias_per_token, dim=0),
            torch.tensor(token_positions, dtype=torch.int32),
            missing_count,
            False,
            None,
            capture_decision,
        )

    def _finalize_allocation_log(self):
        logger.info(
            "Router inputs/logits host cache initialized with token-pool sparse storage. "
            "num_hidden_layers=%s hidden_size=%s num_experts=%s dtype=%s",
            self.num_hidden_layers,
            self.hidden_size,
            self.num_experts,
            self.dtype,
        )


class RouterInputsLogitsCapturer(ABC):
    """Abstract base class for router inputs/logits capturer."""
    
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ):
        """
        Factory method to create capturer instance.
        
        Args:
            enable: Whether to enable capturing
            model_config: Model configuration
            num_tokens: Total number of tokens to cache
            max_running_requests: Maximum concurrent requests
            device: Device string
        """
        if enable:
            return _RouterInputsLogitsCapturerReal(
                model_config=model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                device=device,
            )
        else:
            return _RouterInputsLogitsCapturerNoop()
    
    def capture(
        self,
        layer_id: int,
        router_inputs: torch.Tensor,
        router_logits: torch.Tensor,
        router_bias: Optional[torch.Tensor] = None,
    ):
        """Capture router inputs, logits (and optional bias) for a specific layer."""
        raise NotImplementedError
    
    def get_router_states(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        """
        Get cached router states for a finished request.
        Returns numpy arrays without padding (actual sequence length).
        Returns tuple (inputs, logits, bias) where bias may be None.
        """
        raise NotImplementedError
    
    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        """Called after forward pass to sync device cache to host cache."""
        raise NotImplementedError
    
    def get_host_cache(self):
        """Get host cache object."""
        raise NotImplementedError
    
    def get_device_cache(self):
        """Get device cache object."""
        raise NotImplementedError


class _RouterInputsLogitsCapturerReal(RouterInputsLogitsCapturer):
    """Real implementation of router inputs/logits capturer."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ):
        # Get model dimensions
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.hidden_size = model_config.hf_text_config.hidden_size
        self.num_experts = _resolve_num_experts(model_config.hf_text_config)
        
        # Use model's native dtype (usually bfloat16)
        dtype = getattr(model_config.hf_text_config, "torch_dtype", torch.bfloat16)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        
        # Initialize host cache (CPU)
        self.host_cache = _RouterInputsLogitsHostCache(
            num_tokens=num_tokens,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            dtype=dtype,
        )
        
        # Initialize device cache (GPU)
        self.device_cache = _RouterInputsLogitsDeviceCache(
            max_running_requests=max_running_requests,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            device=device,
            dtype=dtype,
        )
    
    def _sync_fwd_states_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        """Sync device cache to compact per-request CPU storage after forward pass."""
        if is_dp_attention_enabled():
            logger.warning(
                "[RouterStates] DP attention is enabled; per-request router-state capture is not "
                "implemented for this mode. Dropping router states for this forward batch."
            )
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
                    "[RouterStates] Missing extend_seq_lens for forward_mode=%s. "
                    "Dropping router states for this forward batch.",
                    forward_batch.forward_mode,
                )
                return

        total_tokens = int(sum(token_counts))
        if total_tokens != int(forward_batch.out_cache_loc.shape[0]):
            logger.warning(
                "[RouterStates] Token partition mismatch: total_tokens=%s out_cache_loc=%s "
                "forward_mode=%s. Dropping router states for this forward batch.",
                total_tokens,
                int(forward_batch.out_cache_loc.shape[0]),
                forward_batch.forward_mode,
            )
            return

        req_pool_indices = [int(x) for x in forward_batch.req_pool_indices.cpu().tolist()]
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        positions_cpu = forward_batch.positions.cpu()
        if int(positions_cpu.shape[0]) != total_tokens:
            logger.warning(
                "[RouterStates] positions/out_cache_loc mismatch: positions=%s total_tokens=%s "
                "forward_mode=%s. Dropping router states for this forward batch.",
                int(positions_cpu.shape[0]),
                total_tokens,
                forward_batch.forward_mode,
            )
            return
        offset = 0
        for req_pool_idx, token_count in zip(req_pool_indices, token_counts):
            if token_count <= 0:
                continue

            next_offset = offset + token_count
            token_positions_cpu = positions_cpu[offset:next_offset].to(torch.int32)
            inputs_cpu = self.device_cache.inputs_buffer[offset:next_offset, :, :].cpu()
            logits_cpu = self.device_cache.logits_buffer[offset:next_offset, :, :].cpu()
            bias_cpu = None
            if self.device_cache.has_bias:
                bias_cpu = self.device_cache.bias_buffer[offset:next_offset, :, :].cpu()
            self.host_cache.set_token_states(
                req_pool_idx=req_pool_idx,
                cache_locs=out_cache_loc_cpu[offset:next_offset],
                token_positions=token_positions_cpu,
                router_inputs=inputs_cpu,
                router_logits=logits_cpu,
                router_bias=bias_cpu,
            )
            _append_r3_trace(
                "sglang.capture.sync_fwd_states",
                {
                    "req_pool_idx": int(req_pool_idx),
                    "token_count": int(token_count),
                    "cache_locs_head": _normalize_int_list(out_cache_loc_cpu[offset:next_offset])[:8],
                    "cache_locs_checksum": _checksum_array(out_cache_loc_cpu[offset:next_offset]),
                    "token_positions_head": _normalize_int_list(token_positions_cpu)[:8],
                    "token_positions_checksum": _checksum_array(token_positions_cpu),
                    "router_inputs": _tensor_summary(inputs_cpu),
                    "router_logits": _tensor_summary(logits_cpu),
                    "router_bias": _tensor_summary(bias_cpu),
                },
            )
            offset = next_offset
    
    def capture(
        self,
        layer_id: int,
        router_inputs: torch.Tensor,
        router_logits: torch.Tensor,
        router_bias: Optional[torch.Tensor] = None,
    ):
        """Capture router inputs, logits, and optional bias during forward pass."""
        self.device_cache.capture_fwd_router_states(layer_id, router_inputs, router_logits, router_bias)
    
    def get_router_states(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        """
        Get router states for a finished request.
        Returns numpy arrays of actual sequence length (no padding).
        Applies sequence length filtering based on server args.
        
        Returns:
            tuple: (inputs_array, logits_array, bias_array, token_positions_array)
                - inputs_array: numpy array [seqlen-1, num_layers, hidden_size] or None if filtered
                - logits_array: numpy array [seqlen-1, num_layers, num_experts] or None if filtered
                - bias_array: numpy array [seqlen-1, num_layers, num_experts] or None if no bias / filtered
                - token_positions_array: numpy array [seqlen-1] or None if filtered
        """
        del req_to_token_pool
        expected_tokens = max(seqlen, 0)
        if expected_tokens == 0:
            self.host_cache.clear_request(req_pool_idx, reason="empty_request")
            return None, None, None, None

        (
            router_inputs,
            router_logits,
            router_bias,
            router_token_positions,
            missing_count,
            missing_bias,
            drop_reason,
            capture_decision,
        ) = (
            self.host_cache.get_request(
                req_pool_idx=req_pool_idx,
            )
        )
        self.host_cache.clear_request(
            req_pool_idx,
            reason=drop_reason or "success",
        )
        if router_inputs is None:
            if _debug_router_states_enabled():
                print(
                    f"[RouterStates] get_router_states req_pool_idx={req_pool_idx} -> None "
                    f"(selected_tokens={expected_tokens}, "
                    f"expected_tokens={expected_tokens}, missing_count={missing_count}, missing_bias={missing_bias})",
                    flush=True,
                )
            if missing_count > 0:
                logger.warning(
                    "[RouterStates] Request req_pool_idx=%s missing %s/%s token-pool states. "
                    "Likely prefix-cache reuse without matching capture. Dropping router states for this request.",
                    req_pool_idx,
                    missing_count,
                    expected_tokens,
                )
            elif missing_bias:
                logger.warning(
                    "[RouterStates] Request req_pool_idx=%s recovered inputs/logits but bias is missing for at least one token. "
                    "Dropping router states for this request.",
                    req_pool_idx,
                )
            _append_r3_trace(
                "sglang.router_states.get",
                {
                    "req_pool_idx": int(req_pool_idx),
                    "seqlen": int(seqlen),
                    "expected_tokens": int(expected_tokens),
                    "capture_decision": capture_decision,
                    "drop_reason": drop_reason,
                    "missing_count": int(missing_count),
                    "missing_bias": bool(missing_bias),
                },
            )
            return None, None, None, None

        expected_selected_tokens = expected_tokens
        actual_tokens = int(router_inputs.shape[0])
        missing_count = max(expected_selected_tokens - actual_tokens, 0)
        if actual_tokens != expected_selected_tokens:
            if _debug_router_states_enabled():
                print(
                    f"[RouterStates] get_router_states req_pool_idx={req_pool_idx} partial capture "
                    f"expected={expected_selected_tokens} actual={actual_tokens} "
                    f"missing_count={missing_count} partial_reason={drop_reason}",
                    flush=True,
                )
            _append_r3_trace(
                "sglang.router_states.get",
                {
                    "req_pool_idx": int(req_pool_idx),
                    "seqlen": int(seqlen),
                    "expected_tokens": int(expected_tokens),
                    "capture_decision": capture_decision,
                    "drop_reason": drop_reason or "partial",
                    "expected_selected_tokens": expected_selected_tokens,
                    "actual_tokens": actual_tokens,
                    "missing_count": int(missing_count),
                    "router_token_positions_head": _normalize_int_list(router_token_positions)[:8],
                    "router_token_positions_checksum": _checksum_array(router_token_positions),
                },
            )

        # Uniform token subsampling to reduce transfer size
        tokens_per_seq = get_global_server_args().router_states_tokens_per_seq
        if tokens_per_seq is not None and actual_tokens > tokens_per_seq:
            import numpy as np
            idx = np.round(np.linspace(0, actual_tokens - 1, tokens_per_seq)).astype(np.int64)
            idx = np.clip(idx, 0, actual_tokens - 1)
            idx_t = torch.from_numpy(idx).to(router_inputs.device)
            router_inputs = router_inputs[idx_t]
            router_logits = router_logits[idx_t]
            if router_bias is not None:
                router_bias = router_bias[idx_t]
            router_token_positions = router_token_positions[idx_t]
            actual_tokens = tokens_per_seq

        # Keep the current wire format compatible with verl's float16 base64 decoder.
        router_inputs_np = router_inputs.to(torch.float16).contiguous().numpy()
        router_logits_np = router_logits.to(torch.float16).contiguous().numpy()
        router_bias_np = None
        if router_bias is not None:
            router_bias_np = router_bias.to(torch.float16).contiguous().numpy()
        router_token_positions_np = router_token_positions.to(torch.int32).contiguous().numpy()

        if _debug_router_states_enabled():
            print(
                f"[RouterStates] get_router_states req_pool_idx={req_pool_idx} success "
                f"inputs_shape={tuple(router_inputs_np.shape)} logits_shape={tuple(router_logits_np.shape)} "
                f"bias_shape={tuple(router_bias_np.shape) if router_bias_np is not None else None} "
                f"positions_shape={tuple(router_token_positions_np.shape)}",
                flush=True,
            )
        _append_r3_trace(
            "sglang.router_states.get",
            {
                "req_pool_idx": int(req_pool_idx),
                "seqlen": int(seqlen),
                "expected_tokens": int(expected_tokens),
                "capture_decision": capture_decision,
                "drop_reason": drop_reason,
                "missing_count": int(missing_count),
                "router_inputs": _tensor_summary(router_inputs_np),
                "router_logits": _tensor_summary(router_logits_np),
                "router_bias": _tensor_summary(router_bias_np),
                "router_token_positions": _tensor_summary(router_token_positions_np),
                "router_token_positions_head": _normalize_int_list(router_token_positions_np)[:8],
            },
        )
        return router_inputs_np, router_logits_np, router_bias_np, router_token_positions_np
    
    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        """Sync device cache to host cache after forward pass."""
        self._sync_fwd_states_buffer_DtoH(
            forward_batch=forward_batch,
            can_run_graph=can_run_graph,
            cuda_graph_batch=cuda_graph_batch,
        )
        if _debug_router_states_enabled():
            print(
                f"[RouterStates] on_forward_end mode={forward_batch.forward_mode} "
                f"batch_size={forward_batch.batch_size} out_cache_loc={int(forward_batch.out_cache_loc.shape[0])} "
                f"has_bias={self.device_cache.has_bias}",
                flush=True,
            )
        # Reset has_bias flag for next forward pass
        self.device_cache.has_bias = False
    
    def get_host_cache(self):
        return self.host_cache
    
    def get_device_cache(self):
        return self.device_cache


class _RouterInputsLogitsCapturerNoop(RouterInputsLogitsCapturer):
    """No-op implementation when capturing is disabled."""
    
    def __init__(self):
        pass
    
    def capture(
        self,
        layer_id: int,
        router_inputs: torch.Tensor,
        router_logits: torch.Tensor,
        router_bias: Optional[torch.Tensor] = None,
    ):
        pass
    
    def get_router_states(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        return None, None, None, None
    
    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        pass
    
    def get_host_cache(self):
        return None
    
    def get_device_cache(self):
        return None


# Global singleton instance
_global_router_states_capturer: Optional[RouterInputsLogitsCapturer] = _RouterInputsLogitsCapturerNoop()


def get_global_router_states_capturer():
    """Get the global router states capturer instance."""
    return _global_router_states_capturer


def set_global_router_states_capturer(capturer: RouterInputsLogitsCapturer):
    """Set the global router states capturer instance."""
    global _global_router_states_capturer
    _global_router_states_capturer = capturer


def extract_router_states_from_meta_info(data, hidden_size, num_experts, num_layers):
    """
    Extract router inputs and logits from base64-encoded meta_info.
    
    Args:
        data: Output dictionary containing meta_info
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        num_layers: Number of layers
    
    Returns:
        tuple: (router_inputs, router_logits) as numpy arrays
    """
    # Decode inputs
    router_inputs_base64 = data["meta_info"].get("router_inputs", None)
    if router_inputs_base64 is not None:
        router_inputs = np.frombuffer(
            pybase64.b64decode(router_inputs_base64.encode("utf-8")), 
            dtype=np.float16  # Assume bf16 was stored as fp16 for compatibility
        )
        # Reshape to [seqlen, num_layers, hidden_size]
        num_tokens = len(router_inputs) // (num_layers * hidden_size)
        router_inputs = router_inputs.reshape(num_tokens, num_layers, hidden_size)
    else:
        router_inputs = None
    
    # Decode logits
    router_logits_base64 = data["meta_info"].get("router_logits", None)
    if router_logits_base64 is not None:
        router_logits = np.frombuffer(
            pybase64.b64decode(router_logits_base64.encode("utf-8")), 
            dtype=np.float16
        )
        # Reshape to [seqlen, num_layers, num_experts]
        num_tokens = len(router_logits) // (num_layers * num_experts)
        router_logits = router_logits.reshape(num_tokens, num_layers, num_experts)
    else:
        router_logits = None
    
    return router_inputs, router_logits
