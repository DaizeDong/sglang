# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from sglang.srt.managers.expert_distribution import ExpertDistributionRecorder
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.utils import get_compiler_backend, is_cuda, is_hip

try:  # 🔍
    import analysis_utils
    from analysis_utils import (
        PID,
        ANALYSIS_TYPE,
        ANALYSIS_CACHE_DYNAMIC,
    )
    ANALYSIS_MODULE_LOADED = True
except Exception as e:
    PID = os.getpid()
    ANALYSIS_MODULE_LOADED = False

_is_cuda = is_cuda()
_is_hip = is_hip()

if _is_cuda:
    from sgl_kernel import moe_fused_gate

if _is_cuda or _is_hip:
    from sgl_kernel import topk_softmax


expert_distribution_recorder = ExpertDistributionRecorder()


@torch._dynamo.disable
def record_layer_balance_loss(value_name, scores, topk_ids, topk, layer_idx):  # 🔍

    def switch_load_balancing_loss_func(probs: torch.Tensor, tokens_per_expert: torch.Tensor, topk: int, moe_aux_loss_coeff: float):
        """Calculate the auxiliary loss for better load balacing.
        Please refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

        Args:
            probs (torch.Tensor): The softmax probs output by the router for each token. [num_tokens, num_experts]
            tokens_per_expert (torch.Tensor): The number of assigned tokens for each expert. [num_experts]

        Returns:
            torch.Tensor: The auxiliary loss for load balancing.
        """
        num_sub_seq = 1
        num_tokens = probs.shape[0] * topk * num_sub_seq
        num_experts = probs.shape[1]

        probs = torch.nn.functional.normalize(probs, p=1, dim=-1)
        probs_mean_per_expert = probs.clone().float().mean(dim=0)
        aux_loss = torch.sum(probs_mean_per_expert * tokens_per_expert.clone().float()) * (num_experts / num_tokens * moe_aux_loss_coeff)
        return aux_loss

    if not analysis_utils.ANALYSIS_ENABLED:
        return
    if not ANALYSIS_CACHE_DYNAMIC or ANALYSIS_CACHE_DYNAMIC[-1] is None:
        return
    if ANALYSIS_TYPE is None or value_name not in ANALYSIS_TYPE:
        return
    if layer_idx is None:
        return
    if value_name not in ANALYSIS_CACHE_DYNAMIC[-1]:
        ANALYSIS_CACHE_DYNAMIC[-1][value_name] = {}
    balance_loss = switch_load_balancing_loss_func(
        scores,
        torch.zeros_like(scores).scatter(1, topk_ids, 1),
        topk,
        moe_aux_loss_coeff=1.0,
    )
    ANALYSIS_CACHE_DYNAMIC[-1][value_name][layer_idx] = balance_loss.clone().cpu()


@torch._dynamo.disable
def record_layer_router_scores(value_name, logits, scores, topk_scores, topk_ids, layer_idx):  # 🔍
    if not analysis_utils.ANALYSIS_ENABLED:
        return
    if not ANALYSIS_CACHE_DYNAMIC or ANALYSIS_CACHE_DYNAMIC[-1] is None:
        return
    if ANALYSIS_TYPE is None or value_name not in ANALYSIS_TYPE:
        return
    if layer_idx is None:
        return
    if value_name not in ANALYSIS_CACHE_DYNAMIC[-1]:
        ANALYSIS_CACHE_DYNAMIC[-1][value_name] = {}
    ANALYSIS_CACHE_DYNAMIC[-1][value_name][layer_idx] = {
        "logits": logits.clone().cpu(),
        "scores": scores.clone().cpu(),
        "topk_scores": topk_scores.clone().cpu(),
        "topk_ids": topk_ids.clone().cpu(),
    }


def fused_topk_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    layer_idx: Optional[int] = None,  # 🔍
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_weights = F.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if ANALYSIS_MODULE_LOADED:  # 🔍
        scores = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
        record_layer_balance_loss("balance_loss", scores, topk_ids, topk, layer_idx)
        record_layer_router_scores("router_scores", gating_output, scores, topk_weights, topk_ids, layer_idx)

    return topk_weights, topk_ids


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    layer_idx: Optional[int] = None,  # 🔍
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indicies = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),
    )
    del token_expert_indicies

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if ANALYSIS_MODULE_LOADED:  # 🔍
        scores = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
        record_layer_balance_loss("balance_loss", scores, topk_ids, topk, layer_idx)
        record_layer_router_scores("router_scores", gating_output, scores, topk_weights, topk_ids, layer_idx)

    return topk_weights, topk_ids


# This is used by the Deepseek V2/V3/R1 series models
@torch.compile(dynamic=True, backend=get_compiler_backend())
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    n_share_experts_fusion: int = 0,
    routed_scaling_factor: Optional[float] = None,
    layer_idx: Optional[int] = None,  # 🔍
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    if n_share_experts_fusion:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + n_share_experts_fusion,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if n_share_experts_fusion == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum

    if ANALYSIS_MODULE_LOADED:  # 🔍
        record_layer_balance_loss("balance_loss", scores, topk_ids, topk, layer_idx)
        record_layer_router_scores("router_scores", gating_output, scores, topk_weights, topk_ids, layer_idx)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    n_share_experts_fusion: int = 0,
    routed_scaling_factor: Optional[float] = None,
    layer_idx: Optional[int] = None,  # 🔍
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)

    if n_share_experts_fusion:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + n_share_experts_fusion,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if n_share_experts_fusion == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum

    if ANALYSIS_MODULE_LOADED:  # 🔍
        record_layer_balance_loss("balance_loss", scores, topk_ids, topk, layer_idx)
        record_layer_router_scores("router_scores", gating_output, scores, topk_weights, topk_ids, layer_idx)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def is_power_of_two(n):
    return n > 0 and math.log2(n).is_integer()


def biased_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    compiled: bool = True,
    n_share_experts_fusion: int = 0,
    routed_scaling_factor: Optional[float] = None,
):
    assert (
        routed_scaling_factor is not None
    ), "routed_scaling_factor is required for biased_grouped_topk"
    # TODO: moe_fused_gate kernel is not supported for n_share_experts_fusion > 0 now.
    if (
        _is_cuda
        and gating_output.shape[1] // num_expert_group
        <= 32  # moe_fused_gate kernel ensure that num_experts/num_expert_group does not exceed MAX_VPT=32 now. And when kernel can handle MAX_VPT > 32, we can remove this assertion.
        and is_power_of_two(correction_bias.shape[0])
    ):
        return moe_fused_gate(
            gating_output,
            correction_bias,
            num_expert_group,
            topk_group,
            topk,
            n_share_experts_fusion,
            routed_scaling_factor,
        )
    else:
        biased_grouped_topk_fn = (
            torch.compile(
                biased_grouped_topk_impl, dynamic=True, backend=get_compiler_backend()
            )
            if compiled
            else biased_grouped_topk_impl
        )
        return biased_grouped_topk_fn(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
            n_share_experts_fusion=n_share_experts_fusion,
            routed_scaling_factor=routed_scaling_factor,
        )


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    correction_bias: Optional[torch.Tensor] = None,
    torch_native: bool = False,
    routed_scaling_factor: Optional[float] = None,
    layer_idx: Optional[int] = None,  # 🔍
):
    n_share_experts_fusion = global_server_args_dict["n_share_experts_fusion"]
    # DeepSeek V2/V3/R1 series models use grouped_top_k
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        if correction_bias is None:
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                n_share_experts_fusion=n_share_experts_fusion,
                routed_scaling_factor=routed_scaling_factor,
                layer_idx=layer_idx,  # 🔍
            )
        else:
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                n_share_experts_fusion=n_share_experts_fusion,
                routed_scaling_factor=routed_scaling_factor,
                layer_idx=layer_idx,  # 🔍
            )
    elif torch_native and custom_routing_function is None:
        topk_weights, topk_ids = fused_topk_native(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            layer_idx=layer_idx,  # 🔍
        )
    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            layer_idx=layer_idx,  # 🔍
        )
    else:
        if ANALYSIS_MODULE_LOADED and analysis_utils.ANALYSIS_ENABLED:  # 🔍
            import warnings
            warnings.warn(f"[{PID}] Using custom routing function which may be incompatible with `analysis_utils`. Some metrics are not available unless the custom function is manually adapted.")
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )

    expert_distribution_recorder.record_new_token(topk_ids)

    return topk_weights, topk_ids
