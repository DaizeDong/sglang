import os
import re

import torch

try:  # 🔍
    import analysis_utils
    from analysis_utils import (
        PID,
        ANALYSIS_TYPE,
        ANALYSIS_CACHE_DYNAMIC,
        ANALYSIS_CACHE_STATIC,
    )
    ANALYSIS_MODULE_LOADED = True
except Exception as e:
    PID = os.getpid()
    ANALYSIS_MODULE_LOADED = False
print(f"[{PID}] ANALYSIS_MODULE_LOADED: {ANALYSIS_MODULE_LOADED}")


@torch._dynamo.disable
def record_value(value_name, value):  # 🔍
    # print(f"[{PID}] {value_name} ({value.shape})\n{value}")
    if not analysis_utils.ANALYSIS_ENABLED:
        return
    if not ANALYSIS_CACHE_DYNAMIC or ANALYSIS_CACHE_DYNAMIC[-1] is None:
        return
    if ANALYSIS_TYPE is None or value_name not in ANALYSIS_TYPE:
        return
    if value is None:
        return
    ANALYSIS_CACHE_DYNAMIC[-1][value_name] = value.clone().cpu()


@torch._dynamo.disable
def record_layer_value(value_name, value, layer_idx):  # 🔍
    if not analysis_utils.ANALYSIS_ENABLED:
        return
    if not ANALYSIS_CACHE_DYNAMIC or ANALYSIS_CACHE_DYNAMIC[-1] is None:
        return
    if ANALYSIS_TYPE is None or value_name not in ANALYSIS_TYPE:
        return
    if value_name not in ANALYSIS_CACHE_DYNAMIC[-1]:
        ANALYSIS_CACHE_DYNAMIC[-1][value_name] = {}
    ANALYSIS_CACHE_DYNAMIC[-1][value_name][layer_idx] = value.clone().cpu()


@torch._dynamo.disable
def record_layer_activation_magnitude(value_name, value, layer_idx):  # 🔍
    if not analysis_utils.ANALYSIS_ENABLED:
        return
    if not ANALYSIS_CACHE_DYNAMIC or ANALYSIS_CACHE_DYNAMIC[-1] is None:
        return
    for name, p in [
        (string, int(re.search(r"activation_magnitude_l(\d+)", string).group(1)))
        for string in ANALYSIS_TYPE
        if re.search(r"activation_magnitude_l(\d+)", string)
    ]:
        if name not in ANALYSIS_CACHE_DYNAMIC[-1]:
            ANALYSIS_CACHE_DYNAMIC[-1][name] = {}
        if layer_idx not in ANALYSIS_CACHE_DYNAMIC[-1][name]:
            ANALYSIS_CACHE_DYNAMIC[-1][name][layer_idx] = {}
        # element value magnitude
        pow_value = torch.pow(value.abs(), p)  # take the abs first
        ANALYSIS_CACHE_DYNAMIC[-1][name][layer_idx][value_name] = (
            pow_value.min(dim=0).values.cpu(), pow_value.mean(dim=0).cpu(), pow_value.max(dim=0).values.cpu()
        )  # calculate the min, mean and max
        # vector length
        ANALYSIS_CACHE_DYNAMIC[-1][name][layer_idx]["vector#" + value_name] = torch.norm(value, p=p, dim=-1, dtype=torch.float32).cpu()


@torch._dynamo.disable
def record_layer_weights(value_name, value, layer_idx):  # 🔍
    if value_name not in ANALYSIS_CACHE_STATIC:
        ANALYSIS_CACHE_STATIC[value_name] = {}
    ANALYSIS_CACHE_STATIC[value_name][layer_idx] = value.clone().cpu()


@torch._dynamo.disable
def record_layer_weights_magnitude(param_name, value, layer_idx):  # 🔍
    for name, p in [
        (string, int(re.search(r"weights_magnitude_l(\d+)", string).group(1)))
        for string in ANALYSIS_TYPE
        if re.search(r"weights_magnitude_l(\d+)", string)
    ]:
        if name not in ANALYSIS_CACHE_STATIC:
            ANALYSIS_CACHE_STATIC[name] = {}
        if layer_idx not in ANALYSIS_CACHE_STATIC[name]:
            ANALYSIS_CACHE_STATIC[name][layer_idx] = {}
        pow_value = torch.pow(value.abs(), p)  # take the abs first
        ANALYSIS_CACHE_STATIC[name][layer_idx][param_name] = (pow_value.min().cpu(), pow_value.mean().cpu(), pow_value.max().cpu())  # calculate the min, mean and max


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
