from collections.abc import Callable
from typing import Any, List

import torch

class HookVariableNames:
    # b - batch size
    # n - sequence length
    # d - hidden dimension
    # h - number of heads
    # c - head dimension
    # f - feedforward dimension
    # v - vocabulary size

    # [b, n, d]
    POST_TOK_EMBEDDINGS = "post-tok-embeddings"
    # [b, n, d]
    POST_POS_EMBEDDINGS = "post-pos-embeddings"

    # [b, n, d]
    PRE_ATTN = "pre-attn"
    # [b, h, n c]
    QUERY_STATES = "query-states"
    # [b, h, n, c]
    KEY_STATES = "key-states"
    # [b, h, n, c]
    VALUE_STATES = "value-states"
    # [b, h, n, n]
    ATTN_PROBS = "attn-probs"
    # [b, h, n, c]
    ATTN_WEIGHTED_VALUES = "attn-weighted-values"
    # [b, n, d]
    ATTN_RESIDUAL = "attn-residual"
    # [b, n, d]
    POST_ATTN = "post-attn"

    # [b, n, d]
    PRE_MLP = "pre-mlp"
    # [b, n, f]
    POST_FC1 = "post-fc1"
    # [b, n, f]
    POST_ACT = "post-act"
    # [b, n, d]
    MLP_RESIDUAL = "mlp-residual"
    # [b, n, d]
    POST_MLP = "post-mlp"

    # [b, n, d]
    POST_FINAL_LAYER_NORM = "post-final-layer-norm"

    # [b, n, v]
    LOGITS = "logits"

    ALL_VARIABLE_NAMES = [
        POST_TOK_EMBEDDINGS,
        POST_POS_EMBEDDINGS,
        PRE_ATTN,
        QUERY_STATES,
        KEY_STATES,
        VALUE_STATES,
        ATTN_PROBS,
        ATTN_WEIGHTED_VALUES,
        ATTN_RESIDUAL,
        POST_ATTN,
        PRE_MLP,
        POST_FC1,
        POST_ACT,
        MLP_RESIDUAL,
        POST_MLP,
        POST_FINAL_LAYER_NORM,
        LOGITS,
    ]

class PrecomputeContext:
    def __init__(self, config) -> None:
        self.config = config

        # Immutably logged context
        self.training_log = {}
        # Unlogged context
        self.context = {}

        # Hook functions by hook variable name
        self.hooks = { vn: [] for vn in HookVariableNames.ALL_VARIABLE_NAMES }
        # Hook variable values by hook variable name
        # self.hook_variables = {}

    def hook_tensor(self, variable_name: str, value: torch.Tensor) -> None:
        # if 'layer' in self.unlogged_context:
        #     name = f'{name}-layer-{self.unlogged_context["layer"]}'

        # Call matched hooks
        outputs = [hook_fn(value, self) for hook_fn in self.hooks[variable_name]]

        if len(outputs) > 0:
            # Mutate the tensor. Return last output.
            return outputs[-1]

        # Pass through
        return value

    def add_hooks(self, hooks) -> None:
        for hook in hooks:
            self.hooks[hook.hook_variable_name].append(hook.hook_fn)

class Hook:
    def __init__(self, hook_variable_name: str, hook_fn: Callable[[torch.Tensor, PrecomputeContext], torch.Tensor]):
        self.hook_variable_name = hook_variable_name
        self.hook_fn = hook_fn

    def __call__(self, tensor: torch.Tensor, context: PrecomputeContext) -> Any:
        return self.hook_fn(tensor, context)
