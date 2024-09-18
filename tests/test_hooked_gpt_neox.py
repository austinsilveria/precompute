import torch

from transformers import AutoConfig, GPTNeoXForCausalLM

from precompute import Hook, HookVariableNames, HookedGPTNeoXForCausalLM, PrecomputeContext

def test_no_hooks(hooked_model, model, inputs):
    print(f'test_no_hooks')
    pctx = PrecomputeContext(hooked_model.config)
    hooked_output = hooked_model(inputs, pctx=pctx)
    output = model(inputs)

    assert torch.allclose(hooked_output.logits, output.logits)

def test_mutation_hook(hooked_model, model, inputs):
    print(f'test_mutation_hook')
    pctx = PrecomputeContext(hooked_model.config)

    # Tests mutation and output shapes
    plus_one = lambda x, ctx: x + 1
    hooks = [
        Hook(HookVariableNames.POST_TOK_EMBEDDINGS, plus_one),
        Hook(HookVariableNames.POST_POS_EMBEDDINGS, plus_one),
        Hook(HookVariableNames.PRE_ATTN, plus_one),
        Hook(HookVariableNames.QUERY_STATES, plus_one),
        Hook(HookVariableNames.KEY_STATES, plus_one),
        Hook(HookVariableNames.VALUE_STATES, plus_one),
        Hook(HookVariableNames.ATTN_PROBS, plus_one),
        Hook(HookVariableNames.ATTN_WEIGHTED_VALUES, plus_one),
        Hook(HookVariableNames.ATTN_RESIDUAL, plus_one),
        Hook(HookVariableNames.POST_ATTN, plus_one),
        Hook(HookVariableNames.PRE_MLP, plus_one),
        Hook(HookVariableNames.POST_FC1, plus_one),
        Hook(HookVariableNames.POST_ACT, plus_one),
        Hook(HookVariableNames.MLP_RESIDUAL, plus_one),
        Hook(HookVariableNames.POST_MLP, plus_one),
        Hook(HookVariableNames.POST_FINAL_LAYER_NORM, plus_one),
        Hook(HookVariableNames.LOGITS, plus_one),
    ]
    pctx.add_hooks(hooks)

    hooked_output = hooked_model(inputs, pctx=pctx)
    output = model(inputs)

    assert not torch.allclose(hooked_output.logits, output.logits)

def test_logging_hook(hooked_model, model, inputs):
    print(f'test_logging_hook')
    pctx = PrecomputeContext(hooked_model.config)

    # [b, n, d]
    def log_post_mlp(x, ctx):
        if 'post-mlp' not in ctx.context:
            ctx.context['post-mlp'] = torch.zeros(ctx.config.num_hidden_layers, *x.shape, device=x.device)
        ctx.context['post-mlp'][ctx.context['layer']] = x
        return x

    hooks = [
        Hook(HookVariableNames.POST_MLP, log_post_mlp),
    ]
    pctx.add_hooks(hooks)

    hooked_output = hooked_model(inputs, pctx=pctx)
    output = model(inputs)

    # hook shouldn't mutate
    assert torch.allclose(hooked_output.logits, output.logits)

    for i in range(hooked_model.config.num_hidden_layers):
        # hook should add data for each layer
        assert torch.sum(torch.abs(pctx.context['post-mlp'][i])) > 1e-4

def test_shapes(hooked_model, inputs):
    print(f'test_shapes')

    b, n = inputs.shape[:2]
    d = hooked_model.config.hidden_size
    h = hooked_model.config.num_attention_heads
    c = d // h
    f = hooked_model.config.intermediate_size
    v = hooked_model.config.vocab_size

    s = lambda x: torch.Size(x)

    hooks = [
        Hook(HookVariableNames.POST_TOK_EMBEDDINGS, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, n, d]))),
        Hook(HookVariableNames.PRE_ATTN, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, n, d]))),
        Hook(HookVariableNames.QUERY_STATES, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, h, n, c]))),
        Hook(HookVariableNames.KEY_STATES, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, h, n, c]))),
        Hook(HookVariableNames.VALUE_STATES, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, h, n, c]))),
        Hook(HookVariableNames.ATTN_PROBS, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, h, n, n]))),
        Hook(HookVariableNames.ATTN_WEIGHTED_VALUES, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, h, n, c]))),
        Hook(HookVariableNames.ATTN_RESIDUAL, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, n, d]))),
        Hook(HookVariableNames.POST_ATTN, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, n, d]))),
        Hook(HookVariableNames.PRE_MLP, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, n, d]))),
        Hook(HookVariableNames.POST_FC1, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, n, f]))),
        Hook(HookVariableNames.POST_ACT, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, n, f]))),
        Hook(HookVariableNames.MLP_RESIDUAL, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, n, d]))),
        Hook(HookVariableNames.POST_MLP, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, n, d]))),
        Hook(HookVariableNames.POST_FINAL_LAYER_NORM, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, n, d]))),
        Hook(HookVariableNames.LOGITS, lambda x, ctx: ctx.context['shape_checks'].append(x.shape == s([b, n, v]))),
    ]

    pctx = PrecomputeContext(hooked_model.config)
    pctx.context['shape_checks'] = []
    pctx.add_hooks(hooks)

    hooked_output = hooked_model(inputs, pctx=pctx)

    print(f'pctx.context[shape_checks]: {pctx.context["shape_checks"]}')
    assert all(pctx.context['shape_checks'])

model_name = 'EleutherAI/pythia-70m-deduped'

config = AutoConfig.from_pretrained(model_name)
# AMD nans with torch.nn.functional.scaled_dot_product_attention
config._attn_implementation = 'eager'

hooked_model = HookedGPTNeoXForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16).to('cuda')
model = GPTNeoXForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16).to('cuda')

inputs = torch.load('opt-30b-c4-inputs.pt').to('cuda')

test_no_hooks(hooked_model, model, inputs)
test_mutation_hook(hooked_model, model, inputs)
test_logging_hook(hooked_model, model, inputs)
test_shapes(hooked_model, inputs)
