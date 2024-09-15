import torch

from transformers import OPTForCausalLM

from precompute import Hook, HookVariableNames, HookedOPTForCausalLM, PrecomputeContext

def test_no_hooks(hooked_model, model, inputs):
    print(f'test_no_hooks')
    precompute_context = PrecomputeContext(hooked_model.config)
    hooked_output = hooked_model(inputs, precompute_context=precompute_context)
    output = model(inputs)

    assert torch.allclose(hooked_output.logits, output.logits)

def test_mutation_hook(hooked_model, model, inputs):
    print(f'test_mutation_hook')
    precompute_context = PrecomputeContext(hooked_model.config)

    hooks = [
        Hook(HookVariableNames.MLP_RESIDUAL, lambda x, ctx: x + 1),
    ]
    precompute_context.add_hooks(hooks)

    hooked_output = hooked_model(inputs, precompute_context=precompute_context)
    output = model(inputs)

    assert not torch.allclose(hooked_output.logits, output.logits)

def test_logging_hook(hooked_model, model, inputs):
    print(f'test_logging_hook')
    precompute_context = PrecomputeContext(hooked_model.config)

    # [b, n, d]
    def log_post_mlp(x, ctx):
        if 'post-mlp' not in ctx.context:
            ctx.context['post-mlp'] = torch.zeros(ctx.config.num_hidden_layers, *x.shape, device=x.device)
        ctx.context['post-mlp'][ctx.context['layer']] = x
        return x

    hooks = [
        Hook(HookVariableNames.POST_MLP, log_post_mlp),
    ]
    precompute_context.add_hooks(hooks)

    hooked_output = hooked_model(inputs, precompute_context=precompute_context)
    output = model(inputs)

    # hook shouldn't mutate
    assert torch.allclose(hooked_output.logits, output.logits)

    for i in range(hooked_model.config.num_hidden_layers):
        # hook should add data for each layer
        assert torch.sum(torch.abs(precompute_context.context['post-mlp'][i])) > 1e-4

model_name = 'facebook/opt-125m'

hooked_model = HookedOPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')
model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')

inputs = torch.load('opt-30b-c4-inputs.pt').to('cuda')

test_no_hooks(hooked_model, model, inputs)
test_mutation_hook(hooked_model, model, inputs)
test_logging_hook(hooked_model, model, inputs)
