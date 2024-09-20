import torch
import numpy as np

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from precompute import (
    Hook,
    HookVariableNames,
    HookedGPTNeoXForCausalLM,
    HookedOPTForCausalLM,
    PrecomputeContext,
    write_artifact,
    offload,
)

MODEL_CLS = {
    'gpt-neox': HookedGPTNeoXForCausalLM,
    'opt': HookedOPTForCausalLM,
}

TOKENS_PER_STEP = 2097152
def get_pythia_steps():
    out = [0]
    for i in range(10):
        out.append(2**i)

    for i in range(1000, 144000, 1000):
        out.append(i)
    return out
pythia_steps = get_pythia_steps()

model_name = 'facebook/opt-125m'
model_type = 'opt'
config = AutoConfig.from_pretrained(model_name)
if model_type == 'gpt-neox':
    # AMD nans with torch.nn.functional.scaled_dot_product_attention
    config._attn_implementation = 'eager'

offload = False

raw_datasets = load_dataset('c4', 'en', streaming=True, trust_remote_code=True)
context_length = 2048
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)

inputs = torch.stack([torch.tensor(i['input_ids']) for i in tokenized_datasets["validation"].take(8)]).to('cuda')

# [b, n, d]
def log_post_tok_embeddings(x, ctx):
    ctx.context['post-tok-embeddings'] = x.cpu()
    return x

# [b, n, d]
def log_post_pos_embeddings(x, ctx):
    ctx.context['post-pos-embeddings'] = x.cpu()
    return x

def log_attn_residual(x, ctx):
    if 'attn-residual' not in ctx.context:
        ctx.context['attn-residual'] = torch.zeros(ctx.config.num_hidden_layers, *x.shape, device='cpu')
    ctx.context['attn-residual'][ctx.context['layer']] = x.cpu()
    return x

pre_mlp_filter_patch = 'high-pass'
cutoffs = [1, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0]
def patch_pre_mlp(x, ctx):
    if ctx.context['layer'] != 6:
        return x

    print(f'pre filter norm: {torch.norm(x, dim=-1).mean()}')
    x = x.float()
    fft = torch.fft.fft(x, dim=1)
    freqs = torch.fft.fftfreq(x.shape[1])

    if pre_mlp_filter_patch == 'high-pass':
        # high pass filter
        # 5.15
        cutoff_freq = cutoffs[ctx.context['cutoff']]
        # mask = torch.where(torch.abs(freqs) >= cutoff_freq, 1, 0).to('cuda').unsqueeze(0).unsqueeze(-1)
        mask = torch.where(torch.abs(freqs) >= cutoff_freq, 1, 0).to('cuda').unsqueeze(0).unsqueeze(-1)
        # mask = torch.clamp(mask + torch.where(torch.abs(freqs) <= 0.0005, 1, 0).to('cuda').unsqueeze(0).unsqueeze(-1), 0, 1)
        low = torch.zeros_like(mask)
        low[:, 0] = 1
        # mask = torch.clamp(mask + low, 0, 1)
    else:
        # low pass filter
        cutoff_freq = cutoffs[ctx.context['cutoff']]
        lower_cutoff = cutoffs[-1 * (ctx.context['cutoff'] + 1)]
        mask = torch.where(torch.abs(freqs) <= cutoff_freq, 1, 0).to('cuda').unsqueeze(0).unsqueeze(-1)

        if pre_mlp_filter_patch == 'mid-pass':
            # mid pass
            mask = mask * torch.where(torch.abs(freqs) >= lower_cutoff, 1, 0).to('cuda').unsqueeze(0).unsqueeze(-1)
        elif pre_mlp_filter_patch == 'out-pass':
            # out pass
            mask = torch.clamp(mask + torch.where(torch.abs(freqs) >= lower_cutoff, 1, 0).to('cuda').unsqueeze(0).unsqueeze(-1), 0, 1)

    filtered_fft = fft * mask

    # inverse fft
    filtered = torch.fft.ifft(filtered_fft, dim=1).real
    print(f'post filter norm: {torch.norm(filtered, dim=-1).mean()}')

    noise = torch.randn_like(filtered)
    print(f'noise norm: {torch.norm(noise, dim=-1).mean()}')

    # return noise.half()
    return filtered.half()

# HookVariableNames.POST_ACT
def log_mlp_sparsity(x, ctx):
    l = ctx.context['layer']
    if 'tok_500_activated_neurons' not in ctx.context:
        ctx.context['tok_500_activated_neurons'] = torch.zeros(ctx.config.num_hidden_layers, device='cpu')
        ctx.context['mean_activated_neurons'] = torch.zeros(ctx.config.num_hidden_layers, device='cpu')
        ctx.context['tok_50_activated_neurons'] = torch.zeros(ctx.config.num_hidden_layers, device='cpu')
        ctx.context['in_context_sparsity'] = torch.zeros(ctx.config.num_hidden_layers, device='cpu')
        ctx.context['499-500_neuron_diff'] = torch.zeros(ctx.config.num_hidden_layers, device='cpu')
        ctx.context['49-50_activated_neurons'] = torch.zeros(ctx.config.num_hidden_layers, device='cpu')
        ctx.context['499-500_neuron_overlap_pct'] = torch.zeros(ctx.config.num_hidden_layers, device='cpu')
        ctx.context['49-50_neuron_overlap_pct'] = torch.zeros(ctx.config.num_hidden_layers, device='cpu')
        ctx.context['in_context_sparsity_diff'] = torch.zeros(ctx.config.num_hidden_layers, device='cpu')

    activated_neurons = torch.count_nonzero(x, dim=-1) / ctx.config.ffn_dim
    ctx.context["tok_500_activated_neurons"][l] = activated_neurons[:, 500].mean().item()
    ctx.context["mean_activated_neurons"][l] = activated_neurons[:, 12:].mean().item()
    ctx.context["tok_50_activated_neurons"][l] = activated_neurons[:, 50].mean().item()
    ctx.context['in_context_sparsity'][l] = (activated_neurons[:, 500] - activated_neurons[:, 50]).mean().item()

    def get_diff_size(reshaped, idx1):
        active = reshaped[:, idx1] > 0
        diff = active.to(torch.bool) & ~(reshaped[:, idx1 - 1] > 0).to(torch.bool)
        intersection = active.to(torch.bool) & (reshaped[:, idx1 - 1] > 0).to(torch.bool)
        return intersection.sum(dim=-1) / active.sum(dim=-1), diff.sum(dim=-1) / diff.shape[1]

    late_intersection_size, late_diff_size = get_diff_size(x, 500)
    early_intersection_size, early_diff_size = get_diff_size(x, 50)
    ctx.context["499-500_neuron_overlap_pct"][l] = late_intersection_size.float().mean().item()
    ctx.context["49-50_neuron_overlap_pct"][l] = early_intersection_size.float().mean().item()
    ctx.context["499-500_neuron_diff"][l] = late_diff_size.mean().item()
    ctx.context["49-50_activated_neurons"][l] = early_diff_size.mean().item()
    ctx.context['in_context_sparsity_diff'][l] = (late_diff_size - early_diff_size).mean().item()

    return x

# [b, n, d]
def log_mlp_residual(x, ctx):
    if 'mlp-residual' not in ctx.context:
        ctx.context['mlp-residual'] = torch.zeros(ctx.config.num_hidden_layers, *x.shape, device='cpu')
    ctx.context['mlp-residual'][ctx.context['layer']] = x.cpu()
    return x

def log_post_mlp(x, ctx):
    """
    Log post-MLP hidden states for each layer.

    x     : [b, n, d]
    ctx   : PrecomputeContext

    return: [b, n, d]
    """
    if 'post-mlp' not in ctx.context:
        ctx.context['post-mlp'] = torch.zeros(ctx.config.num_hidden_layers, *x.shape, device='cpu')
    ctx.context['post-mlp'][ctx.context['layer']] = x.cpu()
    return x

# [b, n, d]
def log_post_final_layer_norm(x, ctx):
    ctx.context['post-final-layer-norm'] = x.cpu().float()
    return x

def log_in_context_learning_score(unreduced_loss, ctx):
    ctx.context['in-context-learning-score'] = -1 * (unreduced_loss[:, 500] - unreduced_loss[:, 50]).mean().item()
    return unreduced_loss

def log_loss(unreduced_loss, ctx):
    ctx.context['loss'] = unreduced_loss.mean().item()
    return unreduced_loss

hooks = [
    # Hook(HookVariableNames.POST_TOK_EMBEDDINGS, log_post_tok_embeddings),
    # Hook(HookVariableNames.POST_POS_EMBEDDINGS, log_post_pos_embeddings),
    # Hook(HookVariableNames.ATTN_RESIDUAL, log_attn_residual),
    # Hook(HookVariableNames.MLP_RESIDUAL, log_mlp_residual),
    # Hook(HookVariableNames.POST_MLP, log_post_mlp),
    # Hook(HookVariableNames.POST_FINAL_LAYER_NORM, log_post_final_layer_norm),
    # Hook(HookVariableNames.UNREDUCED_LOSS, log_in_context_learning_score),
    # Hook(HookVariableNames.UNREDUCED_LOSS, log_loss),
    Hook(HookVariableNames.PRE_MLP, patch_pre_mlp),
    Hook(HookVariableNames.POST_ACT, log_mlp_sparsity),
]
pctx = PrecomputeContext(config, hooks=hooks)

# 6B tokens
# pythia_steps = pythia_steps[:14]
# pythia_steps = pythia_steps[:18]

tok_500_activated_neurons = torch.zeros((len(cutoffs), config.num_hidden_layers))
mean_activated_neurons = torch.zeros((len(cutoffs), config.num_hidden_layers))
tok_50_activated_neurons = torch.zeros((len(cutoffs), config.num_hidden_layers))
in_context_sparsity = torch.zeros((len(cutoffs), config.num_hidden_layers))
neuron_diff_499_500 = torch.zeros((len(cutoffs), config.num_hidden_layers))
neuron_diff_49_50 = torch.zeros((len(cutoffs), config.num_hidden_layers))
in_context_sparsity_diff = torch.zeros((len(cutoffs), config.num_hidden_layers))

neuron_overlap_pct_499_500 = torch.zeros((len(cutoffs), config.num_hidden_layers))
neuron_overlap_pct_49_50 = torch.zeros((len(cutoffs), config.num_hidden_layers))

config = AutoConfig.from_pretrained(model_name)
model = MODEL_CLS[model_type].from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float16,
)
if offload:
    model = offload(model)
else:
    model.to('cuda')

for j in range(len(cutoffs)):
    pctx.context['cutoff'] = j
    with torch.no_grad():
        output = model(inputs, labels=inputs, pctx=pctx)

    print(f'cutoff {cutoffs[j]} loss: {output.loss.item()}')

    tok_500_activated_neurons[j] = pctx.context['tok_500_activated_neurons']
    mean_activated_neurons[j] = pctx.context['mean_activated_neurons']
    tok_50_activated_neurons[j] = pctx.context['tok_50_activated_neurons']
    in_context_sparsity[j] = pctx.context['in_context_sparsity']
    neuron_diff_499_500[j] = pctx.context['499-500_neuron_diff']
    neuron_diff_49_50[j] = pctx.context['49-50_activated_neurons']
    in_context_sparsity_diff[j] = pctx.context['in_context_sparsity_diff']

    neuron_overlap_pct_499_500[j] = pctx.context['499-500_neuron_overlap_pct']
    neuron_overlap_pct_49_50[j] = pctx.context['49-50_neuron_overlap_pct']

data = [
    ('tok_500_activated_neurons', tok_500_activated_neurons),
    ('mean_activated_neurons', mean_activated_neurons),
    ('tok_50_activated_neurons', tok_50_activated_neurons),
    ('in_context_sparsity', in_context_sparsity),
    ('neuron_diff_499_500', neuron_diff_499_500),
    ('neuron_diff_49_50', neuron_diff_49_50),
    ('in_context_sparsity_diff', in_context_sparsity_diff),
    ('neuron_overlap_pct_499_500', neuron_overlap_pct_499_500),
    ('neuron_overlap_pct_49_50', neuron_overlap_pct_49_50),
]

artifact_model_name = model_name.replace('/', '--').replace('.', 'dot')
for name, d in data:
    artifact_metadata = {
        'name': f'optspectrum-{artifact_model_name}-{name}-{pre_mlp_filter_patch}',
        'visualization': 'linear-line',
        'y_name': 'Value',
        'description': '',
        'config': model.config.to_dict(),
    }
    columns = {'Layer': np.array(list(range(config.num_hidden_layers)))}
    for i, cutoff in enumerate(cutoffs):
        columns[f'{pre_mlp_filter_patch}: {cutoff}'] = d[i].numpy()

    write_artifact(artifact_metadata, columns)