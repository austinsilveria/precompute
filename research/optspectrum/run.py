import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from diffusers import DDPMScheduler

from precompute import (
    Hook,
    HookVariableNames,
    HookedOPTForCausalLM,
    PrecomputeContext,
    write_artifact,
)

model_name = 'facebook/opt-125m'
model = HookedOPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')

raw_datasets = load_dataset('c4', 'en', streaming=True)
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

pctx = PrecomputeContext(model.config)

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
    ctx.context['post-final-layer-norm'] = x.cpu()
    return x

hooks = [
    Hook(HookVariableNames.POST_TOK_EMBEDDINGS, log_post_tok_embeddings),
    Hook(HookVariableNames.POST_POS_EMBEDDINGS, log_post_pos_embeddings),
    Hook(HookVariableNames.ATTN_RESIDUAL, log_attn_residual),
    Hook(HookVariableNames.MLP_RESIDUAL, log_mlp_residual),
    Hook(HookVariableNames.POST_MLP, log_post_mlp),
    Hook(HookVariableNames.POST_FINAL_LAYER_NORM, log_post_final_layer_norm),
]
pctx.add_hooks(hooks)

with torch.no_grad():
    output = model(inputs, pctx=pctx)

# [l, b, n, d]
def compute_power_spectrum(tensor):
    print(f'ps input dims: {tensor.shape}')
    # fourier transform along sequence dimension for each latent dimension independently
    fft_result = torch.fft.rfft(tensor, dim=2)
    print(f'fft shape: {fft_result.shape}')
    power_spectrum = torch.abs(fft_result)**2
    # average across batch and latent dimensions
    # [l, d]
    mean_power_spectrum = torch.mean(power_spectrum, dim=(1, 3))

    frequencies = torch.fft.rfftfreq(tensor.shape[2])

    return frequencies, mean_power_spectrum

# Skip first token
hidden_states = torch.cat([
    pctx.context['post-tok-embeddings'].unsqueeze(0),
    pctx.context['post-pos-embeddings'].unsqueeze(0),
    pctx.context['post-mlp'],
])[:, :, 1:]
hidden_states_freqs, hidden_states_mps = compute_power_spectrum(hidden_states)
mlp_residuals_freqs, mlp_residuals_mps = compute_power_spectrum(pctx.context['mlp-residual'][:, :, 1:])
attn_residuals_freqs, attn_residuals_mps = compute_power_spectrum(pctx.context['attn-residual'][:, :, 1:])

artifact_metadata = {
    'name': 'optspectrum-125m-hidden-states' + '-seq-len-' + str(context_length),
    'visualization': 'line',
    'y_name': 'Power',
    'description': 'Power spectrum along sequence dimension of hidden states for OPT-125M model, averaged across batch and latent dimensions.',
    'config': model.config.to_dict(),
}
columns = {'Frequency': hidden_states_freqs.numpy()}
columns['Tok Embeddings'] = hidden_states_mps[0].numpy()
columns['Tok + Pos Embeddings'] = hidden_states_mps[1].numpy()
for i in range(2, hidden_states_mps.shape[0]):
    columns[f'Layer {i - 1}'] = hidden_states_mps[i].numpy()

write_artifact(artifact_metadata, columns)


artifact_metadata = {
    'name': 'optspectrum-125m-mlp-residuals' + '-seq-len-' + str(context_length),
    'visualization': 'line',
    'y_name': 'Power',
    'description': 'Power spectrum along sequence dimension of MLP residuals for OPT-125M model, averaged across batch and latent dimensions.',
    'config': model.config.to_dict(),
}
columns = {'Frequency': mlp_residuals_freqs.numpy()}
for i in range(mlp_residuals_mps.shape[0]):
    columns[f'Layer {i + 1}'] = mlp_residuals_mps[i].numpy()

write_artifact(artifact_metadata, columns)

artifact_metadata = {
    'name': 'optspectrum-125m-attn-residuals' + '-seq-len-' + str(context_length),
    'visualization': 'line',
    'y_name': 'Power',
    'description': 'Power spectrum along sequence dimension of attention residuals for OPT-125M model, averaged across batch and latent dimensions.',
    'config': model.config.to_dict(),
}
columns = {'Frequency': attn_residuals_freqs.numpy()}
for i in range(attn_residuals_mps.shape[0]):
    columns[f'Layer {i + 1}'] = attn_residuals_mps[i].numpy()

write_artifact(artifact_metadata, columns)


# Forward diffusion
final_hidden_states = (pctx.context['post-final-layer-norm'][:, 1:, :]).float()
num_steps = 1000
beta_end = 0.02

noise_scheduler = DDPMScheduler(num_train_timesteps=num_steps, beta_schedule='linear', beta_end=beta_end)
timesteps = torch.tensor([500], device=hidden_states.device).repeat(final_hidden_states.shape[0])

noise = torch.randn_like(final_hidden_states, device=hidden_states.device)
scheduled_noisy_hidden_states = noise_scheduler.add_noise(final_hidden_states, noise, timesteps)
timesteps = torch.tensor([250], device=hidden_states.device).repeat(final_hidden_states.shape[0])
quarter_noise = noise_scheduler.add_noise(final_hidden_states, noise, timesteps)
added_noise = final_hidden_states + noise

hidden_states_freqs, hidden_states_mps = compute_power_spectrum(final_hidden_states.unsqueeze(0))
noisy_hidden_states_freqs, noisy_hidden_states_mps = compute_power_spectrum(scheduled_noisy_hidden_states.unsqueeze(0))
_, qhs = compute_power_spectrum(quarter_noise.unsqueeze(0))
added_noise_hidden_states_freqs, added_noise_hidden_states_mps = compute_power_spectrum(added_noise.unsqueeze(0))
noise_hidden_states_freqs, noise_hidden_states_mps = compute_power_spectrum(noise.unsqueeze(0))

artifact_metadata = {
    'name': 'optspectrum-125m-final-states-noise-comparison' + '-seq-len-' + str(context_length),
    'visualization': 'line',
    'y_name': 'Power',
    'description': 'Power spectrum along sequence dimension of final hidden states for an untrained OPT-125M model, averaged across batch and latent dimensions. Compared against quarter/half noise from DDPM scheduler, directly added noise, and pure noise.',
    'config': model.config.to_dict(),
}
columns = {'Frequency': hidden_states_freqs.numpy()}
columns['Final Hidden States'] = hidden_states_mps[0].numpy()
columns['Quarter Noise'] = qhs[0].numpy()
columns['Half Noise'] = noisy_hidden_states_mps[0].numpy()
columns['Added Noise'] = added_noise_hidden_states_mps[0].numpy()
columns['Noise'] = noise_hidden_states_mps[0].numpy()
write_artifact(artifact_metadata, columns)

# Untrained
config = model.config
config.torch_dtype = torch.float16
model = HookedOPTForCausalLM(config=config).to('cuda')
pctx = PrecomputeContext(model.config, hooks=hooks)

with torch.no_grad():
    output = model(inputs, pctx=pctx)

# Skip first token
hidden_states = torch.cat([
    pctx.context['post-tok-embeddings'].unsqueeze(0),
    pctx.context['post-pos-embeddings'].unsqueeze(0),
    pctx.context['post-mlp'],
])[:, :, 1:]
hidden_states_freqs, hidden_states_mps = compute_power_spectrum(hidden_states)
mlp_residuals_freqs, mlp_residuals_mps = compute_power_spectrum(pctx.context['mlp-residual'][:, :, 1:])
attn_residuals_freqs, attn_residuals_mps = compute_power_spectrum(pctx.context['attn-residual'][:, :, 1:])

artifact_metadata = {
    'name': 'optspectrum-untrained-125m-hidden-states' + '-seq-len-' + str(context_length),
    'visualization': 'line',
    'y_name': 'Power',
    'description': 'Power spectrum along sequence dimension of hidden states for an untrained OPT-125M model, averaged across batch and latent dimensions.',
    'config': model.config.to_dict(),
}
columns = {'Frequency': hidden_states_freqs.numpy()}
columns['Tok Embeddings'] = hidden_states_mps[0].numpy()
columns['Tok + Pos Embeddings'] = hidden_states_mps[1].numpy()
for i in range(2, hidden_states_mps.shape[0]):
    columns[f'Layer {i - 1}'] = hidden_states_mps[i].numpy()

write_artifact(artifact_metadata, columns)


artifact_metadata = {
    'name': 'optspectrum-untrained-125m-mlp-residuals' + '-seq-len-' + str(context_length),
    'visualization': 'line',
    'y_name': 'Power',
    'description': 'Power spectrum along sequence dimension of MLP residuals for an untrained OPT-125M model, averaged across batch and latent dimensions.',
    'config': model.config.to_dict(),
}
columns = {'Frequency': mlp_residuals_freqs.numpy()}
for i in range(mlp_residuals_mps.shape[0]):
    columns[f'Layer {i + 1}'] = mlp_residuals_mps[i].numpy()

write_artifact(artifact_metadata, columns)

artifact_metadata = {
    'name': 'optspectrum-untrained-125m-attn-residuals' + '-seq-len-' + str(context_length),
    'visualization': 'line',
    'y_name': 'Power',
    'description': 'Power spectrum along sequence dimension of attention residuals for an untrained OPT-125M model, averaged across batch and latent dimensions.',
    'config': model.config.to_dict(),
}
columns = {'Frequency': attn_residuals_freqs.numpy()}
for i in range(attn_residuals_mps.shape[0]):
    columns[f'Layer {i + 1}'] = attn_residuals_mps[i].numpy()

write_artifact(artifact_metadata, columns)
