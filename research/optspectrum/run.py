import torch

from precompute import (
    Hook,
    HookVariableNames,
    HookedOPTForCausalLM,
    PrecomputeContext,
    write_artifact,
)

model_name = 'facebook/opt-125m'
model = HookedOPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')

inputs = torch.load('opt-30b-c4-inputs.pt').to('cuda')

pctx = PrecomputeContext(model.config)

# [b, n, d]
def log_post_tok_embeddings(x, ctx):
    ctx.context['post-tok-embeddings'] = x.cpu()
    return x

# [b, n, d]
def log_post_pos_embeddings(x, ctx):
    ctx.context['post-pos-embeddings'] = x.cpu()
    return x

# [b, n, d]
def log_mlp_residual(x, ctx):
    if 'mlp-residual' not in ctx.context:
        ctx.context['mlp-residual'] = torch.zeros(ctx.config.num_hidden_layers, *x.shape, device='cpu')
    ctx.context['mlp-residual'][ctx.context['layer']] = x.cpu()
    return x

# [b, n, d]
def log_post_mlp(x, ctx):
    if 'post-mlp' not in ctx.context:
        ctx.context['post-mlp'] = torch.zeros(ctx.config.num_hidden_layers, *x.shape, device='cpu')
    ctx.context['post-mlp'][ctx.context['layer']] = x.cpu()
    return x

hooks = [
    Hook(HookVariableNames.POST_TOK_EMBEDDINGS, log_post_tok_embeddings),
    Hook(HookVariableNames.POST_POS_EMBEDDINGS, log_post_pos_embeddings),
    Hook(HookVariableNames.MLP_RESIDUAL, log_mlp_residual),
    Hook(HookVariableNames.POST_MLP, log_post_mlp),
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

artifact_metadata = {
    'name': 'optspectrum-125m-hidden-states',
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
    'name': 'optspectrum-125m-mlp-residuals',
    'visualization': 'line',
    'y_name': 'Power',
    'description': 'Power spectrum along sequence dimension of MLP residuals for OPT-125M model, averaged across batch and latent dimensions.',
    'config': model.config.to_dict(),
}
columns = {'Frequency': mlp_residuals_freqs.numpy()}
for i in range(mlp_residuals_mps.shape[0]):
    columns[f'Layer {i + 1}'] = mlp_residuals_mps[i].numpy()

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

artifact_metadata = {
    'name': 'optspectrum-untrained-125m-hidden-states',
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
    'name': 'optspectrum-untrained-125m-mlp-residuals',
    'visualization': 'line',
    'y_name': 'Power',
    'description': 'Power spectrum along sequence dimension of MLP residuals for an untrained OPT-125M model, averaged across batch and latent dimensions.',
    'config': model.config.to_dict(),
}
columns = {'Frequency': mlp_residuals_freqs.numpy()}
for i in range(mlp_residuals_mps.shape[0]):
    columns[f'Layer {i + 1}'] = mlp_residuals_mps[i].numpy()

write_artifact(artifact_metadata, columns)