import torch
import numpy as np

from sklearn.metrics import r2_score

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

model_name = 'EleutherAI/pythia-160m-deduped'
model_type = 'gpt-neox'
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

attn_residuals_filter_patch = None
cutoffs = [1, 0.1, 0.05, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0]
def log_attn_residual(x, ctx):
    if 'attn-residual' not in ctx.context:
        ctx.context['attn-residual'] = torch.zeros(ctx.config.num_hidden_layers, *x.shape, device='cpu')
    if attn_residuals_filter_patch is not None:
        # fft
        fft = torch.fft.fft(x, dim=1)
        print(f'fft: {fft.shape}')
        freqs = torch.fft.fftfreq(x.shape[1])

        if attn_residuals_filter_patch == 'high-pass':
            # high pass filter
            # 5.15
            cutoff_freq = cutoffs[ctx.context['cutoff']]
            mask = torch.where(torch.abs(freqs) >= cutoff_freq, 1, 0).to('cuda').unsqueeze(0).unsqueeze(-1)
        else:
            # low pass filter
            cutoff_freq = cutoffs[ctx.context['cutoff']]
            lower_cutoff = cutoffs[-1 * (ctx.context['cutoff'] + 1)]
            mask = torch.where(torch.abs(freqs) <= cutoff_freq, 1, 0).to('cuda').unsqueeze(0).unsqueeze(-1)

            if attn_residuals_filter_patch == 'mid-pass':
                # mid pass
                mask = mask * torch.where(torch.abs(freqs) >= lower_cutoff, 1, 0).to('cuda').unsqueeze(0).unsqueeze(-1)
            elif attn_residuals_filter_patch == 'out-pass':
                # out pass
                mask = torch.clamp(mask + torch.where(torch.abs(freqs) >= lower_cutoff, 1, 0).to('cuda').unsqueeze(0).unsqueeze(-1), 0, 1)

        print(f'mask: {mask.shape}')

        filtered_fft = fft * mask

        # inverse fft
        filtered = torch.fft.ifft(filtered_fft, dim=1).real
    else:
        filtered = x
    ctx.context['attn-residual'][ctx.context['layer']] = filtered.cpu()
    return filtered

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
    Hook(HookVariableNames.ATTN_RESIDUAL, log_attn_residual),
    Hook(HookVariableNames.MLP_RESIDUAL, log_mlp_residual),
    # Hook(HookVariableNames.POST_MLP, log_post_mlp),
    Hook(HookVariableNames.POST_FINAL_LAYER_NORM, log_post_final_layer_norm),
    Hook(HookVariableNames.UNREDUCED_LOSS, log_in_context_learning_score),
    Hook(HookVariableNames.UNREDUCED_LOSS, log_loss),
]
pctx = PrecomputeContext(config, hooks=hooks)

# [l, b, n, d]
def compute_power_spectrum(tensor):
    # fourier transform along sequence dimension for each latent dimension independently
    fft_result = torch.fft.rfft(tensor, dim=2)
    power_spectrum = torch.abs(fft_result)**2
    # average across batch and latent dimensions
    # [l, d]
    mean_power_spectrum = torch.mean(power_spectrum, dim=(1, 3))

    frequencies = torch.fft.rfftfreq(tensor.shape[2])

    return frequencies, mean_power_spectrum

# 6B tokens
# pythia_steps = pythia_steps[:14]
pythia_steps = pythia_steps[:18]

frequencies = torch.fft.rfftfreq(context_length-1)
mps_across_training = torch.empty((len(pythia_steps), frequencies.shape[0]), device='cpu')
mlp_residual_mps_across_training = torch.empty((len(pythia_steps), frequencies.shape[0]), device='cpu')
attn_residual_mps_across_training = torch.empty((len(pythia_steps), frequencies.shape[0]), device='cpu')
in_context_learning_score_across_training = torch.empty((len(pythia_steps), len(cutoffs)), device='cpu')
loss_across_training = torch.empty((len(pythia_steps), len(cutoffs)), device='cpu')

for i, step in enumerate(pythia_steps):
    print(f'step: {step}')
    config = AutoConfig.from_pretrained(model_name, revision=f'step{step}')
    config._attn_implementation = 'eager'
    model = MODEL_CLS[model_type].from_pretrained(
        model_name,
        config=config,
        revision=f'step{step}',
        torch_dtype=torch.float16,
        # cache_dir=f'./{model_name}/step{step}',
    )
    # config.torch_dtype = torch.float16
    # model = MODEL_CLS[model_type](config)
    if offload:
        model = offload(model)
    else:
        model.to('cuda')

    for j in range(len(cutoffs)):
        pctx.context['cutoff'] = j
        with torch.no_grad():
            output = model(inputs, labels=inputs, pctx=pctx)

        in_context_learning_score_across_training[i, j] = pctx.context['in-context-learning-score']
        loss_across_training[i, j] = pctx.context['loss']

        # Skip first token
        hidden_states_freqs, hidden_states_mps = compute_power_spectrum(pctx.context['post-final-layer-norm'][:, 1:, :].unsqueeze(0))
        mps_across_training[i] = hidden_states_mps[0].clone()

        hidden_states_freqs, hidden_states_mps = compute_power_spectrum(pctx.context['attn-residual'][-1, :, 1:, :].unsqueeze(0))
        attn_residual_mps_across_training[i] = hidden_states_mps[0].clone()

        hidden_states_freqs, hidden_states_mps = compute_power_spectrum(pctx.context['mlp-residual'][-1, :, 1:, :].unsqueeze(0))
        mlp_residual_mps_across_training[i] = hidden_states_mps[0].clone()

# print(f'mps_across_training: {mps_across_training}')
print(f'mps_across_training: {mps_across_training.shape}')

# log_frequencies = np.log(frequencies[1:].numpy())
# print(f'log_frequencies: {log_frequencies.shape}')
# equally_spaced_log_frequencies = np.linspace(log_frequencies[0], log_frequencies[-1], 1000)
# print(f'equally_spaced_log_frequencies: {equally_spaced_log_frequencies.shape}')
# log_mps = np.log(mps_across_training[:, 1:].numpy())
# print(f'log_mps: {log_mps.shape}')

# slopes = []
# linear_fits = []
# for i in range(log_mps.shape[0]):
#     interpolated_rapsd = np.interp(equally_spaced_log_frequencies, log_frequencies, log_mps[i])
#     # print(f'interpolated_rapsd: {interpolated_rapsd.shape}')

#     m, b = np.polyfit(equally_spaced_log_frequencies, interpolated_rapsd, 1)
#     slopes.append(m)
#     linear_fit = m * log_frequencies + b
#     linear_fits.append(linear_fit)
#     print()
#     print(f'm: {m}')
#     # print(f'b: {b}')
#     # print(f'linear_fit: {linear_fit}')
#     # print(f'linear_fit: {linear_fit.shape}')

#     print(f'r2: {r2_score(mps_across_training[i, 1:], np.exp(linear_fit))}')

# artifact_model_name = model_name.replace('/', '--').replace('.', 'dot')
# artifact_metadata = {
#     'name': f'optspectrum-{artifact_model_name}-power-law-fit-over-training',
#     'visualization': 'line',
#     'y_name': 'Power',
#     'description': 'Power spectrum along sequence dimension of hidden states for OPT-125M model, averaged across batch and latent dimensions.',
#     'config': model.config.to_dict(),
# }
# columns = {'Frequency': hidden_states_freqs[1:].numpy()}
# columns['Final Hidden States'] = mps_across_training[0, 1:].numpy()
# columns['Linear Fit'] = np.exp(linear_fits[0])

# write_artifact(artifact_metadata, columns)

def compute_linear_fit(x, y):
    log_x = np.log(x)
    log_y = np.log(y)
    
    equally_spaced_log_frequencies = np.linspace(log_x[0], log_x[-1], 1000)
    slopes = []
    r2s = []
    for i in range(y.shape[0]):
        interpolated_rapsd = np.interp(equally_spaced_log_frequencies, log_x, log_y[i])

        m, b = np.polyfit(equally_spaced_log_frequencies, interpolated_rapsd, 1)
        slopes.append(m)
        linear_fit = m * log_x + b

        r2s.append(r2_score(y[i], np.exp(linear_fit)))
    return np.array(r2s), np.array(slopes)

# final_hs_r2s, final_hs_slopes = compute_linear_fit(frequencies[1:].numpy(), mps_across_training[:, 1:].numpy())
# attn_residuals_r2s, attn_residuals_slopes = compute_linear_fit(frequencies[1:].numpy(), attn_residual_mps_across_training[:, 1:].numpy())
# mlp_residuals_r2s, mlp_residuals_slopes = compute_linear_fit(frequencies[1:].numpy(), mlp_residual_mps_across_training[:, 1:].numpy())

# artifact_model_name = model_name.replace('/', '--').replace('.', 'dot')
# artifact_metadata = {
#     'name': f'optspectrum-{artifact_model_name}-power-law-r2-over-training-fft-filter-{attn_residuals_filter_patch}',
#     'visualization': 'linear-line',
#     'y_name': 'R2<br>Score',
#     'description': 'Power spectrum along sequence dimension of hidden states for OPT-125M model, averaged across batch and latent dimensions.',
#     'config': model.config.to_dict(),
# }
# columns = {'Tokens': np.array([i * TOKENS_PER_STEP for i in pythia_steps])}
# # columns['R2'] = np.array([r2_score(mps_across_training[i, 1:], np.exp(linear_fits[i])) for i in range(len(linear_fits))])
# columns['R2'] = final_hs_r2s

# write_artifact(artifact_metadata, columns)

# artifact_model_name = model_name.replace('/', '--').replace('.', 'dot')
# artifact_metadata = {
#     'name': f'optspectrum-{artifact_model_name}-power-law-coefficient-over-training-fft-filter-{attn_residuals_filter_patch}',
#     'visualization': 'linear-line',
#     'y_name': 'Power<br>Law<br>Coefficient',
#     'description': 'Power spectrum along sequence dimension of hidden states for OPT-125M model, averaged across batch and latent dimensions.',
#     'config': model.config.to_dict(),
# }
# columns = {'Tokens': np.array([i * TOKENS_PER_STEP for i in pythia_steps])}
# # columns['Coefficient'] = np.array(slopes)
# columns['Coefficient'] = final_hs_slopes

# write_artifact(artifact_metadata, columns)


# artifact_model_name = model_name.replace('/', '--').replace('.', 'dot')
# artifact_metadata = {
#     'name': f'optspectrum-{artifact_model_name}-attn-residuals-power-law-r2-over-training-fft-filter-{attn_residuals_filter_patch}',
#     'visualization': 'linear-line',
#     'y_name': 'R2<br>Score',
#     'description': 'Power spectrum along sequence dimension of hidden states for OPT-125M model, averaged across batch and latent dimensions.',
#     'config': model.config.to_dict(),
# }
# columns = {'Tokens': np.array([i * TOKENS_PER_STEP for i in pythia_steps])}
# # columns['R2'] = np.array([r2_score(mps_across_training[i, 1:], np.exp(linear_fits[i])) for i in range(len(linear_fits))])
# columns['R2'] = attn_residuals_r2s

# write_artifact(artifact_metadata, columns)

# artifact_model_name = model_name.replace('/', '--').replace('.', 'dot')
# artifact_metadata = {
#     'name': f'optspectrum-{artifact_model_name}-attn-residuals-power-law-coefficient-over-training-fft-filter-{attn_residuals_filter_patch}',
#     'visualization': 'linear-line',
#     'y_name': 'Power<br>Law<br>Coefficient',
#     'description': 'Power spectrum along sequence dimension of hidden states for OPT-125M model, averaged across batch and latent dimensions.',
#     'config': model.config.to_dict(),
# }
# columns = {'Tokens': np.array([i * TOKENS_PER_STEP for i in pythia_steps])}
# # columns['Coefficient'] = np.array(slopes)
# columns['Coefficient'] = attn_residuals_slopes

# write_artifact(artifact_metadata, columns)


# artifact_model_name = model_name.replace('/', '--').replace('.', 'dot')
# artifact_metadata = {
#     'name': f'optspectrum-{artifact_model_name}-mlp-residuals-power-law-r2-over-training-fft-filter-{attn_residuals_filter_patch}',
#     'visualization': 'linear-line',
#     'y_name': 'R2<br>Score',
#     'description': 'Power spectrum along sequence dimension of hidden states for OPT-125M model, averaged across batch and latent dimensions.',
#     'config': model.config.to_dict(),
# }
# columns = {'Tokens': np.array([i * TOKENS_PER_STEP for i in pythia_steps])}
# # columns['R2'] = np.array([r2_score(mps_across_training[i, 1:], np.exp(linear_fits[i])) for i in range(len(linear_fits))])
# columns['R2'] = mlp_residuals_r2s

# write_artifact(artifact_metadata, columns)

# artifact_model_name = model_name.replace('/', '--').replace('.', 'dot')
# artifact_metadata = {
#     'name': f'optspectrum-{artifact_model_name}-mlp-residuals-power-law-coefficient-over-training-fft-filter-{attn_residuals_filter_patch}',
#     'visualization': 'linear-line',
#     'y_name': 'Power<br>Law<br>Coefficient',
#     'description': 'Power spectrum along sequence dimension of hidden states for OPT-125M model, averaged across batch and latent dimensions.',
#     'config': model.config.to_dict(),
# }
# columns = {'Tokens': np.array([i * TOKENS_PER_STEP for i in pythia_steps])}
# # columns['Coefficient'] = np.array(slopes)
# columns['Coefficient'] = mlp_residuals_slopes

# write_artifact(artifact_metadata, columns)

artifact_model_name = model_name.replace('/', '--').replace('.', 'dot')
artifact_metadata = {
    # 'name': f'optspectrum-{artifact_model_name}-in-context-learning-score-over-training-fft-filter-{attn_residuals_filter_patch}',
    'name': f'optspectrum-{artifact_model_name}-in-context-learning-score-over-training-fft-filter-out-pass',
    'visualization': 'linear-line',
    'y_name': 'In-Context<br>Learning<br>Score',
    'description': 'Power spectrum along sequence dimension of hidden states for OPT-125M model, averaged across batch and latent dimensions.',
    'config': model.config.to_dict(),
}
columns = {'Tokens': np.array([i * TOKENS_PER_STEP for i in pythia_steps])}
# columns['Coefficient'] = np.array(slopes)
for i in range(len(cutoffs)):
    # columns[f'ICL-{attn_residuals_filter_patch}-{cutoffs[i]}'] = in_context_learning_score_across_training[:, i].numpy()
    columns[f'ICL-out-pass-{cutoffs[i]}'] = in_context_learning_score_across_training[:, i].numpy()

write_artifact(artifact_metadata, columns)

artifact_model_name = model_name.replace('/', '--').replace('.', 'dot')
artifact_metadata = {
    # 'name': f'optspectrum-{artifact_model_name}-loss-over-training-fft-filter-{attn_residuals_filter_patch}',
    'name': f'optspectrum-{artifact_model_name}-loss-over-training-fft-filter-out-pass',
    'visualization': 'linear-line',
    'y_name': 'Loss',
    'description': 'Power spectrum along sequence dimension of hidden states for OPT-125M model, averaged across batch and latent dimensions.',
    'config': model.config.to_dict(),
}
columns = {'Tokens': np.array([i * TOKENS_PER_STEP for i in pythia_steps])}
# columns['Coefficient'] = np.array(slopes)
for i in range(len(cutoffs)):
    # columns[f'Loss-{attn_residuals_filter_patch}-{cutoffs[i]}'] = loss_across_training[:, i].numpy()
    columns[f'Loss-out-pass-{cutoffs[i]}'] = loss_across_training[:, i].numpy()

write_artifact(artifact_metadata, columns)