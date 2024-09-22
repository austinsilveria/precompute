import os
import json
import re
from typing import Dict, Union, Optional, List, Tuple, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import OPTForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from transformers.modeling_utils import unwrap_model
from transformers.trainer_pt_utils import (nested_detach)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from datasets import load_dataset

from diffusers import DDPMScheduler

from diffusers import AutoPipelineForInpainting

import matplotlib.pyplot as plt

from precompute import (
    Hook,
    HookVariableNames,
    HookedOPTForCausalLM,
    PrecomputeContext,
    write_artifact,
)

torch.manual_seed(0)

def entropy(prob_dist, dim=1):
    return -torch.sum(prob_dist * torch.log(prob_dist + 1e-9), dim=dim)  # Adding a small value to avoid log(0)

# HookVariableNames.PRE_MASK_ATTN_WEIGHTS
def steal_pre_mask_attn_weights(x, ctx):
    ctx.context['pre_mask_attn_weights'] = x
    return x

# HookVariableNames.POST_MASK_ATTN_WEIGHTS
def patch_post_mask_attn_weights(x, ctx):
    if ctx.context['layer'] != 0 and ctx.context['layer'] <= config.num_hidden_layers - 4:
        # print(f'Layer {ctx.context["layer"]} bidirectional attention.')
        return ctx.context['pre_mask_attn_weights']
    # print(f'Layer {ctx.context["layer"]} causal attention.')
    return x

# HookVariableNames.POST_MLP
def patch_diffusion_model(x, ctx):
    if ctx.context['layer'] == 0:
        # steal post first layer for later
        ctx.context['post_layer_1'] = x
        # patch with noisy hidden states
        return ctx.context['noisy_hidden_states'].clone()
    elif ctx.context['layer'] == config.num_hidden_layers - 4:
        # reverse diffusion
        pctx.context['diffusion_out'] = x

        # add in post layer 1 with three layers left
        # return ctx.context['denoised_hidden_states'] + ctx.context['post_layer_1']
        # test without diffusion
        # 3.23 cross entropy loss
        return ctx.context['post_layer_1']
    return x

hooks = [
    # Hook(HookVariableNames.PRE_MASK_ATTN_WEIGHTS, steal_pre_mask_attn_weights),
    # Hook(HookVariableNames.POST_MASK_ATTN_WEIGHTS, patch_post_mask_attn_weights),
    Hook(HookVariableNames.POST_MLP, patch_diffusion_model),
]

model_name = "facebook/opt-125m"
# diffusion_checkpoint = 'optspectrum-diffusion-10x-patch-125m-x0-target-True-clip-False-1000-linear-0.02-0.0001-10000-0.2/checkpoint-10000'
# diffusion_checkpoint = 'optspectrum-diffusion-10x-patch-125m-clip-False-1000-linear-0.02-0.0001-150000-0.2/checkpoint-150000'
# diffusion_checkpoint = 'optspectrum-diffusion-10x-patch-125m-1000-linear-0.02-0.0001-150000-0.2/checkpoint-150000'
# diffusion_checkpoint = 'optspectrum-bidir-diffusion-10x-patch-125m-1000-linear-0.02-0.0001-150000-0.2/checkpoint-150000'
diffusion_checkpoint = 'optspectrum-no-diffusion-patch-125m-1000-linear-0.02-0.0001-150000-0.2/checkpoint-150000'

tokenizer = AutoTokenizer.from_pretrained(model_name)
teacher = OPTForCausalLM.from_pretrained(model_name).to('cuda')
model = HookedOPTForCausalLM.from_pretrained(diffusion_checkpoint).to('cuda')
config = model.config
pctx = PrecomputeContext(config, hooks=hooks)

clip = True
noise_scheduler = DDPMScheduler(num_train_timesteps=config.ddpm_num_steps, beta_schedule=config.ddpm_beta_schedule, beta_end=config.ddpm_beta_end, clip_sample=clip)

inputs = torch.load('opt-30b-c4-first-8-validation-inputs.pt').to('cuda')[4:]

token_cutoff = 504
end_before = -3
# i = 0

teacher_ce_sum = torch.tensor(0.0, device='cuda')
diffusion_ce_sum = torch.tensor(0.0, device='cuda')
noisy_ce_sum = torch.tensor(0.0, device='cuda')
for i in range(inputs.shape[0]):
# for i in [7]:
    print(f'text: {tokenizer.decode(inputs[i])}')

    with torch.no_grad():
        teacher_outputs = teacher(
            input_ids=inputs,
            output_hidden_states=True,
        )

        hidden_states = teacher_outputs.hidden_states[-1]

        noise = torch.randn_like(hidden_states, device=hidden_states.device)

        # baseline teacher 12 layers      : 2.30 CE loss
        # baseline last 3 layers          : 3.15 CE loss
        #
        # unidirectional 0  , no clipping : 2.25 CE loss
        # unidirectional 250, no clipping : 2.28 CE loss
        # unidirectional 500, no clipping : 4.29 CE loss
        # unidirectional 750, no clipping : 7.45 CE loss
        # 
        # unidirectional 0  , clipping    :
        # unidirectional 250, clipping    : 2.30 CE loss
        # unidirectional 500, clipping    : 4.17 CE loss
        # unidirectional 750, clipping    : 5.41 CE loss
        #
        # bidirectional  250, no clipping : 2.45 CE loss
        # bidirectional  500, no clipping : 5.63 CE loss
        # bidirectional  750, no clipping : 8.97 CE loss

        # unidirectional 250, clip, 1-step: 2.33 CE loss
        # unidirectional 500, clip, 1-step: 2.66 CE loss
        # unidirectional 750, clip, 1-step: 3.18 CE loss
        # unidirectional 999, clip, 1-step: 3.15 CE loss
        timesteps = torch.tensor([999], device=hidden_states.device).repeat(hidden_states.shape[0])

        noisy_hidden_states = hidden_states
        noisy_hidden_states = noise_scheduler.add_noise(noisy_hidden_states, noise, timesteps)
        pctx.context['denoised_hidden_states'] = noisy_hidden_states[i].unsqueeze(0)

        latent = noisy_hidden_states[i]

        resampling = True
        inference_steps = list(range(timesteps[i], -1, -1))
        if resampling:
            # inference_steps = list(range(timesteps[i], -1, -10))
            # inference_steps = [999, 950, 900, 850, 800, 750, 725, 700]
            inference_steps = [999]
        for j in inference_steps:
            # Keep resetting the latent state to the ground truth
            non_inpaint = noise_scheduler.add_noise(hidden_states, noise, torch.tensor([j], device=hidden_states.device).repeat(hidden_states.shape[0]))
            latent[:token_cutoff, :] = non_inpaint[i, :token_cutoff, :]
            if resampling:
                inpaint = noise_scheduler.add_noise(latent.unsqueeze(0), noise, torch.tensor([j], device=hidden_states.device).unsqueeze(0))
                latent[token_cutoff:] = inpaint[0, token_cutoff:]
            print(f'latent token block norm: {torch.norm(latent[token_cutoff:], dim=-1).mean()}')

            pctx.context['diffusion_step'] = torch.tensor([j], device=latent.device).unsqueeze(0)
            pctx.context['noisy_hidden_states'] = latent.unsqueeze(0)
            out = model(inputs[i].unsqueeze(0), pctx=pctx)
            noise_pred = pctx.context['diffusion_out']
            print(f'noise norm: {torch.norm(noise_pred, dim=-1).mean()}')

            step_out = noise_scheduler.step(noise_pred[0], j, latent, return_dict=True)
            print(f'pred_orig token block norm: {torch.norm(step_out.pred_original_sample[token_cutoff:], dim=-1).mean()}')

            latent = step_out.prev_sample

            if resampling:
                latent = step_out.pred_original_sample
            # break
        latent[:token_cutoff, :] = hidden_states[i, :token_cutoff, :]

        print()
        # latent token block: tensor([[-0.8464,  0.6618,  0.0250,  ..., -0.9641, -1.0000,  0.1164],                              
        #     [-0.5494,  0.3398,  0.6110,  ..., -0.7336, -0.7933,  0.7164],                                                  
        #     [-0.5801, -0.7156,  0.1830,  ..., -0.9897, -0.9885,  0.7508],                                                  
        #     [ 0.1980,  0.9333, -0.3636,  ..., -0.2155, -1.0000,  0.2780],                                                  
        #     [-0.6022,  0.7348, -0.2540,  ..., -0.9563, -0.7869,  0.1282]],                                                 
        #    device='cuda:0')
        print(f'latent token block: {latent[token_cutoff:end_before]}')
        print(f'hidden_states token block: {hidden_states[i, token_cutoff:end_before]}')
        print(f'latent token block norm: {torch.norm(latent[token_cutoff:], dim=-1).mean()}')
        print(f'hidden_states token block norm: {torch.norm(hidden_states[i, token_cutoff:], dim=-1).mean()}')

        pctx.context['denoised_hidden_states'] = latent.unsqueeze(0)
        logits = model(inputs[i].unsqueeze(0), pctx=pctx)[0]
        logits = logits[0, token_cutoff:end_before - 1] * 1

        k = 4
        teacher_logits = teacher.lm_head(hidden_states[i])[token_cutoff:end_before - 1]
        print(f'logits shape: {teacher_logits.shape}')
        # print(f'Teacher CE loss: {CrossEntropyLoss()(teacher_logits, inputs[i][token_cutoff + 1:])}')
        teacher_ce_sum += CrossEntropyLoss()(teacher_logits, inputs[i][token_cutoff + 1:end_before])
        topk_teacher = torch.topk(teacher_logits, k, dim=-1)[1]
        print(f'topk shape: {topk_teacher.shape}')
        print(f'topk entropy: {entropy(torch.softmax(teacher_logits, dim=-1), dim=-1)}')
        for j in range(topk_teacher.shape[0]):
            print(f'{j} token: {tokenizer.decode(topk_teacher[j])}')
        num_tokens_contained = torch.sum(torch.any(topk_teacher == inputs[i][token_cutoff + 1:end_before].unsqueeze(1), dim=-1))
        print(f'num_tokens_contained teacher: {num_tokens_contained}')

        print()
        # Cross Entropy loss against teacher logits
        # print(f'Diffusion CE loss: {CrossEntropyLoss()(logits, inputs[i][token_cutoff + 1:])}')
        print(f'logits shape: {logits.shape}')
        print(f'inputs[i][token_cutoff + 1:end_before]: {inputs[i][token_cutoff + 1:end_before].shape}')
        diffusion_ce_sum += CrossEntropyLoss()(logits, inputs[i][token_cutoff + 1:end_before])
        topk = torch.topk(logits, k, dim=-1)[1]
        print(f'topk entropy: {entropy(torch.softmax(logits, dim=-1), dim=-1)}')
        # print(f'indices: {indices}')
        # print(f'tokens: {tokenizer.batch_decode(topk)}')
        for j in range(topk_teacher.shape[0]):
            print(f'{j} token: {tokenizer.decode(topk[j])}')
        num_tokens_contained = torch.sum(torch.any(topk == inputs[i][token_cutoff + 1:end_before].unsqueeze(1), dim=-1))
        print(f'num_tokens_contained diffusion: {num_tokens_contained}')

print()
print(f'avg teacher CE loss: {teacher_ce_sum / inputs.shape[0]}')
print(f'avg diffusion CE loss: {diffusion_ce_sum / inputs.shape[0]}')
print(f'avg noisy CE loss: {noisy_ce_sum / inputs.shape[0]}')
print(f'clipping: {clip}')