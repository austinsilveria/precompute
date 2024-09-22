# 1/ From pretrained, then override config.num_hidden_layers=3, and decoder.layers = decoder.layers[:3]
# 2/ Patch post pos embeddings to add slightly noisy final hidden states computed from pretrained model
# 3/ Test how efficiently the model can autoreggresively generate if the input representations are slightly noisy

import os
import json
import re
from typing import Dict, Union, Optional, List, Tuple, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from einops import rearrange

from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from transformers.modeling_utils import unwrap_model
from transformers.trainer_pt_utils import (nested_detach)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from datasets import load_dataset

from diffusers import DDPMScheduler

from precompute import HookedOPTForCausalLM, Hook, HookVariableNames, PrecomputeContext, write_artifact

import matplotlib.pyplot as plt

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, is_eval=False, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if is_eval:
            torch.random.set_rng_state(self.eval_rng_state)

        input_ids = inputs['input_ids']

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                labels=input_ids,
                output_hidden_states=True,
            )

        teacher_loss = teacher_outputs.loss
        hidden_states = teacher_outputs.hidden_states[-1]

        noise = torch.randn_like(hidden_states, device=hidden_states.device)

        # t = int(self.model.config.noise_pct * self.model.config.ddpm_num_steps)
        # Only timesteps between e.g. 10% and 100%
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (hidden_states.shape[0],), device=hidden_states.device)
        timesteps = timesteps.long()
        self.pctx.context['diffusion_step'] = timesteps
        noisy_hidden_states = self.noise_scheduler.add_noise(hidden_states, noise, timesteps)
        self.pctx.context['noisy_hidden_states'] = noisy_hidden_states

        student_output = model(input_ids, labels=input_ids, pctx=self.pctx)

        lm_loss = student_output.loss
        # diffusion_loss = F.mse_loss(self.pctx.context['diffusion_out'], noise) * 10
        diffusion_loss = F.mse_loss(self.pctx.context['diffusion_x0'], hidden_states)
        loss = lm_loss + diffusion_loss
        # loss = lm_loss

        if 'Loss' not in self.pctx.training_log:
            self.pctx.training_log['Loss'] = {
                'Student<br>Loss': torch.empty(self.args.max_steps // 50 + 1, device='cpu'),
                'Teacher<br>Loss': torch.empty(self.args.max_steps // 50 + 1, device='cpu'),
            }
            self.pctx.training_log['Total<br>Loss'] = {
                'Total<br>Loss': torch.empty(self.args.max_steps // 50 + 1, device='cpu'),
            }
        self.pctx.training_log['Loss']['Student<br>Loss'][self.state.global_step // 50] = lm_loss.item()
        self.pctx.training_log['Loss']['Teacher<br>Loss'][self.state.global_step // 50] = teacher_loss.item()
        self.pctx.training_log['Total<br>Loss']['Total<br>Loss'][self.state.global_step // 50] = loss.item()

        return (loss, student_output) if return_outputs else loss
    
def get_save_dir_name(config):
    return f'optspectrum-diffusion-10x-patch-125m-x0-target-{config.x0_target}-clip-{config.clip_sample}-{config.ddpm_num_steps}-{config.ddpm_beta_schedule}-{config.ddpm_beta_end}-{config.lr}-{config.max_steps}-{config.noise_pct}'

def main():
    model_name = "facebook/opt-125m"
    dataset_name = "c4"

    raw_datasets = load_dataset(dataset_name, 'en', streaming=True)
    print(next(iter(raw_datasets['train'])))

    context_length = 512
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
    print(f'next iter tokenized: {next(iter(tokenized_datasets["train"]))}')

    teacher = OPTForCausalLM.from_pretrained(model_name).to('cuda')



    config = AutoConfig.from_pretrained(model_name)
    config.lr = 1e-4
    config.max_steps = 10000
    config.ddpm_num_steps = 1000
    config.ddpm_beta_schedule = 'linear'
    config.ddpm_beta_end = 0.02
    config.noise_pct = 0.2
    config.clip_sample = False
    config.x0_target = True

    SAVE_DIR = get_save_dir_name(config)
    CHECKPOINT = None
    TEACHER_CHECKPOINT = None
    if CHECKPOINT is None and os.path.exists(SAVE_DIR + '/log.txt'):
        raise Exception(f'{SAVE_DIR} already exists')
        os.remove(SAVE_DIR + '/log.txt')


    # HookVariableNames.POST_POS_EMBEDDINGS
    def add_noisy_hidden_states(x, ctx):
        # steal embeddings for later
        ctx.context['post_pos_embeddings'] = x
        # patch with noisy hidden states for now
        return ctx.context['noisy_hidden_states']

    # HookVariableNames.PRE_MLP
    def print_layer(x, ctx):
        print(f'layer: {ctx.context["layer"]}')
        return x

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
            orig = torch.empty_like(x)
            pctx.context['diffusion_out'] = x
            for i in range(ctx.context['diffusion_step'].shape[0]):
                t = ctx.context['diffusion_step'][i]
                step_out = ctx.noise_scheduler.step(x[i], t, ctx.context['noisy_hidden_states'][i], return_dict=True)
                orig[i] = step_out.pred_original_sample
            pctx.context['diffusion_x0'] = orig
            # add in post layer 1 with three layers left
            # print(f'Post Layer {ctx.context["layer"]} switching to causal.')
            return orig + ctx.context['post_layer_1']
            # return ctx.context['post_layer_1']
        return x

    hooks = [
        # Hook(HookVariableNames.POST_POS_EMBEDDINGS, add_noisy_hidden_states),
        # Hook(HookVariableNames.PRE_MLP, print_layer),
        # Hook(HookVariableNames.PRE_MASK_ATTN_WEIGHTS, steal_pre_mask_attn_weights),
        # Hook(HookVariableNames.POST_MASK_ATTN_WEIGHTS, patch_post_mask_attn_weights),
        Hook(HookVariableNames.POST_MLP, patch_diffusion_model),
    ]
    pctx = PrecomputeContext(config, hooks=hooks)

    model = HookedOPTForCausalLM.from_pretrained(model_name, config=config).to('cuda')
    # First 3 and last
    # model.model.decoder.layers = model.model.decoder.layers[:3] + [model.model.decoder.layers[-1]]

    pctx.model = model

    model.train()
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{model_name} size: {model_size/1000**2:.1f}M parameters")

    max_steps = config.max_steps

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=1000,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=int(max_steps * 0.2),
        lr_scheduler_type="cosine",
        learning_rate=config.lr,
        save_steps=max_steps,
        fp16=True,
        max_steps=max_steps,
    )

    class StepEndCallback(TrainerCallback):
        def on_step_end(self, args, state, control, logs=None, **kwargs):
            if state.global_step % 50 == 0:
                if 'tokens' not in self.pctx.context:
                    self.pctx.context['tokens'] = torch.empty(((config.max_steps // 50) + 1), device='cuda')
                self.pctx.context['tokens'][state.global_step // 50] = state.global_step * args.per_device_train_batch_size * context_length

                artifact_prefix = SAVE_DIR.replace('.', 'dot').replace('/', '--')
                for key, group in pctx.training_log.items():
                    name = f'{artifact_prefix}-{key}'
                    columns = {'Tokens': self.pctx.context['tokens'][:state.global_step // 50].cpu().numpy()}
                    for key2, value in group.items():
                        columns[key2] = value[:state.global_step // 50].cpu().numpy()

                    artifact_metadata = {
                        'name': name,
                        'artifact_group': f'optspectrum-diffusion-{key}',
                        'visualization': 'linear-line',
                        'window-average': 10,
                        'y_name': key,
                        'description': '',
                        'config': model.config.to_dict(),
                    }
                    write_artifact(artifact_metadata, columns)


    step_end_callback = StepEndCallback()
    step_end_callback.model = model
    step_end_callback.pctx = pctx

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"].take(100),
        callbacks=[step_end_callback],
    )


    noise_scheduler = DDPMScheduler(num_train_timesteps=config.ddpm_num_steps, beta_schedule=config.ddpm_beta_schedule, beta_end=config.ddpm_beta_end, clip_sample=config.clip_sample)
    pctx.noise_scheduler = noise_scheduler
    trainer.noise_scheduler = noise_scheduler
    trainer.scaling_factor = None

    trainer.teacher = teacher
    trainer.eval_rng_state = torch.random.get_rng_state()
    trainer.pctx = pctx

    trainer.train(resume_from_checkpoint=CHECKPOINT)


if __name__ == '__main__':
    main()