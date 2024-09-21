.. Precompute documentation master file, created by
   sphinx-quickstart on Fri Sep 20 12:35:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================================
Precompute: The Sequence Model Analysis Library.
================================================

``Precompute`` is a lightweight sequence model analysis library that decouples logging and architecture tweaks from specific model implementations.

Hooks + Context
---------------

.. code-block:: python

   import torch
   from transformers import AutoTokenizer
   
   from precompute import (
       Hook,
       HookVariableNames,
       HookedOPTForCausalLM,
       HookedGPTNeoXForCausalLM,
       PrecomputeContext,
   )
   
   def log_post_mlp(x: torch.Tensor, pctx: PrecomputeContext) -> torch.Tensor:
       if 'post-mlp' not in pctx.context:
           # [l, b, n, d]
           pctx.context['post-mlp'] = torch.zeros(
                   pctx.config.num_hidden_layers,
                   *x.shape,
                   device='cpu'
               )
       # Log activation
       pctx.context['post-mlp'][pctx.context['layer']] = x.detach()
       # Option to mutate activation or pass through
       return x
   
   hooks = [
       Hook(HookVariableNames.POST_MLP, log_post_mlp),
   ]
   
   model_name = 'facebook/opt-125m'
   model = HookedOPTForCausalLM.from_pretrained(model_name).to('cuda')
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   prompt = 'Saturating the compute and memory hierarchy.'
   inputs = tokenizer(prompt, return_tensors='pt')

   # Hooks automatically executed
   pctx = PrecomputeContext(model.config, hooks=hooks)
   output = model(inputs.input_ids.to('cuda'), pctx=pctx)
   
   # Analyze logged activations
   frequencies = torch.fft.rfftfreq(pctx.context['post-mlp'].shape[2])
   fft = torch.fft.rfft(pctx.context['post-mlp'], dim=2)
   mean_power_spectrum = torch.mean(fft**2, dim=(1, 3))

   # Same hooks for a different model
   model_name = 'EleutherAI/pythia-70m-deduped'
   model = HookedOPTForCausalLM.from_pretrained(model_name).to('cuda')
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   prompt = 'Saturating the compute and memory hierarchy.'
   inputs = tokenizer(prompt, return_tensors='pt')

   # Hooks automatically executed
   pctx = PrecomputeContext(model.config, hooks=hooks)
   output = model(inputs.input_ids.to('cuda'), pctx=pctx)

Mutability
----------

With the global context of ``pctx``, we can also tweak the architecture--e.g. switching to bidirectional attention:

.. code-block:: python

    # HookVariableNames.PRE_MASK_ATTN_WEIGHTS
    def steal_pre_mask_attn_weights(x, pctx):
        pctx.context['pre_mask_attn_weights'] = x
        return x

    # HookVariableNames.POST_MASK_ATTN_WEIGHTS
    def patch_post_mask_attn_weights(x, pctx):
        # Route around attention mask
        return pctx.context['pre_mask_attn_weights']

or applying a high pass filter to the attention residuals in each layer:

.. code-block:: python

   # HookVariableNames.ATTN_RESIDUALS
   def high_pass_filter_attn_residuals(x, pctx):
       fft = torch.fft.fft(x, dim=1)
       freqs = torch.fft.fftfreq(x.shape[1])

       cutoff_freq = cutoffs[pctx.context['cutoff']]
       mask = torch.where(torch.abs(freqs) >= cutoff_freq, 1, 0).to('cuda')

       filtered_fft = fft * mask.unsqueeze(0).unsqueeze(-1)

       filtered = torch.fft.ifft(filtered_fft, dim=1).real
       return filtered

.. toctree::
   :maxdepth: 3
   :caption: Getting Started
   :hidden:

   Introduction <self>
   quickstart
   
.. toctree::
   :maxdepth: 3
   :caption: Tutorials
   :hidden:

   tutorials/analyzing_transformer_power_spectrums
   tutorials/making_OPT_a_diffusion_model

.. title:: Introduction
