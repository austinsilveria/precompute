
<p align="center">
  precompute: Sequence model analytics tooling. Research to saturate the compute and memory hierarchy.
</p>

<p align="center">
<a href="https://github.com/austinsilveria/precompute/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/austinsilveria/precompute.svg"></a>
</p>

### Context + Hooks
Define hook functions triggered by the addition of a specified variable to the computation graph.
```python
import torch
from transformers import AutoTokenizer

from precompute import (
    Hook,
    HookVariableNames,
    HookedOPTForCausalLM,
    PrecomputeContext,
)

model_name = 'facebook/opt-125m'
model = HookedOPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = 'Saturating the compute and memory hierarchy is a prerequisite of the merge.'
inputs = tokenizer(prompt, return_tensors='pt')

def log_post_mlp(x, ctx):
    """
    Log post-MLP hidden states for each layer.

    x     : [b, n, d]
    ctx   : PrecomputeContext

    return: [b, n, d]
    """
    if 'post-mlp' not in ctx.context:
        # [l, b, n, d]
        ctx.context['post-mlp'] = torch.zeros(ctx.config.num_hidden_layers, *x.shape, device='cpu')
    ctx.context['post-mlp'][ctx.context['layer']] = x.detach()
    # Option to mutate activation or pass through
    return x

hooks = [
    Hook(HookVariableNames.POST_MLP, log_post_mlp),
]
pctx = PrecomputeContext(model.config, hooks=hooks)

# Hooks automatically executed
output = model(inputs.input_ids.to('cuda'), pctx=pctx)

# Analyze logged activations
frequencies = torch.fft.rfftfreq(pctx.context['post-mlp'].shape[2])
fft = torch.fft.rfft(pctx.context['post-mlp'], dim=2)
mean_power_spectrum = torch.mean(fft**2, dim=(1, 3))
```

### Super Simple Offloading
Model agnostic offloading with a simple function call.
```python
import torch
from transformers import OPTForCausalLM, AutoTokenizer, TextStreamer, set_seed

from precompute import offload

set_seed(42)
model_name = 'facebook/opt-30b'
model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_special_tokens=True)
prompt = 'Making pesto from scratch can be done with these ingredients in 4 simple steps:\nStep 1'
inputs = tokenizer(prompt, return_tensors='pt')

offloaded = offload(model)

offloaded.generate(inputs.input_ids.to('cuda'), max_new_tokens=50, do_sample=True, top_k=50, top_p=0.9, streamer=streamer)

print(f'Max GPU mem usage: {torch.cuda.max_memory_allocated("cuda") / 1024 ** 3} GB\n===')
```
~~~
Making pesto from scratch can be done with these ingredients in 4 simple steps:
Step 1: Crush the fresh basil leaves (freshly grown or dried basil is fine, though we are not going to discuss the nuances between the two here).
Step 2: Add olive oil, a little salt, a couple of crushed garlic cloves and one

Max GPU mem usage: 3.1706390380859375 GB
~~~


### Artifact Persistence
* Automatic name prefixing based on current configuration values.
* Automatic immutable context log persistence over training.
* Memmap storage for incremental tensor artifact persistence.

### Artifact Visualization
* Define visualization functions.
* Browse and select artifact visualizations.
* Live visualization dashboard.
