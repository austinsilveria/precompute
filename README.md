
<p align="center">
  precompute: Real time sequence modeling research.
</p>

<p align="center">
<a href="https://github.com/austinsilveria/precompute/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/austinsilveria/precompute.svg"></a>
</p>

### Super Simple Offloading
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

### Context, Hooks, Mutability
* Define hook functions triggered by the addition of a specified variable to the computation graph.
* Read context, write context, or write an artifact.

### Artifact Persistence
* Automatic name prefixing based on current configuration values.
* Automatic immutable context log persistence over training.
* Memmap storage for incremental tensor artifact persistence.

### Artifact Visualization
* Define visualization functions.
* Browse and select artifact visualizations.
* Live visualization dashboard.
