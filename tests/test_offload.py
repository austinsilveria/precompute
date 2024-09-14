import torch

from transformers import OPTForCausalLM, AutoTokenizer, TextStreamer, set_seed

from precompute import offload

model_name = 'facebook/opt-125m'
model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model2 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')

x = torch.load('opt-30b-c4-inputs.pt').to('cuda')
x2 = x.clone()

stream = torch.cuda.Stream()
offloaded = offload(model, stream)

with torch.no_grad():
    output = offloaded(x)

output2 = model2(x2)

if not torch.allclose(output.logits, output2.logits):
    raise Exception('Outputs do not match')

model_name = 'facebook/opt-30b'
model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
offloaded = offload(model, stream)

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

prompt = 'Making pesto from scratch can be done with these ingredients in 4 simple steps:\nStep 1'
inputs = tokenizer(prompt, return_tensors='pt')

print('Offloaded generation:')
offloaded.generate(inputs.input_ids.to('cuda'), max_new_tokens=5, do_sample=True, top_k=50, top_p=0.9, streamer=streamer)