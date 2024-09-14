import types
import time
import copy
import gc

import torch

from transformers import OPTForCausalLM, AutoTokenizer, TextStreamer, set_seed

def offload(model, stream, next=None, state_dict=None, cpu_weights=None, name_prefix='', device='cuda'):
    if state_dict is None:
        state_dict = model.state_dict()
        print(f'state_dict: {state_dict.keys()}')
    if cpu_weights is None:
        cpu_weights = {}


    leaves = []
    names = []
    if len([c for c in list(model.named_children()) if len(list(c[1].children())) == 0]) > 0:
        names, leaves = zip(*[c for c in list(model.named_children()) if len(list(c[1].children())) == 0])

    print(f'Name: {name_prefix}')
    print(f'Leaf names: {names}')

    for name, leaf in zip(names, leaves):
        if hasattr(leaf, 'weight') and hasattr(leaf.weight, 'marked'):
            setattr(leaf.weight, 'tied', True)

            # Initilize GPU weights to empty
            # print(f'tied weight name: {name_prefix + name + ".weight"}')
            cpu_weights[name_prefix + name + '.weight'] = cpu_weights[leaf.weight.weight_name]
            leaf.weight.data = torch.tensor(0.0, device='cpu', dtype=leaf.weight.dtype)

            if hasattr(leaf, 'bias') and leaf.bias is not None:
                cpu_weights[name_prefix + name + '.bias'] = cpu_weights[leaf.bias.weight_name]
                leaf.bias.data = torch.tensor(0.0, device='cpu', dtype=leaf.bias.dtype)

        elif hasattr(leaf, 'weight'):
            setattr(leaf.weight, 'marked', True)
            setattr(leaf.weight, 'weight_name', name_prefix + name + '.weight')
        
            # Initilize GPU weights to empty
            # print(f'weight name: {name}')
            cpu_weights[name_prefix + name + '.weight'] = leaf.weight.data.clone().pin_memory()
            leaf.weight.data = torch.tensor(0.0, device='cpu', dtype=leaf.weight.dtype)

            if hasattr(leaf, 'bias') and leaf.bias is not None:
                setattr(leaf.bias, 'weight_name', name_prefix + name + '.bias')
                cpu_weights[name_prefix + name + '.bias'] = leaf.bias.data.clone().pin_memory()
                leaf.bias.data = torch.tensor(0.0, device='cpu', dtype=leaf.bias.dtype)
    
    gc.collect()
    
    # print(f'cpu_weights: {cpu_weights.keys()}')

    # print(f'leaves: {leaves}')
    next_copy = copy.copy(next)
    # if next_copy is not None:
    #     for _, param in next_copy.named_parameters():
    #         param.data = param.data.pin_memory()
    #         print(f'param: {param.data.is_pinned()}')
    # print(f'next: {next_copy}')

    og_forward = model.forward

    cpu_device = torch.device('cpu')
    def forward(self, *args, **kwargs):
        # names = copy.copy(names)
        # leaves = copy.copy(leaves)

        # print()
        # print(f'Name: {name_prefix}')
        # print(f'Leaf weight names: {[leaf.weight.weight_name for leaf in leaves if hasattr(leaf, "weight")]}')
        # print()
        # print()
        # Load leaves
        torch.cuda.synchronize()
        # start = time.time()
        with torch.cuda.stream(stream):
            for name, leaf in zip(names, leaves):
                # print(f'Checking {name}')
                if hasattr(leaf, 'weight') and hasattr(leaf.weight, 'weight_name'):
                    if leaf.weight.data.device == cpu_device:
                        # print(f'Loading {leaf.weight.weight_name} with shape: {cpu_weights[leaf.weight.weight_name].shape}')
                        leaf.weight.data = torch.empty(cpu_weights[leaf.weight.weight_name].shape, device=device, dtype=leaf.weight.dtype)
                        leaf.weight.data.copy_(cpu_weights[leaf.weight.weight_name], non_blocking=True)
                if hasattr(leaf, 'bias') and leaf.bias is not None and hasattr(leaf.bias, 'weight_name'):
                    # print(f'Loading {leaf.weight.bias_name}')
                    if leaf.bias.data.device == cpu_device:
                        leaf.bias.data = torch.empty(cpu_weights[leaf.bias.weight_name].shape, device=device, dtype=leaf.bias.dtype)
                        leaf.bias.data.copy_(cpu_weights[leaf.bias.weight_name], non_blocking=True)

                # leaf.to(device, non_blocking=True)
                # full_name = name_prefix + name
                # if full_name + '.weight' in cpu_weights:
                #     print(f'Loading {full_name + ".weight"}')
                #     leaf.weight.data.copy_(cpu_weights[name_prefix + name + '.weight'], non_blocking=True)
                # if full_name + '.bias' in cpu_weights:
                #     print(f'Loading {full_name + ".bias"}')
                #     leaf.bias.data.copy_(cpu_weights[name_prefix + name + '.bias'], non_blocking=True)
        # torch.cuda.synchronize()
        # print(f'leaf load time: {time.time() - start}')

        torch.cuda.synchronize()

        # start = time.time()
        # Load next
        # print(f'next: {next_copy}')

        if next_copy is not None:
            with torch.cuda.stream(stream):
                for param in next_copy.parameters():
                    # print(f'next copying: {param.weight_name}')
                    if param.data.device == cpu_device:
                        param.data = torch.empty(cpu_weights[param.weight_name].shape, device=device, dtype=param.dtype)
                        param.data.copy_(cpu_weights[param.weight_name])
                # next_copy.to(device, non_blocking=True)

                # # check if next copy's weights are pinned
                # for name, param in next_copy.named_parameters():
                #     print(f'param: {param.data.is_pinned()}')

        # torch.cuda.synchronize()
        # 0.7890572547912598
        # print(f'next load time: {time.time() - start}')

        # Forward
        # torch.cuda.synchronize()
        # start = time.time()
        out = og_forward(*args, **kwargs)
        # torch.cuda.synchronize()
        # 0.23755240440368652
        # print(f'forward time: {time.time() - start}')

        # Unload leaves
        with torch.cuda.stream(stream):
            for name, leaf in zip(names, leaves):
                if hasattr(leaf, 'weight') and not hasattr(leaf.weight, 'tied'):
                    leaf.weight.data = torch.tensor(0.0, device='cpu', dtype=leaf.weight.dtype)
                if hasattr(leaf, 'bias') and not hasattr(leaf.weight, 'tied'):
                    leaf.bias.data = torch.tensor(0.0, device='cpu', dtype=leaf.bias.dtype)
                    # leaf.to('cpu', non_blocking=True)
                # else:
                #     print(f'not unloading tied leaf: {name}')

        return out
    
    model.forward = types.MethodType(forward, model)

    if len(leaves) < len(list(model.children())):
        # submodules = [c for c in model.children() if len(list(c.children())) > 0]
        submodule_names, submodules = zip(*[tup for tup in list(model.named_children()) if len(list(tup[1].children())) > 0])
        # print(f'child names: {names}')
        # print(f'child layers: {crew}')
        # print(f'len(children): {len(crew)}')
        for i in range(len(submodules)):
            next = None
            submodule_name_prefix = name_prefix + submodule_names[i] + '.'
            if i + 1 < len(submodules):
                next = submodules[i + 1]
            offload(submodules[i], stream, next=next, state_dict=state_dict, cpu_weights=cpu_weights, name_prefix=submodule_name_prefix, device=device)
    
    return model

# model_name = 'facebook/opt-125m'
model_name = 'facebook/opt-30b'
model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# model2 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')
# managed_model = LayerManager(model, 'top-level')

# Example input
x = torch.load('opt-30b-c4-inputs.pt')
# x = x[0].unsqueeze(0)
x = x.repeat(3, 1)
torch.cuda.synchronize()
start = time.time()
x = x.to('cuda')
torch.cuda.synchronize()
print(f'input time: {time.time() - start}')

torch.cuda.synchronize()
start = time.time()
x = x.to('cuda')
torch.cuda.synchronize()
print(f'input second time: {time.time() - start}')

x2 = x.clone()
print(f'x shape: {x.shape}')
# x = torch.randn(64, 1024).to('cuda')

# og_forward = model.forward

# def wrapped_forward(self, x):
#     print(f'wrapped forward')
#     return og_forward(x)

# model.forward = types.MethodType(wrapped_forward, model)

stream = torch.cuda.Stream()
# Forward pass
offloaded = offload(model, stream)

# with torch.no_grad():
#     torch.cuda.synchronize()
#     start = time.time()
#     output = offloaded(x)
#     torch.cuda.synchronize()
#     # 117.583815574646
#     print(f'offloaded time: {time.time() - start}')

# print(f'output: {output}')

# torch.cuda.synchronize()
# start = time.time()
# output2 = model2(x2)
# torch.cuda.synchronize()
# print(f'original time: {time.time() - start}')

# print(f'offload logits: {output.logits[0, 0, :10]}')
# print(f'original logits: {output2.logits[0, 0, :10]}')
# print(f'logits allclose: {torch.allclose(output.logits, output2.logits)}')

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

prompt = 'Making pesto from scratch can be done with these ingredients in 4 simple steps:\nStep 1'
inputs = tokenizer(prompt, return_tensors='pt')

print('Offloaded generation:')
offloaded.generate(inputs.input_ids.to('cuda'), max_new_tokens=100, do_sample=True, top_k=50, top_p=0.9, streamer=streamer)

# set_seed(42)
# print('Original generation:')
# model2.generate(inputs.input_ids.to('cuda'), max_new_tokens=100, do_sample=True, top_k=50, top_p=0.9, streamer=streamer)