import copy
import types

import torch

def offload(model, stream=None, next=None, state_dict=None, cpu_weights=None, name_prefix='', device='cuda'):
    if stream is None:
        stream = torch.cuda.Stream()
    if state_dict is None:
        state_dict = model.state_dict()
    if cpu_weights is None:
        cpu_weights = {}
    cpu_device = torch.device('cpu')

    leaves = []
    names = []
    if len([c for c in list(model.named_children()) if len(list(c[1].children())) == 0]) > 0:
        names, leaves = zip(*[c for c in list(model.named_children()) if len(list(c[1].children())) == 0])

    for name, leaf in zip(names, leaves):
        if hasattr(leaf, 'weight'):
            if hasattr(leaf.weight, 'marked'):
                setattr(leaf.weight, 'tied', True)
                weights_from = cpu_weights[leaf.weight.weight_name]
                biases_from = cpu_weights[leaf.bias.weight_name] if hasattr(leaf, 'bias') and leaf.bias is not None else None
            else:
                setattr(leaf.weight, 'marked', True)
                setattr(leaf.weight, 'weight_name', name_prefix + name + '.weight')
                weights_from = leaf.weight.data
                biases_from = None
                if hasattr(leaf, 'bias') and leaf.bias is not None:
                    setattr(leaf.bias, 'weight_name', name_prefix + name + '.bias')
                    biases_from = leaf.bias.data

            # Move weights out of model
            cpu_weights[name_prefix + name + '.weight'] = weights_from.clone().pin_memory()
            # Dummy value since can't be None
            leaf.weight.data = torch.tensor(0.0, device='cpu', dtype=leaf.weight.dtype)

            if hasattr(leaf, 'bias') and leaf.bias is not None:
                cpu_weights[name_prefix + name + '.bias'] = biases_from.clone().pin_memory()
                leaf.bias.data = torch.tensor(0.0, device='cpu', dtype=leaf.bias.dtype)
    
    # Flush stream and print progress
    pct = int(100 * len(list(cpu_weights.keys())) // len(list(state_dict.keys())))
    num_hyphens = int(50 * pct // 100)
    num_spaces = 50 - num_hyphens
    print(f'Offloading Progress: [{pct}%] |{chr(9608) * num_hyphens}{" " * num_spaces}|', end='\r')
    
    # Shallow copy next layer
    next_copy = copy.copy(next)

    original_forward = model.forward

    # Wrap forward pass
    def forward(self, *args, **kwargs):
        # Load leaves if we don't have them
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            for leaf in leaves:
                if hasattr(leaf, 'weight') and hasattr(leaf.weight, 'weight_name'):
                    if leaf.weight.data.device == cpu_device:
                        leaf.weight.data = torch.empty(cpu_weights[leaf.weight.weight_name].shape, device=device, dtype=leaf.weight.dtype)
                        leaf.weight.data.copy_(cpu_weights[leaf.weight.weight_name], non_blocking=True)
                if hasattr(leaf, 'bias') and leaf.bias is not None and hasattr(leaf.bias, 'weight_name'):
                    if leaf.bias.data.device == cpu_device:
                        leaf.bias.data = torch.empty(cpu_weights[leaf.bias.weight_name].shape, device=device, dtype=leaf.bias.dtype)
                        leaf.bias.data.copy_(cpu_weights[leaf.bias.weight_name], non_blocking=True)

        torch.cuda.synchronize()

        # Overlap copy with compute
        if next_copy is not None:
            with torch.cuda.stream(stream):
                for param in next_copy.parameters():
                    if param.data.device == cpu_device:
                        param.data = torch.empty(cpu_weights[param.weight_name].shape, device=device, dtype=param.dtype)
                        param.data.copy_(cpu_weights[param.weight_name])

        out = original_forward(*args, **kwargs)

        # Unload leaves
        with torch.cuda.stream(stream):
            for name, leaf in zip(names, leaves):
                if hasattr(leaf, 'weight') and not hasattr(leaf.weight, 'tied'):
                    leaf.weight.data = torch.tensor(0.0, device='cpu', dtype=leaf.weight.dtype)
                if hasattr(leaf, 'bias') and not hasattr(leaf.weight, 'tied'):
                    leaf.bias.data = torch.tensor(0.0, device='cpu', dtype=leaf.bias.dtype)

        return out
    
    model.forward = types.MethodType(forward, model)

    # Recurse through module children
    if len(leaves) < len(list(model.children())):
        submodule_names, submodules = zip(*[tup for tup in list(model.named_children()) if len(list(tup[1].children())) > 0])
        for i in range(len(submodules)):
            next = None
            submodule_name_prefix = name_prefix + submodule_names[i] + '.'
            if i + 1 < len(submodules):
                next = submodules[i + 1]
            offload(submodules[i], stream=stream, next=next, state_dict=state_dict, cpu_weights=cpu_weights, name_prefix=submodule_name_prefix, device=device)
    
    return model
