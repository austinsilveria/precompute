import hashlib
import inspect
import json
from functools import wraps
import os
import types

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import pyarrow as pa
import pyarrow.parquet as pq

from precompute import PrecomputeContext, HOOKED_MODELS

pctx = PrecomputeContext()

def key(f):
    pctx.key = f()

    # Load cache metadata for key if it exists
    # { 
    #   'hook-hashes': [ 'hash1', 'hash2', ... ], 
    #   'checkpoint-hashes': [ 'hash1', 'hash2', ... ], 
    # }
    cache_metadata: dict
    pctx.cache_metadata = cache_metadata

    return f

def hook(hook_variable):
    def decorator(f):
        pctx.hooks[hook_variable].append(f)

        ser = f'{hook_variable}-{inspect.getsource(f).encode()}'
        pctx.hook_hashes.append(hashlib.sha256(ser).hexdigest())

        if ser not in pctx.cache_metadata['hook-hashes']:
            pctx.hooks_cached = False

        # @wraps(f)
        # def wrapper(*args, **kwargs):
        #     return f(*args, **kwargs)
        # return wrapper
        return f
    return decorator

# Hash the function
# If hooks match and function hash is in cache, return the cached pctx
def checkpoint(cache='./cache'):
    def decorator(f):
        # later checkpoints need to be a function of earlier checkpoints
        hash = hashlib.sha256(inspect.getsource(f).encode()).hexdigest()

        @wraps(f)
        def wrapper(*args, **kwargs):
            if pctx.hooks_cached and hash in pctx.cache_metadata['checkpoint-hashes']:
                pctx.cache = read_cache(hash)
                return pctx
            pctx = f(*args, **kwargs)
            write_cache(hash, pctx)
            return pctx
        return wrapper
    return decorator

# Executes function and stores the result
def vis(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        data = f(*args, **kwargs)

        x = None
        y = None
        for k in data['cols']:
            if isinstance(data['cols'][k], dict):
                y = k
            else:
                x = k
        cols = {
            x: cols[x],
            **cols[y],
        }

        data['y_name'] = y
        
        table = pa.table(data['cols'])
        path: str
        pq.write_table(table, path)

        metadata = { k: v for k, v in data.items() if k != 'cols' }

        metadata_path: str
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    return wrapper

def model(model):
    # If model class not in HOOKED_MODELS, raise error
    if model not in HOOKED_MODELS:
        raise NotImplementedError(f'Model {model} not supported')

    pctx.config = model.config

    hooked_model = HOOKED_MODELS[model](config=model.config)
    hooked_model.load_state_dict(model.state_dict())

    fwd = hooked_model.forward
    def forward(*args, **kwargs):
        out = fwd(*args, **kwargs, pctx=pctx)
        return out

    hooked_model.forward = types.MethodType(forward, hooked_model)

    return hooked_model, pctx

def write_cache(hash, pctx):
    tensors = { k: v for k, v in pctx.cache.items() if isinstance(v, torch.Tensor) }
    other = { k: v for k, v in pctx.cache.items() if not isinstance(v, torch.Tensor) }

    save_file(tensors, f'{pctx.key}-{hash}-tensors.safetensors')

    with open(f'{hash}-other.json', 'w') as f:
        json.dump(other, f)

def read_cache(hash, pctx):
    tensors = safe_open(f'{pctx.key}-{hash}-tensors.safetensors')
    with open(f'{hash}-other.json', 'r') as f:
        other = json.load(f)

    pctx = { **tensors, **other }

    return pctx
    
