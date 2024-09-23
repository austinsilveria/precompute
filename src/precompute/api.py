import hashlib
import inspect
import json
from functools import wraps
import os
import types
import uuid

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import pyarrow as pa
import pyarrow.parquet as pq

from precompute import PrecomputeContext, HOOKED_MODELS

pctx = PrecomputeContext()

def key(k):
    pctx.key = json.dumps(k)

    # Load cache metadata for key if it exists
    # { 
    #   'hook-hashes': [ 'hash1', 'hash2', ... ], 
    #   'checkpoint-hashes': [ 'hash1', 'hash2', ... ], 
    # }
    path = f'{pctx.cache_dir}/{hash(pctx.key)}-cache-metadata.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            cache_metadata = json.load(f)
        pctx.cache_metadata = cache_metadata

    return f

def hook(hook_variable):
    def decorator(f):
        pctx.hooks[hook_variable].append(f)

        hook_hash = hash(f'{hook_variable}-{inspect.getsource(f)}')
        pctx.new_cache_metadata['hook-hashes'].append(hook_hash)
        if hook_hash not in pctx.cache_metadata['hook-hashes']:
            pctx.hooks_cached = False

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
    return decorator

# Hash the function
# If hooks match and function hash is in cache, return the cached pctx
def checkpoint(f):
    # later checkpoints need to be a function of earlier checkpoints
    last_hash = pctx.new_cache_metadata['checkpoint-hashes'][-1] if len(pctx.new_cache_metadata['checkpoint-hashes']) > 0 else ''
    func_hash = hash(f'{last_hash}-{inspect.getsource(f)}')
    pctx.new_cache_metadata['checkpoint-hashes'].append(func_hash)

    with open(f'{pctx.cache_dir}/{pctx.key}-cache-metadata.json', 'w') as f:
        json.dump(pctx.new_cache_metadata, f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        if pctx.hooks_cached and func_hash in pctx.cache_metadata['checkpoint-hashes']:
            pctx.cache = read_cache_data(func_hash, pctx)
            return pctx
        pctx = f(*args, **kwargs)
        write_cache_data(func_hash, pctx)
        return pctx
    return wrapper

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
        path = f'{pctx.cache_dir}/{pctx.key}-vis-{pctx.vis_num}-data.arrow'
        pq.write_table(table, path)

        metadata = { k: v for k, v in data.items() if k != 'cols' }
        metadata['key'] = json.loads(pctx.key)

        metadata_path = f'{pctx.cache_dir}/{pctx.key}-vis-{pctx.vis_num}-metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        pctx.vis_num += 1
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

def write_cache_data(checkpt_hash, pctx):
    tensors = { k: v for k, v in pctx.cache.items() if isinstance(v, torch.Tensor) }
    other = { k: v for k, v in pctx.cache.items() if not isinstance(v, torch.Tensor) }

    save_file(tensors, f'{pctx.key}-{checkpt_hash}-tensors.safetensors')

    with open(f'{pctx.key}-{checkpt_hash}-non-tensors.json', 'w') as f:
        json.dump(other, f)

def read_cache_data(checkpt_hash, pctx):
    tensors = safe_open(f'{pctx.key}-{checkpt_hash}-tensors.safetensors')
    with open(f'{checkpt_hash}-other.json', 'r') as f:
        other = json.load(f)

    pctx = { **tensors, **other }

    return pctx
    
def hash(s):
    return hashlib.sha256(s.encode()).hexdigest()
