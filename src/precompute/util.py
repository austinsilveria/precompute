import inspect

def getclosure(name, func, stack=[]):
    if inspect.ismodule(func):
        # e.g. import torch
        if name == func.__name__:
            p = f'import {func.__name__}'
            if p not in stack:
                stack.append(f'import {func.__name__}')
        else:
            p = f'import {func.__name__} as {name}'
            if p not in stack:
                stack.append(f'import {func.__name__} as {name}')
        return
    if inspect.isfunction(func):
        # e.g. from einops.einops import rearrange
        # or referencing a function defined in the local project
        try:
            path = inspect.getfile(func)
            if 'site-packages' in path:
                p = f'from {inspect.getmodule(func).__name__} import {func.__name__}'
                if p not in stack:
                    stack.append(f'from {inspect.getmodule(func).__name__} import {func.__name__}')
                return
        except:
            return
        p = inspect.getsource(func)
        if p not in stack:
            stack.append(inspect.getsource(func))

        # ClosureVars named tuple with nonlocals, globals, builtins, and unbound
        closure = inspect.getclosurevars(func)

        for n, obj in closure.nonlocals.items():
            getclosure(n, obj, stack)
        for n, obj in closure.globals.items():
            getclosure(n, obj, stack)
        return

    # e.g. SCALE = 2
    p = f'\n{name} = {func}'
    if p not in stack:
        stack.append(p)