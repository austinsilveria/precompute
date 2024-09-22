import torch
import precompute as pre


model: SupportedModel
x: torch.Tensor

# Key, Hook, Run, Vis
# 
# Key the experiment.
# Hook into the model activations.
# Run a function of the model.
# Visualize the accumulated results.
#
# Easily copied key (with highlighted diff from previous experiment) shown as a collapsible left sidebar on the visualizations
#     return {
#         'model': 'facebook/opt-125m',
#         'filter': 'low-pass',
#     }
#
# Serializable key and cache for sharing
pre.key({
    'model': 'facebook/opt-125m',
    'filter': 'low-pass',
})

# Registers hook function
@pre.hook('post-mlp')
def log_post_mlp(x, pctx):
    pass

@pre.checkpoint(cache='./cache')
def run(model, pctx):
    # Call model as usual, pctx is passed under the hood
    out = model(x)

    # Iterative visualization during training

    return pctx

@pre.checkpoint(cache='./cache')
def second_run(model, pctx):
    out2 = model(x)

    return pctx

# Instantiate hooked model and context
model, pctx = pre.model(model)

pctx = run(model, pctx)
pctx = second_run(model, pctx)

# pctx now serialized and cached
# Skips if executed again with the same key, hooks, and run

@pre.vis
def power_spectrum(title, x, pctx):
    cols = { 'Frequency': f, 'Power': {} }
    for i in range(pctx.config.num_hidden_layers):
        cols['Power'][f'Layer {i}'] = x[i]

    # Write instance specific notes in the studio
    # Link to specific instances in posts (see context of other experiments at the time)
    return {
        'title': title,
        'type': 'line',
        'cols': cols,

        'args': {
            # Pass through to visualization interface
        },
    }

power_spectrum('Hidden States Power Spectrum', pctx.context['post-mlp-activations'], pctx)
power_spectrum('Attention Residuals Power Spectrum', pctx.context['attn-residual'], pctx)

@pre.vis
def loss(title, x, groupkey, pctx):
    return {
        'title': title,
        'type': 'line',
        'cols': { 'Tokens': pctx.training_log['tokens'], 'Loss': x },

        # Merge columns into one visualization across multiple experiments
        'groupkey': groupkey,
    }

loss('LM Loss', pctx.log['lm-loss'], 'lm-loss', pctx)