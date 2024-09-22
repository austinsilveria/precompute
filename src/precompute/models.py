from transformers import (
    GPTNeoXForCausalLM,
    OPTForCausalLM,
)

from precompute import (
    HookedGPTNeoXForCausalLM,
    HookedOPTForCausalLM,
)

# Map of supported models to their hooked models
HOOKED_MODELS = {
    GPTNeoXForCausalLM: HookedGPTNeoXForCausalLM,
    OPTForCausalLM: HookedOPTForCausalLM,
}
