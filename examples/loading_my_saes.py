# %%
from nnsight import LanguageModel

model = LanguageModel("roneneldan/TinyStories-3M", device_map="cuda", dispatch=True)

# %%

layer = 6
site = "resid_pre"

# %%
submodule = model.transformer.h[layer]

# %%
from mlsae.model import DeepSAE

sae = DeepSAE.load("9", load_from_s3=True, model_id="safely-bright-kit").eval()

# %%
from functools import partial
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents

def _forward(sae, x):
    return sae(x)[4]

autoencoder_latents = AutoencoderLatents(
    sae,
    partial(_forward, sae),
    width=sae.sparse_dim,
)


# %%

submodule.ae = autoencoder_latents
# %%
with model.edit(" "):
    acts = submodule.input[0]
    submodule.ae(acts, hook=True)

# %%

from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data

cfg = CacheConfig(
    dataset_repo="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
    dataset_split="train[:1%]",
    batch_size=8,
    ctx_len=64,
    n_tokens=1_000_000,
    n_splits=5,
)
