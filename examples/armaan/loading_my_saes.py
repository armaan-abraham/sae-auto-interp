# %%
from nnsight import LanguageModel

model_orig = LanguageModel("roneneldan/TinyStories-3M", device_map="cuda", dispatch=True)

# %%

layer = 6
site = "resid_pre"
submodule_path = f"layers.{layer}.{site}"

# %%
submodule = model_orig.transformer.h[layer-1]

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

with model_orig.edit(" ") as model:
    acts = submodule.output[0]
    submodule.ae(acts, hook=True)

# %%

from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data

cfg = CacheConfig(
    dataset_repo="roneneldan/TinyStories",
    dataset_split="train[:1%]",
    batch_size=8,
    ctx_len=64,
    n_tokens=1_000_000,
    n_splits=5,
)

# %%

tokens = load_tokenized_data(
    ctx_len=cfg.ctx_len,
    tokenizer=model.tokenizer,
    dataset_repo=cfg.dataset_repo,
    dataset_split=cfg.dataset_split,
    dataset_row="text",
)

# %%

submodule_dict = {submodule_path: submodule}

cache = FeatureCache(
    model,
    submodule_dict,
    batch_size = cfg.batch_size,
)

cache.run(cfg.n_tokens, tokens)
# %%
