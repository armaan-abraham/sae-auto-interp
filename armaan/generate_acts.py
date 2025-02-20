from nnsight import LanguageModel
import torch
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from functools import partial
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
from utils import (
    load_sae,
    cfg,
    save_sae_config,
    latents_dir,
    data_dir,
    alive_features_dir,
    arch_name_to_id,
)


def generate_acts(arch_name):
    model_orig = LanguageModel("gpt2", device_map="cpu", dispatch=True)
    # Later tokenization functions will assume that EOS token != PAD token
    model_orig.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    submodule = model_orig.transformer.h[cfg.layer - 1]

    sae = load_sae(arch_name)

    sae.start_act_stat_tracking()

    def _forward(sae, x):
        x -= x.mean(dim=-1, keepdim=True)
        x /= x.norm(dim=-1, keepdim=True)
        return sae(x)[3]

    autoencoder_latents = AutoencoderLatents(
        sae,
        partial(_forward, sae),
        width=sae.sparse_dim,
    )

    submodule.ae = autoencoder_latents

    with model_orig.edit(" ") as model:
        acts = submodule.output[0]
        submodule.ae(acts, hook=True)

    cache_cfg = CacheConfig(
        dataset_repo="allenai/c4",
        dataset_split="train",
        batch_size=16,
        ctx_len=64,
        n_tokens=1_000_000,
        n_splits=cfg.cache_splits,
        dataset_row="text",
    )

    tokens = torch.load(data_dir / "tokens.pt")

    submodule_dict = {cfg.submodule_path: submodule}

    model.to("cuda")
    cache = FeatureCache(
        model,
        submodule_dict,
        batch_size=cache_cfg.batch_size,
    )

    cache.run(cache_cfg.n_tokens, tokens)

    print(sae.get_activation_stats())

    this_latents_dir = latents_dir / arch_name
    this_latents_dir.mkdir(parents=True, exist_ok=True)

    cache.save_splits(n_splits=cache_cfg.n_splits, save_dir=this_latents_dir)

    cache.save_config(
        save_dir=this_latents_dir,
        cfg=cache_cfg,
        model_name="gpt2"
    )

    # Record and save dead features
    alive_features_dir.mkdir(parents=True, exist_ok=True)
    alive_features = cache.cache.feature_locations[cfg.submodule_path][:, 2].unique()
    torch.save(alive_features, alive_features_dir / f"{arch_name}.pt")

    save_sae_config(arch_name, sae)


if __name__ == "__main__":
    for arch in arch_name_to_id.keys():
        generate_acts(arch)