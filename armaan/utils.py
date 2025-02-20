from dataclasses import dataclass
import json
from pathlib import Path
from mlsae.model import DeepSAE
import torch
from sae_auto_interp.config import FeatureConfig
from sae_auto_interp.features.loader import FeatureDataset
from transformers import AutoTokenizer


@dataclass
class Config:
    layer: int = 9
    site: str = "resid_pre"
    cache_splits: int = 10

    @property
    def submodule_path(self):
        return f"layers.{self.layer}.{self.site}"

    experiment_name: str = "gpt2"


cfg = Config()

base_data_dir = Path(__file__).parent.parent / "data"

data_dir = base_data_dir / cfg.experiment_name

results_dir = Path(__file__).parent.parent / "results" / cfg.experiment_name

arch_name_to_id = {
    "0-0": "mildly-good-bear",
    "2-2": "only-suited-cat",
}

def load_sae(arch_name):
    sae = DeepSAE.load(
        arch_name, load_from_s3=True, model_id=arch_name_to_id[arch_name]
    ).eval()
    return sae


def load_feature_config(arch_name):
    config = load_sae_config(arch_name)
    feature_cfg = FeatureConfig(
        width=config["width"],  # The number of latents of your SAE
        min_examples=50,  # The minimum number of examples to consider for the feature to be explained
        max_examples=10000,  # The maximum number of examples to be sampled from
        n_splits=cfg.cache_splits,  # How many splits was the cache split into
    )
    return feature_cfg


alive_features_dir = data_dir / "alive_features"


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    return tokenizer


def load_feature_dataset(arch_name):
    feature_cfg = load_feature_config(arch_name)

    num_features = 250
    alive_features = torch.load(alive_features_dir / f"{arch_name}.pt")
    feature_dict = {cfg.submodule_path: alive_features[num_features:(num_features * 2)]}

    tokens = torch.load(data_dir / "tokens.pt")

    dataset = FeatureDataset(
        raw_dir=latents_dir / arch_name,
        cfg=feature_cfg,
        modules=[cfg.submodule_path],
        features=feature_dict,
        tokenizer=load_tokenizer(),
        tokens=tokens,
    )
    return dataset, feature_cfg


latents_dir = data_dir / "latents"


def save_sae_config(arch_name, sae):
    with open(latents_dir / arch_name / "sae_config.json", "w") as f:
        json.dump({"width": sae.sparse_dim}, f)


def load_sae_config(arch_name):
    with open(latents_dir / arch_name / "sae_config.json", "r") as f:
        return json.load(f)

current_arch = "2-2"