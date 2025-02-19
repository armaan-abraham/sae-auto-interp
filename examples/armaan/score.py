# %%
from nnsight import LanguageModel
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
import importlib
import sae_auto_interp.features.loader
importlib.reload(sae_auto_interp.features.loader)
from sae_auto_interp.features.loader import FeatureDataset
from sae_auto_interp.features.loader import (
    FeatureLoader
)
import torch
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.explainers import DefaultExplainer
import sae_auto_interp.explainers.explainer
importlib.reload(sae_auto_interp.explainers.explainer)
from pathlib import Path
import asyncio
from sae_auto_interp.explainers import explanation_loader
import sae_auto_interp.scorers.classifier.classifier
importlib.reload(sae_auto_interp.scorers.classifier.classifier)
import sae_auto_interp.scorers.classifier.fuzz
from sae_auto_interp.scorers import DetectionScorer
from sae_auto_interp.clients import OpenRouter
import os
import orjson
import pkgutil
import importlib
import sae_auto_interp
import torch


model_orig = LanguageModel("gpt2", device_map="cuda", dispatch=True)
# Later tokenization functions will assume that EOS token != PAD token
model_orig.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

# %%

layer = 9
site = "resid_pre"
submodule_path = f"layers.{layer}.{site}"

# %%
submodule = model_orig.transformer.h[layer-1]

# %%
import importlib
import mlsae.model.model
importlib.reload(mlsae.model.model)
from mlsae.model import DeepSAE

arch_name = "2-2"
arch_name_to_id = {
    "0-0": "mildly-good-bear",
    "2-2": "only-suited-cat",
}
sae = DeepSAE.load(arch_name, load_from_s3=False, model_id=arch_name_to_id[arch_name]).eval()

sae.start_act_stat_tracking()

# %%
from functools import partial
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents


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

# %%

with model_orig.edit(" ") as model:
    acts = submodule.output[0]
    submodule.ae(acts, hook=True)

# %%

import importlib
import sae_auto_interp.utils
import transformer_lens.utils
importlib.reload(sae_auto_interp.utils)
importlib.reload(transformer_lens.utils)
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data

cfg = CacheConfig(
    dataset_repo="allenai/c4",
    dataset_split="train",
    batch_size=8,
    ctx_len=64,
    n_tokens=1_000_000,
    n_splits=5,
    dataset_row="text",
)

# %%

tokens = torch.load("/root/sae-auto-interp/examples/armaan/tokens.pt")

# %%
def reload_recursively(package):
    """Recursively reload all submodules of a package"""
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        module = importlib.import_module(full_name)
        if is_pkg:
            reload_recursively(module)
        else:
            importlib.reload(module)

reload_recursively(sae_auto_interp)


# %%

feature_cfg = FeatureConfig(
    width=sae.sparse_dim, # The number of latents of your SAE
    min_examples=50, # The minimum number of examples to consider for the feature to be explained
    max_examples=10000, # The maximum number of examples to be sampled from
    n_splits=5 # How many splits was the cache split into
)

# %%
# feature_dict = {submodule_path: alive_features[:10]} # What latents to explain
alive_features = torch.load(f"alive_features_{arch_name}.pt")
feature_dict = {submodule_path: alive_features[:5]}

dataset = FeatureDataset(
        raw_dir=f"latents_{arch_name}", # The folder where the cache is stored
        cfg=feature_cfg,
        modules=[submodule_path],
        features=feature_dict,
        tokenizer=model.tokenizer,
        tokens=tokens,
)

# %%

client = OpenRouter("anthropic/claude-3.5-sonnet", api_key=os.environ["OPENROUTER_API_KEY"])


# %%

experiment_cfg = ExperimentConfig(
    n_examples_test=10, # Number of examples to sample for testing
    n_random=10,
    n_quantiles=4, # Number of quantiles to sample
    example_ctx_len=64, # Length of each example
    test_type="quantiles", # Type of sampler to use for testing. 

    n_examples_train=0,
)
constructor=partial(
            default_constructor,
            tokens=dataset.tokens,
            n_random=experiment_cfg.n_random, 
            ctx_len=experiment_cfg.example_ctx_len, 
            max_examples=feature_cfg.max_examples
        )
sampler=partial(sample,cfg=experiment_cfg)
loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)

# %%
EXPLANATION_DIR = Path(os.getcwd()) / f"explanations_{arch_name}"

# Load the explanations already generated
explainer_pipe = partial(explanation_loader, explanation_dir=EXPLANATION_DIR)


# Builds the record from result returned by the pipeline
def scorer_preprocess(result):
    print("Scorer preprocess")
    record = result.record   
    record.explanation = result.explanation
    record.extra_examples = record.random_examples
    return record

SCORE_DIR = Path(os.getcwd()) / f"scores_{arch_name}"
SCORE_DIR.mkdir(parents=True, exist_ok=True)

# Saves the score to a file
def scorer_postprocess(result):
    print("Scorer postprocess")
    with open(SCORE_DIR / f"{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))

scorer_pipe = process_wrapper(
    DetectionScorer(client, tokenizer=dataset.tokenizer),
    preprocess=scorer_preprocess,
    postprocess=scorer_postprocess,
)

# %%

pipeline = Pipeline(
    loader,
    explainer_pipe,
    scorer_pipe,
)
number_of_parallel_latents = 20
asyncio.run(pipeline.run(number_of_parallel_latents))

# %%
