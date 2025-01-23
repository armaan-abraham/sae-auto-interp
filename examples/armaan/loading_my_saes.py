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

arch_name = "12"
# sae = DeepSAE.load("5", load_from_s3=True, model_id="duly-needed-dassie").eval()
# sae = DeepSAE.load("9", load_from_s3=True, model_id="safely-bright-kit").eval()
sae = DeepSAE.load(arch_name, load_from_s3=True, model_id="hardly-quick-dingo").eval()
sae.start_act_stat_tracking()

# %%
from functools import partial
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents


def _forward(sae, x):
    return sae(x * 10)[4]

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
    dataset_row="text",
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
# %%

cache.run(cfg.n_tokens, tokens)

# %%

print(sae.get_activation_stats())

# %%

cache.save_splits(
    n_splits=cfg.n_splits,  # We split the activation and location indices into different files to make loading faster
    save_dir=f"latents_{arch_name}"
)

# %%
# The config of the cache should be saved with the results such that it can be loaded later.

cache.save_config(
    save_dir=f"latents_{arch_name}",
    cfg=cfg,
    model_name="roneneldan/TinyStories-3M"
)

# %%
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader
)
import torch

feature_cfg = FeatureConfig(
    width=sae.sparse_dim, # The number of latents of your SAE
    min_examples=50, # The minimum number of examples to consider for the feature to be explained
    max_examples=10000, # The maximum number of examples to be sampled from
    n_splits=5 # How many splits was the cache split into
)

# %%
feature_dict = {submodule_path: torch.arange(0,50)} # The what latents to explain

dataset = FeatureDataset(
        raw_dir="latents", # The folder where the cache is stored
        cfg=feature_cfg,
        modules=[submodule_path],
        features=feature_dict,
        tokenizer=model.tokenizer,
)

# %%

experiment_cfg = ExperimentConfig(
    n_examples_train=40, # Number of examples to sample for training
    example_ctx_len=32, # Length of each example
    train_type="top", # Type of sampler to use for training. 
)

# %%
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample

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
from sae_auto_interp.clients import OpenRouter
import os
import orjson

client = OpenRouter("anthropic/claude-3.5-sonnet", api_key=os.environ["OPENROUTER_API_KEY"])

# %%
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.explainers import DefaultExplainer
from pathlib import Path

EXPLANATION_DIR = Path(os.getcwd()) / "results" / "explanations"
EXPLANATION_DIR.mkdir(parents=True, exist_ok=True)

# The function that saves the explanations
def explainer_postprocess(result):
    with open(EXPLANATION_DIR / f"{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.explanation))
    del result
    return None

explainer_pipe = process_wrapper(
        DefaultExplainer(
            client, 
            tokenizer=dataset.tokenizer,
        ),
        postprocess=explainer_postprocess,
    )

# %%
import asyncio
import nest_asyncio
nest_asyncio.apply()

pipeline = Pipeline(
    loader,
    explainer_pipe,
)
number_of_parallel_latents = 10
asyncio.run(pipeline.run(number_of_parallel_latents)) # This will start generating the explanations.

# %%

experiment_cfg = ExperimentConfig(
    n_examples_test=20, # Number of examples to sample for testing
    n_quantiles=10, # Number of quantiles to sample
    example_ctx_len=32, # Length of each example
    test_type="quantiles", # Type of sampler to use for testing. 
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

from sae_auto_interp.explainers import explanation_loader
from sae_auto_interp.scorers import FuzzingScorer

# Load the explanations already generated
explainer_pipe = partial(explanation_loader, explanation_dir=EXPLANATION_DIR)


# Builds the record from result returned by the pipeline
def scorer_preprocess(result):
    record = result.record   
    record.explanation = result.explanation
    record.extra_examples = record.random_examples
    return record

SCORE_DIR = Path(os.getcwd()) / "results" / "scores"
SCORE_DIR.mkdir(parents=True, exist_ok=True)

# Saves the score to a file
def scorer_postprocess(result, score_dir):
    with open(SCORE_DIR / f"{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))

scorer_pipe = process_wrapper(
    FuzzingScorer(client, tokenizer=dataset.tokenizer),
    preprocess=scorer_preprocess,
    postprocess=partial(scorer_postprocess, score_dir="fuzz"),
)

# %%

pipeline = Pipeline(
    loader,
    explainer_pipe,
    scorer_pipe,
)
number_of_parallel_latents = 10
asyncio.run(pipeline.run(number_of_parallel_latents))