# %%
from nnsight import LanguageModel
import torch

model_orig = LanguageModel("gpt2", device_map="cpu", dispatch=True)
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

arch_name = "0-0"
exp_name = "1"
arch_name_to_id = {
    "0-0": "mildly-good-bear",
    "2-2": "only-suited-cat",
}
sae = DeepSAE.load(arch_name, load_from_s3=True, model_id=arch_name_to_id[arch_name]).eval()

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
    n_splits=10,
    dataset_row="text",
)

# %%

from mlsae.data import stream_training_chunks


iterator = stream_training_chunks(dataset_batch_size_entries=2, act_block_size_seqs=2 ** 15)


# %%
import torch

MAX_TOKENS = 1_000_000
chunks = []
num_tokens = 0

while num_tokens < MAX_TOKENS:
    chunk = next(iterator)
    chunks.append(chunk)
    num_tokens += chunk.numel()

tokens = torch.cat(chunks)

print(tokens.shape)

# %%

torch.save(tokens, f"tokens.pt")

# %%

tokens = torch.load("tokens.pt")

# %%

tokens_2 = load_tokenized_data(
    ctx_len=cfg.ctx_len,
    tokenizer=model_orig.tokenizer,
    dataset_repo=cfg.dataset_repo,
    dataset_split=cfg.dataset_split,
    dataset_row="text",
)

# %%

print(tokens_2.shape)

# %%

submodule_dict = {submodule_path: submodule}

model.to("cuda")
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
def upload_latents(arch_name):
    import shutil
    # Name of the local folder
    folder = f"latents_{arch_name}"
    # Create a zip archive of the folder
    archive_name = f"{folder}.zip"
    shutil.make_archive(folder, 'zip', folder)
    import boto3
    s3_client = boto3.client("s3")
    s3_client.upload_file(f"{folder}.zip", "deep-sae", f"latents/{exp_name}/{arch_name}.zip")

def download_latents(arch_name):
    import boto3
    import zipfile
    s3_client = boto3.client("s3")
    s3_client.download_file("deep-sae", f"latents/{exp_name}/{arch_name}.zip", f"{arch_name}.zip")
    with zipfile.ZipFile(f"{arch_name}.zip", 'r') as zip_ref:
        zip_ref.extractall(f"latents_{arch_name}")

def download_tokens():
    import boto3
    s3_client = boto3.client("s3")
    s3_client.download_file("deep-sae", f"latents/{exp_name}/tokens.pt", "tokens.pt")

# %%
download_latents("0-0")

# %%
download_tokens()

# %%
# The config of the cache should be saved with the results such that it can be loaded later.

cache.save_config(
    save_dir=f"latents_{arch_name}",
    cfg=cfg,
    model_name="gpt2"
)

# %%
# Get dead features
print(cache.cache.feature_locations[submodule_path])
# Get unique feature indices from the third column (index 2) of feature_locations
alive_features = cache.cache.feature_locations[submodule_path][:, 2].unique()
print(f"Number of unique features activated: {len(alive_features)}")
print(f"Number of dead features: {sae.sparse_dim - len(alive_features)}")
torch.save(alive_features, f"alive_features_{arch_name}.pt")

# %%
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
import nest_asyncio
from sae_auto_interp.explainers import explanation_loader
import sae_auto_interp.scorers
import sae_auto_interp.scorers.classifier
import sae_auto_interp.scorers.classifier.classifier
importlib.reload(sae_auto_interp.scorers)
importlib.reload(sae_auto_interp.scorers.classifier)
importlib.reload(sae_auto_interp.scorers.classifier.classifier)
import sae_auto_interp.scorers.classifier.fuzz
importlib.reload(sae_auto_interp.scorers.classifier.fuzz)
from sae_auto_interp.scorers import FuzzingScorer
from sae_auto_interp.clients import OpenRouter
import os
import orjson
import pkgutil
import importlib
import sae_auto_interp

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
# feature_dict = {submodule_path: alive_features[:100]} # What latents to explain
num_features = 100
feature_dict_full = {submodule_path: torch.arange(1000)} # Load more than we need because some may be dead
# feature_dict = {submodule_path: torch.arange(10)}
exp_dir = Path(__file__).parent.parent.parent / "data" / "latents" / exp_name

tokens = torch.load(exp_dir / "tokens.pt")


dataset = FeatureDataset(
        raw_dir=exp_dir / f"latents_{arch_name}", # The folder where the cache is stored
        cfg=feature_cfg,
        modules=[submodule_path],
        features=feature_dict_full,
        tokenizer=model.tokenizer,
        tokens=tokens,
)

# %%
import re

count = 0
for buffer in dataset.buffers[0]:
    if buffer is not None:
        print(buffer.feature)
        count += 1
        assert not buffer.locations.numel() == 0, "Buffer is empty"
        if count == num_features:
            term_feature = int(re.split(r'feature', str(buffer.feature))[1]) + 1
            print(term_feature)
            break
if count < num_features:
    print("Not enough features")

# %%
feature_dict = {submodule_path: torch.arange(term_feature)}

# Load FeatureDataset again now that we know which features are alive
dataset = FeatureDataset(
        raw_dir=exp_dir / f"latents_{arch_name}", # The folder where the cache is stored
        cfg=feature_cfg,
        modules=[submodule_path],
        features=feature_dict,
        tokenizer=model.tokenizer,
        tokens=tokens,
)

count = 0
for buffer in dataset.buffers[0]:
    if buffer is not None:
        print(buffer.feature)
        count += 1
assert count == num_features


# %%

experiment_cfg = ExperimentConfig(
    n_examples_train=100, # Number of examples to sample for training
    example_ctx_len=64, # Length of each example
    train_type="quantiles", # Type of sampler to use for training. 
)

# %%

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

client = OpenRouter("anthropic/claude-3.5-sonnet", api_key=os.environ["OPENROUTER_API_KEY"])

# %%


EXPLANATION_DIR = Path(os.getcwd()).parent.parent / "data" / "explanations" / f"explanations_{arch_name}"
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

nest_asyncio.apply()
pipeline = Pipeline(
    loader,
    explainer_pipe,
)
number_of_parallel_latents = 20
asyncio.run(pipeline.run(number_of_parallel_latents)) # This will start generating the explanations.

# %%

experiment_cfg = ExperimentConfig(
    n_examples_test=20, # Number of examples to sample for testing
    n_random=20,
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
sampler = partial(sample,cfg=experiment_cfg)
loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)

# %%

# Load the explanations already generated
explainer_pipe = partial(explanation_loader, explanation_dir=EXPLANATION_DIR)

# Builds the record from result returned by the pipeline
def scorer_preprocess(result):
    record = result.record   
    record.explanation = result.explanation
    record.extra_examples = record.random_examples
    return record

SCORE_DIR = Path(os.getcwd()) / f"scores_{arch_name}"
SCORE_DIR.mkdir(parents=True, exist_ok=True)

# Saves the score to a file
def scorer_postprocess(result):
    with open(SCORE_DIR / f"{result.record.feature}.txt", "wb") as f:
        f.write(orjson.dumps(result.score))

scorer_pipe = process_wrapper(
    FuzzingScorer(client, tokenizer=dataset.tokenizer, batch_size=1),
    preprocess=scorer_preprocess,
    postprocess=scorer_postprocess,
)

# %%

nest_asyncio.apply()
pipeline = Pipeline(
    loader,
    explainer_pipe,
    scorer_pipe,
)
number_of_parallel_latents = 20
asyncio.run(pipeline.run(number_of_parallel_latents))

# %%
