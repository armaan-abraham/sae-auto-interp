from sae_auto_interp.config import ExperimentConfig
from sae_auto_interp.features.loader import FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.explainers import DefaultExplainer
import asyncio
from sae_auto_interp.clients import OpenRouter
import os
import orjson
from functools import partial
from utils import data_dir, load_feature_dataset, arch_name_to_id

explanation_dir = data_dir / "explanations"
explanation_dir.mkdir(parents=True, exist_ok=True)

def generate_explanations(arch_name):
    experiment_cfg = ExperimentConfig(
        n_examples_train=100,  # Number of examples to sample for training
        example_ctx_len=64,  # Length of each example
        train_type="quantiles",  # Type of sampler to use for training.
    )

    dataset, feature_cfg = load_feature_dataset(arch_name)

    constructor = partial(
        default_constructor,
        tokens=dataset.tokens,
        n_random=experiment_cfg.n_random,
        ctx_len=experiment_cfg.example_ctx_len,
        max_examples=feature_cfg.max_examples,
    )
    sampler = partial(sample, cfg=experiment_cfg)
    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)

    client = OpenRouter(
        "anthropic/claude-3.5-sonnet", api_key=os.environ["OPENROUTER_API_KEY"]
    )

    this_explanation_dir = explanation_dir / arch_name
    this_explanation_dir.mkdir(parents=True, exist_ok=True)

    # The function that saves the explanations
    def explainer_postprocess(result):
        with open(this_explanation_dir / f"{result.record.feature}.txt", "wb") as f:
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

    pipeline = Pipeline(
        loader,
        explainer_pipe,
    )
    number_of_parallel_latents = 20
    asyncio.run(pipeline.run(number_of_parallel_latents))


if __name__ == "__main__":
    # generate_explanations("2-2")
    for arch in arch_name_to_id.keys(): generate_explanations(arch)