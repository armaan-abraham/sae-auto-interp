from sae_auto_interp.config import ExperimentConfig
import torch
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
import asyncio
from sae_auto_interp.explainers import explanation_loader
from sae_auto_interp.scorers import FuzzingScorer, DetectionScorer
from sae_auto_interp.clients import OpenRouter
import os
import orjson
from functools import partial
from utils import load_feature_dataset, data_dir, cfg, arch_name_to_id
from generate_explanations import explanation_dir
from sae_auto_interp.features.loader import FeatureLoader


score_dir = data_dir / "scores"

scorers = {
    "detection": DetectionScorer,
    "fuzz": FuzzingScorer,
}


def generate_scores(arch_name, scorer_name):
    experiment_cfg = ExperimentConfig(
        n_examples_test=10,  # Number of examples to sample for testing
        n_random=10,
        n_quantiles=4,  # Number of quantiles to sample
        example_ctx_len=64,  # Length of each example
        test_type="quantiles",  # Type of sampler to use for testing.
        n_examples_train=0,
    )
    tokens = torch.load(data_dir / "tokens.pt")
    constructor = partial(
        default_constructor,
        tokens=tokens,
        n_random=experiment_cfg.n_random,
        ctx_len=experiment_cfg.example_ctx_len,
        max_examples=10000,
    )
    sampler = partial(sample, cfg=experiment_cfg)

    dataset, feature_cfg = load_feature_dataset(arch_name)

    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)

    # Load the explanations already generated
    this_explanation_dir = explanation_dir / arch_name
    explainer_pipe = partial(explanation_loader, explanation_dir=this_explanation_dir)

    # Builds the record from result returned by the pipeline
    def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.random_examples
        return record

    this_score_dir = score_dir / scorer_name / arch_name
    this_score_dir.mkdir(parents=True, exist_ok=True)

    # Saves the score to a file
    def scorer_postprocess(result):
        with open(this_score_dir / f"{result.record.feature}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    client = OpenRouter(
        "anthropic/claude-3.5-sonnet", api_key=os.environ["OPENROUTER_API_KEY"]
    )

    scorer_pipe = process_wrapper(
        scorers[scorer_name](client, tokenizer=dataset.tokenizer, batch_size=1),
        preprocess=scorer_preprocess,
        postprocess=scorer_postprocess,
    )

    pipeline = Pipeline(
        loader,
        explainer_pipe,
        scorer_pipe,
    )
    number_of_parallel_latents = 20
    asyncio.run(pipeline.run(number_of_parallel_latents))

if __name__ == "__main__":
    # generate_scores("2-4-4-2", "detection")
    generate_scores("2-4-4-2", "fuzz")
