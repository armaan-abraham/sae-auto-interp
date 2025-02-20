from mlsae.data import stream_training_chunks
from mlsae.config import data_cfg
import torch
from utils import data_dir


def generate_tokens():
    MAX_TOKENS = 1_000_000

    iterator = stream_training_chunks(
        dataset_batch_size_entries=2,
        act_block_size_seqs=MAX_TOKENS // data_cfg.seq_len + 2,
    )

    chunks = []
    num_tokens = 0

    while num_tokens < MAX_TOKENS:
        chunk = next(iterator)
        chunks.append(chunk)
        num_tokens += chunk.numel()

    tokens = torch.cat(chunks).flatten()
    tokens = tokens[:MAX_TOKENS]
    print(f"{tokens.shape=}")

    data_dir.mkdir(parents=True, exist_ok=True)

    torch.save(tokens, data_dir / "tokens.pt")


if __name__ == "__main__":
    generate_tokens()
