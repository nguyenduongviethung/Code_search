"""
Embedding extraction pipeline.

Supports:
- Sentence-level embedding extraction
- Token-level embedding extraction
- Optional token compression
"""

import os
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy.lib.format import open_memmap

from utils import load_json
from models.embedding_models import get_embedding_model, ModelEmbedding


# ============================================================
# Dataset
# ============================================================

def get_datasets(
    args: argparse.Namespace,
    data_file: str,
):
    """
    Build DataLoader for a dataset using sequential sampling.
    """
    data = load_json(data_file)

    return DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )


# ============================================================
# Path helpers
# ============================================================

def get_data_path(
    args: argparse.Namespace,
    lang: str,
    format: str,
):
    """
    Build path to dataset file.
    """
    return os.path.join(
        args.data_root,
        lang,
        f"{lang}_{format}.json",
    )


def get_model_name(model_path: str):
    """
    Convert model path into a stable directory name.

    Examples:
        microsoft/unixcoder-base
            -> microsoft/unixcoder-base

        /path/to/unixcoder-base
            -> to/unixcoder-base
    """
    from pathlib import Path

    path = Path(model_path)

    if path.exists():
        return str(Path(path.parent.name) / path.name)

    return model_path


def get_emb_path(
    args: argparse.Namespace,
    lang: str,
    model: str,
    format: str,
):
    """
    Build path to sentence-level embedding file.
    """
    return os.path.join(
        args.emb_root,
        model,
        lang,
        f"{lang}_{format}_emb.npy",
    )


def get_token_emb_path(
    args: argparse.Namespace,
    lang: str,
    model: str,
    format: str,
):
    """
    Build path to token-level embedding file.
    """
    return os.path.join(
        args.emb_root,
        model,
        lang,
        f"{lang}_{format}_token_emb.npy",
    )


def get_mask_path(
    args: argparse.Namespace,
    lang: str,
    model: str,
    format: str,
):
    """
    Build path to attention mask file.
    """
    return os.path.join(
        args.emb_root,
        model,
        lang,
        f"{lang}_{format}_mask.npy",
    )


# ============================================================
# Sentence-level embeddings
# ============================================================

@torch.no_grad()
def get_embeddings(
    logger: logging.Logger,
    args: argparse.Namespace,
    embedding_model: ModelEmbedding,
    dataloader: DataLoader,
    is_nl: bool,
    save_vector_path: str | None = None,
):
    """
    Compute sentence-level embeddings for the entire dataset.

    Returns:
        np.ndarray:
            Shape = (num_samples, hidden_size)
    """
    vecs = []

    for batch in tqdm(dataloader):
        embeds = embedding_model.get_embedding(
            args.device,
            batch["nl"] if is_nl else batch["code"],
            args.nl_length if is_nl else args.code_length,
        )

        vecs.append(
            embeds.cpu().numpy()
        )

    vecs = np.concatenate(vecs, axis=0)

    if save_vector_path:
        logger.info(
            f"Saving vector to {save_vector_path} "
            f"{vecs.shape}"
        )
        np.save(save_vector_path, vecs)

    return vecs


# ============================================================
# Token compression
# ============================================================

def compress_tokens(
    x: torch.Tensor,
    mask: torch.Tensor,
    n_chunks: int = 8,
):
    """
    Compress token embeddings into fixed-size chunks.

    Args:
        x:
            (B, L, D)

        mask:
            (B, L)

        n_chunks:
            Number of chunks.

    Returns:
        pooled:
            (B, n_chunks, D)

        valid:
            (B, n_chunks)
    """
    B, L, D = x.shape

    chunk_size = L // n_chunks

    xs = []
    ms = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, L)

        x_chunk = x[:, start:end]
        m_chunk = mask[:, start:end]

        m = m_chunk.unsqueeze(-1)

        pooled = (
            (x_chunk * m).sum(dim=1)
            / m.sum(dim=1).clamp(min=1)
        )

        valid = m_chunk.sum(dim=1) > 0

        xs.append(pooled)
        ms.append(valid)

    return (
        torch.stack(xs, dim=1),
        torch.stack(ms, dim=1),
    )


# ============================================================
# Token-level embeddings
# ============================================================

@torch.no_grad()
def get_token_embeddings(
    logger: logging.Logger,
    args: argparse.Namespace,
    embedding_model: ModelEmbedding,
    dataloader: DataLoader,
    is_nl: bool,
    token_emb_path: str,
    mask_path: str,
):
    """
    Compute token-level embeddings and attention masks.

    Embeddings and masks are stored using memory-mapped arrays
    to avoid keeping the entire dataset in RAM.

    Returns:
        token_embeddings:
            (num_samples, seq_length, hidden_size)

        attention_masks:
            (num_samples, seq_length)
    """
    dataset_size = len(dataloader.dataset)  # type: ignore

    if args.n_chunks:
        seq_len = args.n_chunks
    else:
        seq_len = (
            args.nl_length
            if is_nl
            else args.code_length
        )

    # --------------------------------------------------------
    # Infer hidden size from first batch
    # --------------------------------------------------------

    first_batch = next(iter(dataloader))

    token_embs, _ = embedding_model.get_token_embedding(
        args.device,
        first_batch["nl"] if is_nl else first_batch["code"],
        args.nl_length if is_nl else args.code_length,
    )

    hidden_size = token_embs.shape[-1]

    # --------------------------------------------------------
    # Allocate memory-mapped arrays
    # --------------------------------------------------------

    token_arr = open_memmap(
        token_emb_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            dataset_size,
            seq_len,
            hidden_size,
        ),
    )

    mask_arr = open_memmap(
        mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(
            dataset_size,
            seq_len,
        ),
    )

    # --------------------------------------------------------
    # Extract embeddings
    # --------------------------------------------------------

    offset = 0

    for batch in tqdm(dataloader):
        token_embs, masks = embedding_model.get_token_embedding(
            args.device,
            batch["nl"] if is_nl else batch["code"],
            args.nl_length if is_nl else args.code_length,
        )

        if args.n_chunks:
            token_embs, masks = compress_tokens(
                token_embs,
                masks,
                n_chunks=args.n_chunks,
            )

        batch_size = token_embs.shape[0]

        token_arr[
            offset:offset + batch_size
        ] = token_embs.cpu().numpy().astype(np.float16)

        mask_arr[
            offset:offset + batch_size
        ] = masks.cpu().numpy()

        offset += batch_size

    token_arr.flush()
    mask_arr.flush()

    logger.info(
        f"Saved token embeddings to {token_emb_path} "
        f"and masks to {mask_path}"
    )


# ============================================================
# Embedding extraction
# ============================================================

def extract_sentence_embeddings(
    args: argparse.Namespace,
    logger: logging.Logger,
):
    """
    Extract sentence-level embeddings for all requested
    languages, formats, and models.
    """
    n_gpu = torch.cuda.device_count()

    langs = args.langs.split(",")
    formats = args.formats.split(",")
    model_paths = args.model_paths.split(",")

    for lang in langs:
        for model_path in model_paths:

            model_name = get_model_name(model_path)

            logger.info(
                f"\n===== Loading model {model_name} ====="
            )

            embedding_model = get_embedding_model(
                n_gpu,
                args.device,
                model_path,
            )

            for format in formats:

                logger.info(
                    f"\n===== Embedding "
                    f"{lang} {format} "
                    f"with model {model_name} ====="
                )

                data_path = get_data_path(
                    args,
                    lang,
                    format,
                )

                dataloader = get_datasets(
                    args,
                    data_path,
                )

                is_nl = (
                    "query" in format
                    or "comment" in format
                )

                save_path = get_emb_path(
                    args,
                    lang,
                    model_name,
                    format,
                )

                os.makedirs(
                    os.path.dirname(save_path),
                    exist_ok=True,
                )

                get_embeddings(
                    logger,
                    args,
                    embedding_model,
                    dataloader,
                    is_nl,
                    save_vector_path=save_path,
                )


def extract_token_embeddings(
    args: argparse.Namespace,
    logger: logging.Logger,
):
    """
    Extract token-level embeddings and attention masks
    for all requested languages, formats, and models.
    """
    n_gpu = torch.cuda.device_count()

    langs = args.langs.split(",")
    formats = args.formats.split(",")
    model_paths = args.model_paths.split(",")

    for lang in langs:
        for model_path in model_paths:

            model_name = get_model_name(model_path)

            logger.info(
                f"\n===== Loading model {model_name} ====="
            )

            embedding_model = get_embedding_model(
                n_gpu,
                args.device,
                model_path,
            )

            for format in formats:

                logger.info(
                    f"\n===== Token Embedding "
                    f"{lang} {format} "
                    f"with model {model_name} ====="
                )

                data_path = get_data_path(
                    args,
                    lang,
                    format,
                )

                dataloader = get_datasets(
                    args,
                    data_path,
                )

                is_nl = (
                    "query" in format
                    or "comment" in format
                )

                token_emb_path = get_token_emb_path(
                    args,
                    lang,
                    model_name,
                    format,
                )

                mask_path = get_mask_path(
                    args,
                    lang,
                    model_name,
                    format,
                )

                os.makedirs(
                    os.path.dirname(token_emb_path),
                    exist_ok=True,
                )

                get_token_embeddings(
                    logger,
                    args,
                    embedding_model,
                    dataloader,
                    is_nl,
                    token_emb_path=token_emb_path,
                    mask_path=mask_path,
                )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from an embedding model."
    )

    parser.add_argument(
        "--langs",
        type=str,
        default="cosqa,solidity,sql",
    )

    parser.add_argument(
        "--formats",
        type=str,
        default="query,code,comment,gencode",
    )

    parser.add_argument(
        "--model_paths",
        type=str,
        default="microsoft/unixcoder-base",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--emb_root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--nl_length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--code_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--n_chunks",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "embedding",
            "token_embedding"
        ],
        default="embedding",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(__name__)

    logger.info(f"args: {args}")

    n_gpu = torch.cuda.device_count()
    logger.info(f"n_gpu: {n_gpu}")

    if args.mode == "token_embedding":
        extract_token_embeddings(args, logger)
    else:
        extract_sentence_embeddings(args, logger)


if __name__ == "__main__":
    main()
