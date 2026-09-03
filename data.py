import os
import argparse
import logging

import numpy as np
import torch

from utils import load_json


# ============================================================
# Constants
# ============================================================

SPLITS = ("train", "valid", "test")

# ============================================================
# Raw text loading
# ============================================================

def load_text_data(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, list[dict[str, str]]]:
    """
    Load raw text data.

    Returned structure:

        text_data = {
            "query": [...],
            "code": [...],
            "comment": [...],   # optional
            "gencode": [...],   # optional
        }

    Each item contains at least:
        id
        split
        nl / code
    """

    if args.data_root is None:
        return {}

    logger.info(
        "Loading text data from: %s",
        args.data_root,
    )

    fields = ["query", "code"]

    if args.use_comment:
        fields.append("comment")

    if args.use_gencode:
        fields.append("gencode")

    text_data: dict[
        str,
        list[dict[str, str]],
    ] = {}

    for field in fields:
        path = os.path.join(args.data_root, args.lang, f"{args.lang}_{field}.json")

        text_data[field] = load_json(path)

        logger.info(
            "#%s=%d",
            field,
            len(text_data[field]),
        )

    return text_data


# ============================================================
# Raw embedding loading
# ============================================================

def load_embeddings(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, torch.Tensor]:
    """
    Load raw embeddings from disk.

    Returned structure:

        emb_data = {
            "q2c_query": Tensor,
            "q2c_code": Tensor,
            "q2com_query": Tensor,       # optional
            "q2com_comment": Tensor,     # optional
            "c2c_gencode": Tensor,       # optional
            "c2c_code": Tensor,          # optional
        }

    Embedding order is assumed to match the corresponding
    original text modality order.
    """

    if args.emb_root is None:
        return {}

    logger.info(
        "Loading embeddings from: %s",
        args.emb_root,
    )

    fields = [
        "q2c_query",
        "q2c_code",
    ]

    if args.use_comment:
        fields.extend([
            "q2com_query",
            "q2com_comment",
        ])

    if args.use_gencode:
        fields.extend([
            "c2c_gencode",
            "c2c_code",
        ])

    emb_data: dict[str, torch.Tensor] = {}

    for field in fields:

        prefix = field.split("_")[0]
        suffix = "_".join(
            field.split("_")[1:]
        )

        model_name = getattr(
            args,
            f"{prefix}_model",
        )

        path = os.path.join(
            args.emb_root,
            model_name,
            args.lang,
            f"{args.lang}_{suffix}_emb.npy"
        )

        array = np.load(
            path,
            mmap_mode="r",
        )

        emb_data[field] = torch.from_numpy(array)

        logger.info(
            "%s: %s",
            field,
            tuple(emb_data[field].shape),
        )

    return emb_data


# ============================================================
# Alignment / split index construction
# ============================================================

def build_split_indices(
    text_data: dict[str, list[dict[str, str]]],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[
    str,
    dict[str, list[int]],
]:
    """
    Build original indices grouped by split and modality.

    Returned structure:

        indices = {
            "train": {
                "query": [original_query_idx, ...],
                "code": [original_code_idx, ...],
                "comment": [...],
                "gencode": [...],
            },
            "valid": {...},
            "test": {...},
        }

    Important:
    - Query order defines the final sample order.
    - Other modalities are aligned by ID.
    - Only enabled modalities are required.
    """

    logger.info(
        "===== Building split indices ====="
    )

    # --------------------------------------------------------
    # Build ID -> index maps
    # --------------------------------------------------------

    index_maps: dict[
        str,
        dict[str, int],
    ] = {}

    required_modalities = [
        "query",
        "code",
    ]

    if args.use_comment:
        required_modalities.append(
            "comment"
        )

    if args.use_gencode:
        required_modalities.append(
            "gencode"
        )

    for modality in required_modalities:

        if modality == "query":
            continue

        index_maps[modality] = {
            item["id"]: index
            for index, item in enumerate(
                text_data[modality]
            )
        }

    # --------------------------------------------------------
    # Initialize indices
    # --------------------------------------------------------

    split_indices: dict[
        str,
        dict[str, list[int]],
    ] = {
        split: {
            modality: []
            for modality in required_modalities
        }
        for split in SPLITS
    }

    skipped_missing = 0
    skipped_mismatch = 0

    # --------------------------------------------------------
    # Query defines aligned sample order
    # --------------------------------------------------------

    for query_idx, query_item in enumerate(
        text_data["query"]
    ):

        sample_id = query_item["id"]
        split = query_item["split"]

        if split not in split_indices:
            raise ValueError(
                f"Unknown split '{split}' "
                f"for sample '{sample_id}'"
            )

        modality_indices = {
            "query": query_idx,
        }

        valid_sample = True

        # Find aligned modality indices
        for modality in required_modalities:

            if modality == "query":
                continue

            modality_idx = index_maps[
                modality
            ].get(sample_id)

            if modality_idx is None:
                valid_sample = False
                break

            modality_item = text_data[
                modality
            ][modality_idx]

            # Require same split
            if "split" in modality_item and modality_item["split"] != split:
                skipped_mismatch += 1
                valid_sample = False
                break

            modality_indices[
                modality
            ] = modality_idx

        if not valid_sample:
            skipped_missing += 1
            continue

        # Append all original indices in exactly the same order
        for modality, modality_idx in (
            modality_indices.items()
        ):
            split_indices[split][
                modality
            ].append(modality_idx)

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    for split in SPLITS:
        logger.info(
            "%s: %d aligned samples",
            split,
            len(
                split_indices[split]["query"]
            ),
        )

    logger.info(
        "Skipped missing/incomplete samples: %d",
        skipped_missing,
    )

    logger.info(
        "Skipped split mismatches: %d",
        skipped_mismatch,
    )

    return split_indices


# ============================================================
# Reorder text data
# ============================================================

def split_text_data(
    raw_text_data: dict[
        str,
        list[dict[str, str]],
    ],
    split_indices: dict[
        str,
        dict[str, list[int]],
    ],
    args: argparse.Namespace,
) -> dict[
    str,
    list[dict[str, str]],
]:
    """
    Convert modality-based raw data into split-based aligned samples.

    Returned structure:

        text_data = {
            "train": [
                {
                    "id": ...,
                    "query": ...,
                    "code": ...,
                    "comment": ...,
                    "gencode": ...,
                },
                ...
            ],
            "valid": [...],
            "test": [...],
        }

    Therefore:
        text_data["train"][i]["query"]
        text_data["train"][i]["code"]

    are directly aligned.
    """

    result: dict[
        str,
        list[dict[str, str]],
    ] = {
        split: []
        for split in SPLITS
    }

    for split in SPLITS:

        query_indices = split_indices[
            split
        ]["query"]

        for local_idx, query_idx in enumerate(
            query_indices
        ):

            query_item = raw_text_data[
                "query"
            ][query_idx]

            sample = {
                "id": query_item["id"],
                "query": query_item["nl"],
                "code": "",
                "comment": "",
                "gencode": "",
            }

            # ------------------------------------------------
            # Code
            # ------------------------------------------------

            code_idx = split_indices[
                split
            ]["code"][local_idx]

            sample["code"] = raw_text_data[
                "code"
            ][code_idx]["code"]

            # ------------------------------------------------
            # Optional comment
            # ------------------------------------------------

            if args.use_comment:

                comment_idx = split_indices[
                    split
                ]["comment"][local_idx]

                sample["comment"] = raw_text_data[
                    "comment"
                ][comment_idx]["nl"]

            # ------------------------------------------------
            # Optional generated code
            # ------------------------------------------------

            if args.use_gencode:

                gencode_idx = split_indices[
                    split
                ]["gencode"][local_idx]

                sample["gencode"] = raw_text_data[
                    "gencode"
                ][gencode_idx]["code"]

            result[split].append(sample)

    return result


# ============================================================
# Reorder embedding data
# ============================================================

def split_embedding_data(
    raw_emb_data: dict[
        str,
        torch.Tensor,
    ],
    split_indices: dict[
        str,
        dict[str, list[int]],
    ],
    args,
) -> dict[
    str,
    dict[str, torch.Tensor],
]:
    """
    Split and reorder embeddings to match split_text_data().

    Returned structure:

        emb_data = {
            "train": {
                "q2c_query": Tensor,
                "q2c_code": Tensor,
                ...
            },
            "valid": {...},
            "test": {...},
        }

    The local index is always aligned:

        text_data["train"][i]
        emb_data["train"]["q2c_query"][i]
        emb_data["train"]["q2c_code"][i]
    """

    if not raw_emb_data:
        return {}

    result: dict[
        str,
        dict[str, torch.Tensor],
    ] = {
        split: {}
        for split in SPLITS
    }

    # Map embedding -> corresponding original text modality
    emb_to_modality = {
        "q2c_query": "query",
        "q2c_code": "code",
        "q2com_query": "query",
        "q2com_comment": "comment",
        "c2c_gencode": "gencode",
        "c2c_code": "code",
    }

    for split in SPLITS:

        for emb_name, embedding in (
            raw_emb_data.items()
        ):

            modality = emb_to_modality[
                emb_name
            ]

            # Skip optional embeddings when disabled
            if (
                modality not in split_indices[split]
            ):
                continue

            indices = split_indices[
                split
            ][modality]

            index_tensor = torch.tensor(
                indices,
                dtype=torch.long,
                device=embedding.device,
            )

            result[split][emb_name] = (
                embedding.index_select(
                    0,
                    index_tensor,
                )
            )

    return result


# ============================================================
# Data loading
# ============================================================

def load_data(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, torch.Tensor]]]:
    """
    Load and align text data and embeddings.

    Returns:

        text_data:
            dict[split, list[sample]]

        emb_data:
            dict[split, dict[embedding_name, Tensor]]

    Either structure may be empty depending on:
        args.data_root
        args.emb_root
    """

    logger.info(
        "===== Loading dataset ====="
    )

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    raw_text_data = load_text_data(
        args,
        logger,
    )

    raw_emb_data = load_embeddings(
        args,
        logger,
    )

    # --------------------------------------------------------
    # Build original indices by split/modality
    # --------------------------------------------------------

    split_indices = build_split_indices(
        raw_text_data,
        args,
        logger,
    )

    # --------------------------------------------------------
    # Reorder text and embeddings
    # --------------------------------------------------------

    text_data = split_text_data(
        raw_text_data,
        split_indices,
        args,
    )

    emb_data = split_embedding_data(
        raw_emb_data,
        split_indices,
        args,
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    for split in SPLITS:

        logger.info(
            "%s: %d final samples",
            split,
            len(text_data[split]),
        )

        if emb_data:

            for emb_name, embedding in (
                emb_data[split].items()
            ):

                if len(embedding) != len(
                    text_data[split]
                ):
                    raise RuntimeError(
                        f"Alignment error in {split}/"
                        f"{emb_name}: "
                        f"{len(embedding)} embeddings "
                        f"but "
                        f"{len(text_data[split])} "
                        f"text samples"
                    )

    return text_data, emb_data
