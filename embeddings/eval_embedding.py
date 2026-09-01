"""
Evaluation pipeline for embedding-based code retrieval.

Supports:
- Query-code retrieval
- Weighted score fusion
- MRR / Top-k evaluation
- Weight traversal

Data and embeddings are loaded through data.load_data().
The returned test split is already locally aligned:

    text_data["test"][i]
    emb_data["test"]["q2c_query"][i]
    emb_data["test"]["q2c_code"][i]

all refer to the same aligned sample.
"""

import os
import csv
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from prettytable import PrettyTable

from data import load_data


# ============================================================
# Result table
# ============================================================

class ResultTable:
    """
    Utility wrapper for displaying evaluation results
    in a formatted table.
    """

    def __init__(
        self,
        title: list[str],
    ):
        self.table = PrettyTable(title)

    def add_row(
        self,
        row: list,
    ):
        self.table.add_row(row)

    def print_table(
        self,
        logger: logging.Logger,
    ):
        logger.info(self.table)


# ============================================================
# Top-k visualization
# ============================================================

def _truncate_text(
    text: str,
    max_len: int,
) -> str:
    """
    Normalize and truncate text for CSV display.
    """

    text = " ".join(str(text).split())

    if len(text) > max_len:
        return text[:max_len - 3] + "..."

    return text


def save_top10_results(
    output_path: str,
    test_data: list[dict[str, str]],
    best_indices: torch.Tensor,
    q_start: int,
    max_text_len: int = 200,
):
    """
    Save Top-10 retrieval results to CSV.

    Since test_data is already aligned and both query/code use
    local indices, candidate_index can directly index test_data.
    """

    assert best_indices.size(1) == 1, (
        "Top-10 visualization only supports one "
        "weight configuration at a time."
    )

    top_indices = best_indices[:, 0].detach().cpu()

    file_exists = os.path.exists(
        output_path
    )

    with open(
        output_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_index",
                "query_id",
                "query_text",
                "target_index",
                "rank",
                "candidate_index",
                "candidate_id",
                "candidate_text",
                "is_correct",
                "is_top1",
            ],
        )

        if not file_exists:
            writer.writeheader()

        for local_q_idx in range(
            top_indices.size(0)
        ):

            query_idx = (
                q_start + local_q_idx
            )

            query_item = test_data[
                query_idx
            ]

            query_text = _truncate_text(
                query_item["query"],
                max_text_len,
            )

            # Because the query/code arrays are aligned by
            # load_data(), the correct target has the same
            # local index as the query.
            target_idx = query_idx

            for rank in range(10):

                code_idx = int(top_indices[
                    local_q_idx,
                    rank,
                ].item())

                if code_idx < 0:
                    continue

                code_item = test_data[
                    code_idx
                ]

                code_text = _truncate_text(
                    code_item["code"],
                    max_text_len,
                )

                is_correct = (
                    code_idx == target_idx
                )

                writer.writerow({
                    "query_index": query_idx,
                    "query_id": query_item["id"],
                    "query_text": query_text,
                    "target_index": target_idx,
                    "rank": rank + 1,
                    "candidate_index": code_idx,
                    "candidate_id": code_item["id"],
                    "candidate_text": code_text,
                    "is_correct": is_correct,
                    "is_top1": rank == 0,
                })


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    lang: str,
    args: argparse.Namespace,
    logger: logging.Logger,
    test_data: list[dict[str, str]],
    test_emb: dict[str, torch.Tensor],
    w_list: list[list[float]] | None,
):
    """
    Evaluate the aligned test split.

    Required embeddings:

        query_code:
            q2c_query
            q2c_code

        fusion:
            q2c_query
            q2c_code
            q2com_query
            q2com_comment
            c2c_gencode
            c2c_code

    Since load_data() already split and reordered all modalities,
    index i directly refers to the same aligned sample across
    text and embedding tensors.
    """

    device = torch.device(args.device)

    # ========================================================
    # Test embeddings
    # ========================================================

    q_qc = test_emb["q2c_query"]
    c_qc = test_emb["q2c_code"]

    if args.eval_type == "fusion":

        required_fields = (
            "q2com_query",
            "q2com_comment",
            "c2c_gencode",
            "c2c_code",
        )

        missing = [
            field
            for field in required_fields
            if field not in test_emb
        ]

        if missing:
            raise ValueError(
                "Missing fusion embeddings: "
                + ", ".join(missing)
            )

        q_qm = test_emb[
            "q2com_query"
        ]

        c_qm = test_emb[
            "q2com_comment"
        ]

        g_cc = test_emb[
            "c2c_gencode"
        ]

        c_cc = test_emb[
            "c2c_code"
        ]

    else:

        q_qm = None
        c_qm = None
        g_cc = None
        c_cc = None

    # ========================================================
    # Alignment validation
    # ========================================================

    num_samples = len(test_data)

    if num_samples == 0:
        raise ValueError(
            f"No samples found in test split "
            f"for language '{lang}'."
        )

    for name, embedding in (
        test_emb.items()
    ):

        if embedding.size(0) != num_samples:
            raise RuntimeError(
                f"Test alignment error for {name}: "
                f"{embedding.size(0)} embeddings, "
                f"but {num_samples} text samples."
            )

    logger.info(
        "Evaluating %d aligned test samples",
        num_samples,
    )

    # ========================================================
    # Targets
    # ========================================================

    # Query i and code i are the positive pair after alignment.
    query_target_indices = torch.arange(
        num_samples,
        dtype=torch.long,
        device=device,
    )

    # ========================================================
    # Weight matrix
    # ========================================================

    if w_list is None:

        W = torch.tensor(
            [[1.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )

    else:

        W = torch.tensor(
            w_list,
            dtype=torch.float32,
            device=device,
        )

    num_weights = W.size(0)

    # ========================================================
    # Metrics
    # ========================================================

    mrr = torch.zeros(
        num_weights,
        device=device,
    )

    top1 = torch.zeros(
        num_weights,
        device=device,
    )

    top5 = torch.zeros(
        num_weights,
        device=device,
    )

    top10 = torch.zeros(
        num_weights,
        device=device,
    )

    query_bs = args.batch_size
    code_bs = args.batch_size

    num_queries = num_samples
    num_codes = num_samples

    # ========================================================
    # Top-10 output
    # ========================================================

    top10_output_path = None

    if args.print_top10:

        q2c_name = Path(
            args.q2c_model
        ).name

        if args.eval_type == "fusion":

            q2com_name = Path(
                args.q2com_model
            ).name

            c2c_name = Path(
                args.c2c_model
            ).name

            top10_output_path = (
                f"eval_"
                f"{q2c_name}_"
                f"{q2com_name}_"
                f"{c2c_name}_"
                f"{lang}.csv"
            )

        else:

            top10_output_path = (
                f"eval_"
                f"{q2c_name}_"
                f"{lang}.csv"
            )

        if os.path.exists(
            top10_output_path
        ):
            os.remove(
                top10_output_path
            )

    # ========================================================
    # Query batches
    # ========================================================

    for q_start in range(
        0,
        num_queries,
        query_bs,
    ):

        q_end = min(
            q_start + query_bs,
            num_queries,
        )

        targets = query_target_indices[
            q_start:q_end
        ]

        q_qc_batch = q_qc[
            q_start:q_end
        ].to(
            device,
            non_blocking=True,
        )

        if args.eval_type == "fusion":

            assert q_qm is not None
            assert g_cc is not None

            q_qm_batch = q_qm[
                q_start:q_end
            ].to(
                device,
                non_blocking=True,
            )

            g_cc_batch = g_cc[
                q_start:q_end
            ].to(
                device,
                non_blocking=True,
            )

        else:

            q_qm_batch = None
            g_cc_batch = None

        current_batch_size = (
            q_qc_batch.size(0)
        )

        # ----------------------------------------------------
        # Running top-10
        # ----------------------------------------------------

        best_scores = torch.full(
            (
                current_batch_size,
                num_weights,
                10,
            ),
            -float("inf"),
            device=device,
        )

        best_indices = torch.full(
            (
                current_batch_size,
                num_weights,
                10,
            ),
            -1,
            dtype=torch.long,
            device=device,
        )

        # ====================================================
        # Candidate batches
        # ====================================================

        for c_start in range(
            0,
            num_codes,
            code_bs,
        ):

            c_end = min(
                c_start + code_bs,
                num_codes,
            )

            c_qc_batch = c_qc[
                c_start:c_end
            ].to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Query-code similarity
            # ------------------------------------------------

            s_qc = (
                q_qc_batch
                @ c_qc_batch.T
            )

            if args.eval_type == "query_code":

                scores = s_qc.unsqueeze(1)

            else:

                assert q_qm is not None
                assert c_qm is not None
                assert g_cc is not None
                assert c_cc is not None
                assert q_qm_batch is not None
                assert g_cc_batch is not None

                c_qm_batch = c_qm[
                    c_start:c_end
                ].to(
                    device,
                    non_blocking=True,
                )

                c_cc_batch = c_cc[
                    c_start:c_end
                ].to(
                    device,
                    non_blocking=True,
                )

                # --------------------------------------------
                # Query-comment similarity
                # --------------------------------------------

                s_qm = (
                    q_qm_batch
                    @ c_qm_batch.T
                )

                # --------------------------------------------
                # Generated-code / code similarity
                # --------------------------------------------

                s_cc = (
                    g_cc_batch
                    @ c_cc_batch.T
                )

                # --------------------------------------------
                # Weighted fusion
                # --------------------------------------------

                scores = (
                    W[:, 0][None, :, None]
                    * s_qc[:, None, :]
                    +
                    W[:, 1][None, :, None]
                    * s_qm[:, None, :]
                    +
                    W[:, 2][None, :, None]
                    * s_cc[:, None, :]
                )

            # =================================================
            # Local top-k
            # =================================================

            k = min(
                10,
                scores.size(-1),
            )

            local_scores, local_indices = (
                torch.topk(
                    scores,
                    k=k,
                    dim=-1,
                    largest=True,
                    sorted=True,
                )
            )

            local_indices += c_start

            # ------------------------------------------------
            # Pad if the corpus is smaller than 10
            # ------------------------------------------------

            if k < 10:

                pad_scores = torch.full(
                    (
                        current_batch_size,
                        num_weights,
                        10 - k,
                    ),
                    -float("inf"),
                    device=device,
                )

                pad_indices = torch.full(
                    (
                        current_batch_size,
                        num_weights,
                        10 - k,
                    ),
                    -1,
                    dtype=torch.long,
                    device=device,
                )

                local_scores = torch.cat(
                    [
                        local_scores,
                        pad_scores,
                    ],
                    dim=-1,
                )

                local_indices = torch.cat(
                    [
                        local_indices,
                        pad_indices,
                    ],
                    dim=-1,
                )

            # =================================================
            # Merge global running top-k
            # =================================================

            all_scores = torch.cat(
                [
                    best_scores,
                    local_scores,
                ],
                dim=-1,
            )

            all_indices = torch.cat(
                [
                    best_indices,
                    local_indices,
                ],
                dim=-1,
            )

            best_scores, keep = torch.topk(
                all_scores,
                k=10,
                dim=-1,
                largest=True,
                sorted=True,
            )

            best_indices = torch.gather(
                all_indices,
                -1,
                keep,
            )

        # ====================================================
        # Save Top-10
        # ====================================================

        if (
            args.print_top10
            and num_weights == 1
            and top10_output_path is not None
        ):

            save_top10_results(
                output_path=top10_output_path,
                test_data=test_data,
                best_indices=best_indices,
                q_start=q_start,
                max_text_len=args.top10_text_len,
            )

        # ====================================================
        # Metrics
        # ====================================================

        # best_indices:
        #   [batch, num_weights, top10]
        #
        # targets:
        #   [batch]
        #
        # Since positive candidate index == query index:
        matched = (
            best_indices
            == targets[:, None, None]
        )

        has_match = matched.any(
            dim=-1
        )

        ranks = (
            matched.float()
            .argmax(dim=-1)
            + 1
        )

        reciprocal = torch.where(
            has_match,
            1.0 / ranks.float(),
            torch.zeros_like(
                ranks,
                dtype=torch.float32,
            ),
        )

        mrr += reciprocal.sum(
            dim=0
        )

        top1 += (
            (ranks == 1)
            & has_match
        ).sum(dim=0)

        top5 += (
            (ranks <= 5)
            & has_match
        ).sum(dim=0)

        top10 += (
            (ranks <= 10)
            & has_match
        ).sum(dim=0)

    # ========================================================
    # Logging
    # ========================================================

    if top10_output_path is not None:
        logger.info(
            "Saved Top-10 retrieval results to %s",
            top10_output_path,
        )

    # ========================================================
    # Format results
    # ========================================================

    weight_iter = (
        [[1.0, 0.0, 0.0]]
        if w_list is None
        else w_list
    )

    results = []

    for wi, weights in enumerate(
        weight_iter
    ):

        results.append([
            "-".join(
                f"{weight:.2f}"
                for weight in weights
            ),
            round(
                (
                    mrr[wi]
                    / num_queries
                ).item(),
                3,
            ),
            round(
                (
                    top1[wi]
                    / num_queries
                ).item(),
                3,
            ),
            round(
                (
                    top5[wi]
                    / num_queries
                ).item(),
                3,
            ),
            round(
                (
                    top10[wi]
                    / num_queries
                ).item(),
                3,
            ),
        ])

    return results


# ============================================================
# Weight traversal
# ============================================================

def build_weight_list(
    step_size: float,
):
    """
    Generate all weight combinations satisfying:

        w1 + w2 + w3 = 1
        w1, w2, w3 >= 0
    """

    weights = []

    for w1 in np.arange(
        0,
        1 + step_size,
        step_size,
    ):

        for w2 in np.arange(
            0,
            1 + step_size,
            step_size,
        ):

            if w1 + w2 > 1:
                continue

            w3 = 1 - w1 - w2

            if w3 < 0:
                continue

            weights.append([
                round(w1, 2),
                round(w2, 2),
                round(w3, 2),
            ])

    return weights


# ============================================================
# Evaluation modes
# ============================================================

def load_test_split(
    args: argparse.Namespace,
    logger: logging.Logger,
    lang: str,
):
    """
    Load and return only the already-aligned test split.

    load_data() uses args.lang to construct data/embedding
    paths, so it is set before loading.
    """

    args.lang = lang

    text_data, emb_data = load_data(
        args,
        logger,
    )

    test_data = text_data["test"]
    test_emb = emb_data["test"]

    if not test_data:
        raise ValueError(
            f"Empty test split for language '{lang}'."
        )

    logger.info(
        "Loaded test split: %d samples",
        len(test_data),
    )

    return test_data, test_emb


def run_eval(
    args: argparse.Namespace,
    logger: logging.Logger,
    lang: str,
):
    """
    Evaluate one language using a single weight configuration.
    """

    test_data, test_emb = load_test_split(
        args,
        logger,
        lang,
    )

    if args.eval_type == "fusion":

        w_list = [[
            args.w1,
            args.w2,
            args.w3,
        ]]

    else:

        w_list = None

    results = evaluate(
        lang=lang,
        args=args,
        logger=logger,
        test_data=test_data,
        test_emb=test_emb,
        w_list=w_list,
    )

    table = ResultTable([
        "weights",
        "MRR",
        "Top-1",
        "Top-5",
        "Top-10",
    ])

    for row in results:
        table.add_row(row)

    logger.info("\n")
    table.print_table(logger)


def run_traverse(
    args: argparse.Namespace,
    logger: logging.Logger,
    lang: str,
):
    """
    Traverse the weight simplex and save results to CSV.
    """

    test_data, test_emb = load_test_split(
        args,
        logger,
        lang,
    )

    w_list = build_weight_list(
        args.step_size,
    )

    results = evaluate(
        lang=lang,
        args=args,
        logger=logger,
        test_data=test_data,
        test_emb=test_emb,
        w_list=w_list,
    )

    q2c_name = Path(
        args.q2c_model
    ).name

    q2com_name = Path(
        args.q2com_model
    ).name

    c2c_name = Path(
        args.c2c_model
    ).name

    output_path = (
        f"traverse_"
        f"{q2c_name}_"
        f"{q2com_name}_"
        f"{c2c_name}_"
        f"{lang}.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Weights",
            "MRR",
            "Top-1",
            "Top-5",
            "Top-10",
        ])

        writer.writerows(results)

    logger.info(
        "Saved traversal results to %s",
        output_path,
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate embedding-based "
            "code retrieval."
        )
    )

    parser.add_argument(
        "--langs",
        type=str,
        default="cosqa,solidity,sql",
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
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--q2c_model",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--q2com_model",
        type=str,
    )

    parser.add_argument(
        "--c2c_model",
        type=str,
    )

    parser.add_argument(
        "--eval_type",
        type=str,
        choices=[
            "fusion",
            "query_code",
        ],
        default="fusion",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "eval",
            "traverse",
        ],
        default="eval",
    )

    parser.add_argument(
        "--w1",
        type=float,
    )

    parser.add_argument(
        "--w2",
        type=float,
    )

    parser.add_argument(
        "--w3",
        type=float,
    )

    parser.add_argument(
        "--step_size",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--print_top10",
        action="store_true",
    )

    parser.add_argument(
        "--top10_text_len",
        type=int,
        default=80,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
):
    """
    Validate arguments and configure modalities required
    by load_data().
    """

    if args.eval_type == "fusion":

        if (
            args.q2com_model is None
            or args.c2c_model is None
        ):
            raise ValueError(
                "Fusion evaluation requires "
                "--q2com_model and --c2c_model."
            )

        # Required by load_data().
        args.use_comment = True
        args.use_gencode = True

    else:

        # Avoid loading unnecessary modalities.
        args.use_comment = False
        args.use_gencode = False

    if (
        args.mode == "eval"
        and args.eval_type == "fusion"
    ):

        if (
            args.w1 is None
            or args.w2 is None
            or args.w3 is None
        ):
            raise ValueError(
                "Fusion evaluation requires "
                "--w1, --w2 and --w3."
            )


def main():

    args = parse_args()
    validate_args(args)

    logging.basicConfig(
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(
        __name__
    )

    logger.info(
        "args: %s",
        args,
    )

    for lang in args.langs.split(","):

        lang = lang.strip()

        if not lang:
            continue

        logger.info(
            "\n===== Evaluating %s =====",
            lang,
        )

        if args.mode == "eval":

            run_eval(
                args,
                logger,
                lang,
            )

        elif args.mode == "traverse":

            run_traverse(
                args,
                logger,
                lang,
            )


if __name__ == "__main__":
    main()