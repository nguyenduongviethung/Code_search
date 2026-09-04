import argparse
import logging
import torch
import matplotlib.pyplot as plt
import os

from training.handlers import BaseModelHandler

@torch.no_grad()
def static_negative_mining(
    args: argparse.Namespace,
    logger: logging.Logger,
    emb_data: dict[str, torch.Tensor],
):
    """
    Mine hard negatives using similarity in embedding space.

    Strategy:
        For each query:
            - compute similarity against ALL code candidates
            - pick top-k highest (excluding positive)

    Output:
        List of samples:
            (query_idx, pos_code_idx, [neg_code_idx...])
    """

    logger.info("===== Static Hard Negative Mining =====")

    device = args.device
    batch_size = args.hn_batch_size

    # Move ALL candidate embeddings to GPU once
    q2c_code = emb_data["q2c_code"].to(device)
    q2com_comment = emb_data["q2com_comment"].to(device) if args.use_comment else None
    c2c_code = emb_data["c2c_code"].to(device) if args.use_gencode else None

    q2c_query = emb_data["q2c_query"]
    q2com_query = emb_data["q2com_query"] if args.use_comment else None
    c2c_gencode = emb_data["c2c_gencode"] if args.use_gencode else None

    samples: list[tuple[int, list[int]]] = []
    N = q2c_query.size(0)

    log_interval = max(1, N // args.num_logs) if args.num_logs > 0 else N

    for i in range(0, N, batch_size):
        B = min(batch_size, N - i)

        # =========================
        # Slice batch 
        # =========================
        q_batch = q2c_query[i:i+B].to(args.device)

        qcom_batch = None
        if args.use_comment:
            assert q2com_query is not None
            qcom_batch = q2com_query[i:i+B].to(args.device)

        gencode_batch = None
        if args.use_gencode:
            assert c2c_gencode is not None
            gencode_batch = c2c_gencode[i:i+B].to(args.device)

        # =========================
        # similarity (matrix multiply)
        # (B, C)
        # =========================
        s1 = q_batch @ q2c_code.T

        s2 = None
        if args.use_comment:
            assert qcom_batch is not None and q2com_comment is not None
            s2 = qcom_batch @ q2com_comment.T

        s3 = None
        if args.use_gencode:
            assert gencode_batch is not None and c2c_code is not None
            s3 = gencode_batch @ c2c_code.T

        # =========================
        # mask positive
        # =========================
        row = torch.arange(B, device=device)
        col = torch.arange(i, i + B, device=device)

        positive_scores1 = s1[row, col].clone()
        s1[row, col] = -torch.inf

        positive_scores2 = None
        if args.use_comment:
            assert s2 is not None
            positive_scores2 = s2[row, col].clone()
            s2[row, col] = -torch.inf

        positive_scores3 = None
        if args.use_gencode:
            assert s3 is not None
            positive_scores3 = s3[row, col].clone()
            s3[row, col] = -torch.inf

        idx1: list[list[int]] | None = None
        idx2: list[list[int]] | None = None
        idx3: list[list[int]] | None = None
        if args.miner_mode == "topk":
            # =========================
            # top-k negatives
            # =========================
            idx1 = torch.topk(s1, args.static_topk, dim=1).indices.tolist()

            if args.use_comment:
                assert s2 is not None
                idx2 = torch.topk(s2, args.static_topk, dim=1).indices.tolist()

            if args.use_gencode:
                assert s3 is not None
                idx3 = torch.topk(s3, args.static_topk, dim=1).indices.tolist()

        elif args.miner_mode == "threshold":
            # =========================
            # threshold negatives
            # =========================
            threshold = args.miner_threshold

            idx1 = [
                torch.where(s1[b] > positive_scores1[b] - threshold)[0].tolist()
                for b in range(B)
            ]

            if args.use_comment:
                assert s2 is not None and positive_scores2 is not None
                idx2 = [
                    torch.where(s2[b] > positive_scores2[b] - threshold)[0].tolist()
                    for b in range(B)
                ]

            if args.use_gencode:
                assert s3 is not None and positive_scores3 is not None
                idx3 = [
                    torch.where(s3[b] > positive_scores3[b] - threshold)[0].tolist()
                    for b in range(B)
                ]
        else:
            raise ValueError(f"Unknown miner_mode: {args.miner_mode}")

        # =========================
        # build samples
        # =========================
        for b in range(B):
            positive = i + b

            # merge 3 sources of negatives
            negatives = idx1[b]

            if args.use_comment:
                assert idx2 is not None
                negatives += idx2[b]

            if args.use_gencode:
                assert idx3 is not None
                negatives += idx3[b]

            negatives = list(set(negatives))  # remove duplicates

            samples.append((positive, negatives))

        # =========================
        # logging
        # =========================
        if logger and ((i + B) % log_interval == 0 or (i + B) >= N):
            logger.info(f"[HN] {i+B}/{N} ({(i+B)/N:.1%})")

    logger.info(f"Built {len(samples)} samples")
    return samples

@torch.no_grad()
def dynamic_negative_mining(
    handler: BaseModelHandler,
    args: argparse.Namespace,
    logger: logging.Logger,
    text_data: list[dict[str, str]],
    emb_data: dict[str, torch.Tensor],
):
    """
    Dynamic hard negative mining:
    - Pre-filter by 3 embedding scores
    - Rerank by model
    """

    handler.eval()

    device = args.device
    log_interval = max(1, len(text_data) // args.num_logs) if args.num_logs > 0 else len(text_data)

    # preload embeddings (same as static mining)
    q2c_code = emb_data["q2c_code"].to(device)
    q2com_comment = emb_data["q2com_comment"].to(device) if args.use_comment else None
    c2c_code = emb_data["c2c_code"].to(device) if args.use_gencode else None

    q2c_query = emb_data["q2c_query"]
    q2com_query = emb_data["q2com_query"] if args.use_comment else None
    c2c_gencode = emb_data["c2c_gencode"] if args.use_gencode else None

    samples: list[tuple[int, list[int]]] = []
    N = q2c_query.size(0)
    K0 = args.dynamic_negatives_per_source

    for i in range(0, N, args.hn_batch_size):
        B = min(args.hn_batch_size, N - i)

        q_batch = q2c_query[i:i+B].to(device)

        qcom_batch = None
        if args.use_comment:
            assert q2com_query is not None
            qcom_batch = q2com_query[i:i+B].to(device)

        gen_batch = None
        if args.use_gencode:
            assert c2c_gencode is not None
            gen_batch = c2c_gencode[i:i+B].to(device)

        # =========================
        # compute 3 similarity scores
        # =========================
        s1 = q_batch @ q2c_code.T

        s2 = None
        if args.use_comment:
            assert qcom_batch is not None and q2com_comment is not None
            s2 = qcom_batch @ q2com_comment.T

        s3 = None
        if args.use_gencode:
            assert gen_batch is not None and c2c_code is not None
            s3 = gen_batch @ c2c_code.T

        # =========================
        # mask positive
        # =========================
        row = torch.arange(B, device=device)
        col = torch.arange(i, i+B, device=device)
        
        s1[row, col] = -torch.inf

        if args.use_comment:
            assert s2 is not None
            s2[row, col] = -torch.inf

        if args.use_gencode:
            assert s3 is not None
            s3[row, col] = -torch.inf
        
        # =========================
        # top-k candidates from each
        # =========================
        idx1 = torch.topk(s1, K0, dim=1).indices

        idx2 = None
        if args.use_comment:
            assert s2 is not None
            idx2 = torch.topk(s2, K0, dim=1).indices

        idx3 = None
        if args.use_gencode:
            assert s3 is not None
            idx3 = torch.topk(s3, K0, dim=1).indices

        positive: list[int] = [i + b for b in range(B)]
        candidates: list[list[int]] = []

        for b in range(B):
            negs = idx1[b].tolist()

            if args.use_comment:
                assert idx2 is not None
                negs += idx2[b].tolist()

            if args.use_gencode:
                assert idx3 is not None
                negs += idx3[b].tolist()

            candidates.append(list(set(negs)))  # remove duplicates

        pos_score, _ = handler.compute_scores(
            args,
            text_data,
            emb_data,
            positive,
            [[positive[b]] for b in range(B)]
        )

        neg_scores, neg_mask = handler.compute_scores(
            args,
            text_data,
            emb_data,
            positive,
            candidates
        )

        for b in range(B):
            # filter out padding candidates
            valid_scores = neg_scores[b][neg_mask[b].bool()]

            indices: list[int]

            if args.miner_mode == "topk":
                indices = torch.topk(valid_scores, args.dynamic_topk, largest=True).indices.tolist()

            elif args.miner_mode == "threshold":
                indices = torch.where(valid_scores > pos_score[b] - args.miner_threshold)[0].tolist()

            else:
                raise ValueError(f"Unknown miner_mode: {args.miner_mode}")

            negatves = [candidates[b][n] for n in indices]

            samples.append((positive[b], negatves))

        if logger and ((i + B) % log_interval == 0 or (i + B) >= N):
            logger.info(f"[HN] {i+B}/{N} ({(i+B)/N:.1%})")

    return samples