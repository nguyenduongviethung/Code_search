import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from typing import override

from models.embedding_models import get_embedding_model

class BaseModelHandler:
    def build_optimizer(self, args: argparse.Namespace) -> torch.optim.Optimizer:
        raise NotImplementedError()

    def compute_scores(
        self,
        args: argparse.Namespace,
        text_data: list[dict[str, str]],
        emb_data: dict[str, torch.Tensor],
        query_idx: list[int],
        cand_idx: list[list[int]],
        include_in_batch_negatives: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def compute_eval_scores(
        self,
        args: argparse.Namespace,
        text_data: list[dict[str, str]],
        emb_data: dict[str, torch.Tensor],
        batch_size: int
    ) -> torch.Tensor:
        raise NotImplementedError()

    def save_model(self, args: argparse.Namespace):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

class ModelEmbeddingHandler(BaseModelHandler):
    def __init__(self, args: argparse.Namespace):
        self.embedding_model = get_embedding_model(args.n_gpu, args.device, args.model_path)

        if args.freeze_layers <= 0:
            return
    
        for parameter in self.embedding_model.parameters():
            parameter.requires_grad = False
    
        model = (
            self.embedding_model.model.module
            if isinstance(self.embedding_model.model, torch.nn.DataParallel)
            else self.embedding_model.model
        )
    
        for layer in model.encoder.layer[args.freeze_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    @override
    def build_optimizer(self, args: argparse.Namespace) -> torch.optim.Optimizer:
        model = (
            self.embedding_model.model.module
            if isinstance(self.embedding_model.model, torch.nn.DataParallel)
            else self.embedding_model.model
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return optimizer

    @override
    def save_model(self, args: argparse.Namespace):
        model = (
            self.embedding_model.model.module
            if isinstance(self.embedding_model.model, torch.nn.DataParallel)
            else self.embedding_model.model
        )

        model.save_pretrained(args.output_dir)
        self.embedding_model.tokenizer.save_pretrained(args.output_dir)

    @override
    def compute_scores(
        self,
        args: argparse.Namespace,
        text_data: list[dict[str, str]],
        emb_data: dict[str, torch.Tensor],
        query_idx: list[int],
        cand_idx: list[list[int]],
        include_in_batch_negatives: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scores for a batch of queries and candidates.

        If include_in_batch_negatives is True and args.in_batch_negatives
        is enabled, positive candidates from other queries in the same
        batch are prepended as additional negatives.

        Returns:
            scores: Tensor [B, N + B - 1] when in-batch negatives are enabled,
                    otherwise [B, N]
            mask: Tensor [B, N + B - 1] when in-batch negatives are enabled,
                otherwise [B, N]
        """
        B = len(query_idx)

        query_text = [
            text_data[i]["query"]
            for i in query_idx
        ]

        gencode_text = None
        if args.use_gencode:
            gencode_text = [
                text_data[i]["gencode"]
                for i in query_idx
            ]

        # ============================================================
        # Compute query embeddings
        # ============================================================
        query_emb = self.embedding_model.get_embedding(
            args.device,
            query_text,
            args.nl_length,
        )

        gencode_emb = None
        if args.use_gencode:
            assert gencode_text is not None

            gencode_emb = self.embedding_model.get_embedding(
                args.device,
                gencode_text,
                args.code_length,
            )

        # ============================================================
        # Flatten hard-negative candidates
        # ============================================================
        flatten_cand_idx = [
            i
            for sublist in cand_idx
            for i in sublist
        ]

        # ============================================================
        # Explicit hard-negative scores
        #
        # If there are no hard negatives, create [B, 0] tensors
        # and DO NOT call the embedding model.
        # ============================================================
        if flatten_cand_idx:
            code_text = [
                text_data[i]["code"]
                for i in flatten_cand_idx
            ]

            comment_text = None
            if args.use_comment:
                comment_text = [
                    text_data[i]["comment"]
                    for i in flatten_cand_idx
                ]

            # --------------------------------------------------------
            # Compute candidate embeddings
            # --------------------------------------------------------
            flatten_code_emb = self.embedding_model.get_embedding(
                args.device,
                code_text,
                args.code_length,
            )

            flatten_comment_emb = None
            if args.use_comment:
                assert comment_text is not None

                flatten_comment_emb = self.embedding_model.get_embedding(
                    args.device,
                    comment_text,
                    args.nl_length,
                )

            # --------------------------------------------------------
            # Reshape candidate embeddings to [B, N, D]
            # --------------------------------------------------------
            N = max(
                len(cand)
                for cand in cand_idx
            )

            D = flatten_code_emb.size(1)

            code_emb = torch.zeros(
                B,
                N,
                D,
                device=args.device,
                dtype=flatten_code_emb.dtype,
            )

            comment_emb = None

            mask = torch.zeros(
                B,
                N,
                device=args.device,
                dtype=torch.bool,
            )

            offset = 0

            for b in range(B):
                num_candidates = len(cand_idx[b])

                if num_candidates == 0:
                    continue

                mask[b, :num_candidates] = True

                code_emb[
                    b,
                    :num_candidates
                ] = flatten_code_emb[
                    offset:offset + num_candidates
                ]

                if args.use_comment:
                    assert flatten_comment_emb is not None

                    if comment_emb is None:
                        comment_emb = torch.zeros(
                            B,
                            N,
                            D,
                            device=args.device,
                            dtype=flatten_comment_emb.dtype,
                        )

                    comment_emb[
                        b,
                        :num_candidates
                    ] = flatten_comment_emb[
                        offset:offset + num_candidates
                    ]

                offset += num_candidates

            # --------------------------------------------------------
            # query -> code
            # --------------------------------------------------------
            q2c_scores = torch.einsum(
                "bd,bnd->bn",
                query_emb,
                code_emb,
            )

            # --------------------------------------------------------
            # query -> comment
            # --------------------------------------------------------
            if args.use_comment:
                assert comment_emb is not None

                q2com_scores = torch.einsum(
                    "bd,bnd->bn",
                    query_emb,
                    comment_emb,
                )
            else:
                q2com_scores = torch.zeros_like(
                    q2c_scores
                )

            # --------------------------------------------------------
            # gencode -> code
            # --------------------------------------------------------
            if args.use_gencode:
                assert gencode_emb is not None

                c2c_scores = torch.einsum(
                    "bd,bnd->bn",
                    gencode_emb,
                    code_emb,
                )
            else:
                c2c_scores = torch.zeros_like(
                    q2c_scores
                )

            weight_sum = (
                args.w1
                + args.w2
                + args.w3
            )

            scores = (
                args.w1 * q2c_scores
                + args.w2 * q2com_scores
                + args.w3 * c2c_scores
            ) / weight_sum

        else:
            # --------------------------------------------------------
            # No hard negatives.
            #
            # IMPORTANT:
            # Do not call embedding_model for empty candidate lists.
            #
            # scores: [B, 0]
            # mask:   [B, 0]
            # --------------------------------------------------------
            scores = torch.empty(
                B,
                0,
                device=args.device,
                dtype=query_emb.dtype,
            )

            mask = torch.empty(
                B,
                0,
                device=args.device,
                dtype=torch.bool,
            )

        # ============================================================
        # In-batch negatives
        #
        # Only enabled when BOTH:
        #
        #   args.in_batch_negatives
        #   include_in_batch_negatives
        #
        # The positive candidate belonging to query j is used as a
        # negative for every query i != j.
        #
        # [B, B] -> [B, B - 1]
        # ============================================================
        if (
            args.in_batch_negatives
            and include_in_batch_negatives
            and B > 1
        ):
            # --------------------------------------------------------
            # Positive code embeddings
            # --------------------------------------------------------
            positive_code_text = [
                text_data[i]["code"]
                for i in query_idx
            ]

            positive_code_emb = self.embedding_model.get_embedding(
                args.device,
                positive_code_text,
                args.code_length,
            )

            # query -> positive code
            in_batch_q2c_scores = torch.einsum(
                "bd,cd->bc",
                query_emb,
                positive_code_emb,
            )

            # --------------------------------------------------------
            # query -> positive comment
            # --------------------------------------------------------
            if args.use_comment:
                positive_comment_text = [
                    text_data[i]["comment"]
                    for i in query_idx
                ]

                positive_comment_emb = self.embedding_model.get_embedding(
                    args.device,
                    positive_comment_text,
                    args.nl_length,
                )

                in_batch_q2com_scores = torch.einsum(
                    "bd,cd->bc",
                    query_emb,
                    positive_comment_emb,
                )
            else:
                in_batch_q2com_scores = torch.zeros_like(
                    in_batch_q2c_scores
                )

            # --------------------------------------------------------
            # gencode -> positive code
            # --------------------------------------------------------
            if args.use_gencode:
                assert gencode_emb is not None

                in_batch_c2c_scores = torch.einsum(
                    "bd,cd->bc",
                    gencode_emb,
                    positive_code_emb,
                )
            else:
                in_batch_c2c_scores = torch.zeros_like(
                    in_batch_q2c_scores
                )

            # --------------------------------------------------------
            # Combine
            # --------------------------------------------------------
            weight_sum = (
                args.w1
                + args.w2
                + args.w3
            )

            in_batch_scores = (
                args.w1 * in_batch_q2c_scores
                + args.w2 * in_batch_q2com_scores
                + args.w3 * in_batch_c2c_scores
            ) / weight_sum

            # --------------------------------------------------------
            # Remove diagonal:
            #
            # [B, B] -> [B, B - 1]
            # --------------------------------------------------------
            non_diagonal = ~torch.eye(
                B,
                device=args.device,
                dtype=torch.bool,
            )

            in_batch_scores = in_batch_scores[
                non_diagonal
            ].view(B, B - 1)

            # Every remaining candidate is a negative.
            in_batch_mask = torch.ones(
                B,
                B - 1,
                device=args.device,
                dtype=torch.bool,
            )

            # --------------------------------------------------------
            # Prepend in-batch negatives
            # --------------------------------------------------------
            scores = torch.cat(
                [
                    in_batch_scores,
                    scores,
                ],
                dim=1,
            )

            mask = torch.cat(
                [
                    in_batch_mask,
                    mask,
                ],
                dim=1,
            )

        return scores, mask

    @override
    def compute_eval_scores(
        self,
        args: argparse.Namespace,
        text_data: list[dict[str, str]],
        emb_data: dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compute scores for all queries and candidates in the dataset.

        The embeddings are computed once, then the score matrix is computed
        batch-by-batch over queries to avoid materializing the full [N, N]
        matrix on GPU.

        Returns:
            scores: Tensor [num_queries, num_candidates]
        """
        N = len(text_data)
        device = args.device

        # ============================================================
        # Compute all query embeddings
        # ============================================================
        query_embeddings = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            query_text = [
                text_data[i]["query"]
                for i in range(start, end)
            ]

            query_emb = self.embedding_model.get_embedding(
                device,
                query_text,
                args.nl_length,
            )

            query_embeddings.append(query_emb)

        query_embeddings = torch.cat(query_embeddings, dim=0)

        # ============================================================
        # Compute all code embeddings
        # ============================================================
        code_embeddings = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            code_text = [
                text_data[i]["code"]
                for i in range(start, end)
            ]

            code_emb = self.embedding_model.get_embedding(
                device,
                code_text,
                args.code_length,
            )

            code_embeddings.append(code_emb)

        code_embeddings = torch.cat(code_embeddings, dim=0)

        # ============================================================
        # Compute all comment embeddings
        # ============================================================
        comment_embeddings = None

        if args.use_comment:
            comment_embeddings = []

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)

                comment_text = [
                    text_data[i]["comment"]
                    for i in range(start, end)
                ]

                comment_emb = self.embedding_model.get_embedding(
                    device,
                    comment_text,
                    args.nl_length,
                )

                comment_embeddings.append(comment_emb)

            comment_embeddings = torch.cat(comment_embeddings, dim=0)

        # ============================================================
        # Compute all gencode embeddings
        # ============================================================
        gencode_embeddings = None

        if args.use_gencode:
            gencode_embeddings = []

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)

                gencode_text = [
                    text_data[i]["gencode"]
                    for i in range(start, end)
                ]

                gencode_emb = self.embedding_model.get_embedding(
                    device,
                    gencode_text,
                    args.code_length,
                )

                gencode_embeddings.append(gencode_emb)

            gencode_embeddings = torch.cat(gencode_embeddings, dim=0)

        # ============================================================
        # Compute score matrix batch-by-batch
        #
        # Instead of:
        #
        #   [N, D] @ [D, N] -> [N, N]
        #
        # we compute:
        #
        #   [B, D] @ [D, N] -> [B, N]
        #
        # ============================================================
        score_batches = []

        weight_sum = args.w1 + args.w2 + args.w3

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            q_batch = query_embeddings[start:end]

            # --------------------------------------------------------
            # query -> code
            # --------------------------------------------------------
            q2c_scores = torch.einsum(
                "bd,cd->bc",
                q_batch,
                code_embeddings,
            )

            # --------------------------------------------------------
            # query -> comment
            # --------------------------------------------------------
            if args.use_comment:
                assert comment_embeddings is not None

                q2com_scores = torch.einsum(
                    "bd,cd->bc",
                    q_batch,
                    comment_embeddings,
                )
            else:
                q2com_scores = torch.zeros_like(q2c_scores)

            # --------------------------------------------------------
            # gencode -> code
            # --------------------------------------------------------
            if args.use_gencode:
                assert gencode_embeddings is not None

                gen_batch = gencode_embeddings[start:end]

                c2c_scores = torch.einsum(
                    "bd,cd->bc",
                    gen_batch,
                    code_embeddings,
                )
            else:
                c2c_scores = torch.zeros_like(q2c_scores)

            # --------------------------------------------------------
            # Combine scores immediately
            # --------------------------------------------------------
            scores = (
                args.w1 * q2c_scores
                + args.w2 * q2com_scores
                + args.w3 * c2c_scores
            ) / weight_sum

            # Move to CPU immediately so GPU memory stays bounded.
            score_batches.append(scores.cpu())

        # ============================================================
        # Final [N, N] score matrix on CPU
        # ============================================================
        return torch.cat(score_batches, dim=0)


    def train(self):
        self.embedding_model.train()

    def eval(self):
        self.embedding_model.eval()

def get_handler(args: argparse.Namespace) -> BaseModelHandler:
    if args.mode == "fine_tune":
        return ModelEmbeddingHandler(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")