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
        cand_idx: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def compute_eval_scores(
        self,
        args: argparse.Namespace,
        text_data: list[dict[str, str]],
        emb_data: dict[str, torch.Tensor],
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
        cand_idx: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scores for a batch of queries and candidates.

        Returns:
            scores: Tensor [B, N]
            mask: Tensor [B, N]
        """
        query_text = [text_data[i]["query"] for i in query_idx]

        gencode_text = None
        if args.use_gencode:
            gencode_text = [text_data[i]["gencode"] for i in query_idx]

        flatten_cand_idx = [i for sublist in cand_idx for i in sublist]

        code_text = [text_data[i]["code"] for i in flatten_cand_idx]

        comment_text = None
        if args.use_comment:
            comment_text = [text_data[i]["comment"] for i in flatten_cand_idx]

        # Compute embeddings
        query_emb = self.embedding_model.get_embedding(args.device, query_text, args.nl_length)

        gencode_emb = None
        if args.use_gencode:
            assert gencode_text is not None
            gencode_emb = self.embedding_model.get_embedding(args.device, gencode_text, args.code_length)

        flatten_code_emb = self.embedding_model.get_embedding(args.device, code_text, args.code_length)

        flatten_comment_emb = None
        if args.use_comment:
            assert comment_text is not None
            flatten_comment_emb = self.embedding_model.get_embedding(args.device, comment_text, args.nl_length)

        # Reshape code and comment embeddings to [B, N, D]
        B = len(query_idx)
        N = max(len(cand) for cand in cand_idx)
        D = flatten_code_emb.size(1)

        code_emb = torch.zeros(B, N, D, device=args.device)
        comment_emb = None
        mask = torch.zeros(B, N, device=args.device)

        for b in range(B):
            mask[b, :len(cand_idx[b])] = 1

        code_emb[mask.bool()] = flatten_code_emb

        if args.use_comment:
            assert flatten_comment_emb is not None
            comment_emb = torch.zeros(B, N, D, device=args.device)
            comment_emb[mask.bool()] = flatten_comment_emb

        # Compute scores
        q2c_scores = torch.einsum("bd,bnd->bn", query_emb, code_emb)

        if args.use_comment:
            assert comment_emb is not None
            q2com_scores = torch.einsum("bd,bnd->bn", query_emb, comment_emb)
        else:
            q2com_scores = torch.zeros_like(q2c_scores)

        if args.use_gencode:
            assert gencode_emb is not None
            c2c_scores = torch.einsum("bd,bnd->bn", gencode_emb, code_emb)
        else:
            c2c_scores = torch.zeros_like(q2c_scores)

        scores = (args.w1 * q2c_scores + args.w2 * q2com_scores + args.w3 * c2c_scores) / (args.w1 + args.w2 + args.w3)

        return scores, mask

    

    @override
    def compute_eval_scores(
        self,
        args: argparse.Namespace,
        text_data: list[dict[str, str]],
        emb_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute scores for all queries and candidates in the dataset.

        Returns:
            scores: Tensor [num_queries, num_candidates]
        """
        batch_size = args.eval_batch_size
        N = len(text_data)

        query_embeddings = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            query_text = [text_data[i]["query"] for i in range(start, end)]
            query_emb = self.embedding_model.get_embedding(args.device, query_text, args.nl_length)
            query_embeddings.append(query_emb)

        query_embeddings = torch.cat(query_embeddings, dim=0)

        code_embeddings = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            code_text = [text_data[i]["code"] for i in range(start, end)]
            code_emb = self.embedding_model.get_embedding(args.device, code_text, args.code_length)
            code_embeddings.append(code_emb)
        code_embeddings = torch.cat(code_embeddings, dim=0)

        comment_embeddings = None
        if args.use_comment:
            comment_embeddings = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                comment_text = [text_data[i]["comment"] for i in range(start, end)]
                comment_emb = self.embedding_model.get_embedding(args.device, comment_text, args.nl_length)
                comment_embeddings.append(comment_emb)
            comment_embeddings = torch.cat(comment_embeddings, dim=0)

        gencode_embeddings = None
        if args.use_gencode:
            gencode_embeddings = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                gencode_text = [text_data[i]["gencode"] for i in range(start, end)]
                gencode_emb = self.embedding_model.get_embedding(args.device, gencode_text, args.code_length)
                gencode_embeddings.append(gencode_emb)
            gencode_embeddings = torch.cat(gencode_embeddings, dim=0)

        # Compute scores
        q2c_scores = torch.einsum("qd,cd->qc", query_embeddings, code_embeddings)
        q2com_scores = torch.zeros_like(q2c_scores)
        c2c_scores = torch.zeros_like(q2c_scores)

        if args.use_comment:
            assert comment_embeddings is not None
            q2com_scores = torch.einsum("qd,cd->qc", query_embeddings, comment_embeddings)

        if args.use_gencode:
            assert gencode_embeddings is not None
            c2c_scores = torch.einsum("qd,cd->qc", gencode_embeddings, code_embeddings)

        scores = (args.w1 * q2c_scores + args.w2 * q2com_scores + args.w3 * c2c_scores) / (args.w1 + args.w2 + args.w3)
        return scores

    def train(self):
        self.embedding_model.train()

    def eval(self):
        self.embedding_model.eval()

def get_handler(args: argparse.Namespace) -> BaseModelHandler:
    if args.mode == "fine_tune":
        return ModelEmbeddingHandler(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")