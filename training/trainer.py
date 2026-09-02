import argparse
import torch
import logging
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm

from data import load_data
from training.miners import static_negative_mining, dynamic_negative_mining
from training.handlers import BaseModelHandler, get_handler
from training.losses import LOSSES

def train_epoch(
    args: argparse.Namespace,
    logger: logging.Logger,
    handler: BaseModelHandler,
    text_data: list[dict[str, str]],
    emb_data: dict[str, torch.Tensor],
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Callable[..., torch.Tensor]
):
    handler.train()

    total_loss = 0.0
    total_qc = 0.0
    total_qcomment = 0.0
    total_gc = 0.0

    for query_idx, cand_idx in train_loader:
        query_idx: list[int] = query_idx
        cand_idx: list[list[int]] = cand_idx
        optimizer.zero_grad(set_to_none=True)

        positive_score, _ = handler.compute_scores(
            args,
            text_data=text_data,
            emb_data=emb_data,
            query_idx=query_idx,
            cand_idx=[[query_idx[i]] for i in range(len(query_idx))],
        )

        negative_score, negative_mask = handler.compute_scores(
            args,
            text_data=text_data,
            emb_data=emb_data,
            query_idx=query_idx,
            cand_idx=cand_idx,
        )

        loss = loss_fn(
            positive_score,
            negative_score,
            negative_mask,
            temperature=args.loss_temperature,
            sigma=args.loss_sigma,
            pos_weight=args.loss_pos_weight,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    num_batches = len(train_loader)

    return total_loss / num_batches

@torch.no_grad()
def compute_metrics(
    args: argparse.Namespace,
    logger: logging.Logger,
    handler: BaseModelHandler,
    text_data: list[dict[str, str]],
    emb_data: dict[str, torch.Tensor]
):
    """
    Evaluate query-code retrieval using the current embedding model.

    For each query, all candidate codes in the split are ranked by
    cosine similarity. Since the embeddings returned by ModelEmbedding
    are L2-normalized, the dot product is equivalent to cosine similarity.

    Returns:
        dict[str, float]: MRR, Top1, Top5, and Top10.
    """
    logger.info("Computing metrics...")

    scores = handler.compute_eval_scores(
        args,
        text_data=text_data,
        emb_data=emb_data,
    )

    # Because query_embeddings and code_embeddings are constructed
    # from the same ordered pair_indices list, position i contains
    # the positive query-code pair for each other.
    positive_positions = torch.arange(
        scores.size(0),
    )

    mrr = 0.0
    top1 = 0
    top5 = 0
    top10 = 0

    for query_position in range(scores.size(0)):

        ranking = torch.argsort(
            scores[query_position],
            descending=True,
        )

        positive_position = int(
            positive_positions[query_position]
        )

        rank: int = int(
            (ranking == positive_position)
            .nonzero(as_tuple=True)[0]
            .item()
            + 1
        )

        mrr += 1.0 / rank

        if rank <= 1:
            top1 += 1

        if rank <= 5:
            top5 += 1

        if rank <= 10:
            top10 += 1

    num_queries = scores.size(0)

    if num_queries == 0:
        return {
            "mrr": 0.0,
            "top1": 0.0,
            "top5": 0.0,
            "top10": 0.0,
        }

    return {
        "mrr": mrr / num_queries,
        "top1": top1 / num_queries,
        "top5": top5 / num_queries,
        "top10": top10 / num_queries,
    }

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[tuple[int, list[int]]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[int, list[int]]:
        return self.samples[idx]

def collate_fn(batch: list[tuple[int, list[int]]]) -> tuple[list[int], list[list[int]]]:
    query_indices = [item[0] for item in batch]
    candidate_indices = [item[1] for item in batch]
    return query_indices, candidate_indices


def fit(args: argparse.Namespace, logger: logging.Logger):
    text_data, emb_data = load_data(
        args,
        logger,
    )

    handler = get_handler(args)

    optimizer = handler.build_optimizer(args)

    loss_fn = LOSSES[args.loss_type]

    valid_metrics = compute_metrics(
        args=args,
        logger=logger,
        handler=handler,
        text_data=text_data["valid"],
        emb_data=emb_data["valid"],
    )

    best_mrr = valid_metrics["mrr"]

    logger.info(
        "Initial Valid "
        f"MRR {valid_metrics['mrr']:.4f} "
        f"Top1 {valid_metrics['top1']:.4f} "
        f"Top5 {valid_metrics['top5']:.4f} "
        f"Top10 {valid_metrics['top10']:.4f}"
    )

    samples = static_negative_mining(
        args,
        logger,
        emb_data["train"],
    )

    dataset = TrainDataset(samples)

    for epoch in range(args.epochs):
        if epoch >= args.static_mining_epochs:
            samples = dynamic_negative_mining(
                handler,
                args,
                text_data["train"],
                emb_data["train"],
            )

            dataset = TrainDataset(samples)

        dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        loss = train_epoch(
            args,
            logger,
            handler,
            text_data["train"],
            emb_data["train"],
            dataloader,
            optimizer,
            loss_fn,
        )

        valid_metrics = compute_metrics(
            args=args,
            logger=logger,
            handler=handler,
            text_data=text_data["valid"],
            emb_data=emb_data["valid"],
        )

        logger.info(
            f"Epoch {epoch:03d} "
            f"Loss {loss:.4f} "
            f"MRR {valid_metrics['mrr']:.4f} "
            f"Top1 {valid_metrics['top1']:.4f} "
            f"Top5 {valid_metrics['top5']:.4f} "
            f"Top10 {valid_metrics['top10']:.4f}"
        )

        if valid_metrics["mrr"] > best_mrr:
            best_mrr = valid_metrics["mrr"]
            handler.save_model(args)

    test_metrics = compute_metrics(
        args=args,
        logger=logger,
        handler=handler,
        text_data=text_data["test"],
        emb_data=emb_data["test"],
    )

    logger.info(
        "Test "
        f"MRR {test_metrics['mrr']:.4f} "
        f"Top1 {test_metrics['top1']:.4f} "
        f"Top5 {test_metrics['top5']:.4f} "
        f"Top10 {test_metrics['top10']:.4f}"
    )