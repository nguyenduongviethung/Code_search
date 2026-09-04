import os
import argparse
import logging

import torch

from training.trainer import fit
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an embedding model for code search."
    )

    # =========================
    # Dataset
    # =========================
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
        "--use_comment",
        action="store_true"
    )
    parser.add_argument(
        "--use_gencode",
        action="store_true"
    )

    # =========================
    # Model
    # =========================
    parser.add_argument(
        "--mode",
        type=str,
        default="fine_tune",
        choices=["fine_tune"],
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/unixcoder-base",
    )

    parser.add_argument(
        "--q2c_model",
        type=str,
        default="microsoft/unixcoder-base",
    )

    parser.add_argument(
        "--q2com_model",
        type=str,
        default="microsoft/unixcoder-base",
    )

    parser.add_argument(
        "--c2c_model",
        type=str,
        default="microsoft/unixcoder-base",
    )

    parser.add_argument(
        "--freeze_layers",
        type=int,
        default=10,
        help="Freeze the first N encoder layers.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoint",
    )

    # =========================
    # Sequence length
    # =========================
    
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

    # =========================
    # Training
    # =========================
    
    parser.add_argument(
        "--hn_batch_size",
        type=int,
        default=64,
        help="Batch size for hard negative mining.",
    )
    
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--static_mining_epochs",
        type=int,
        default=2,
        help="Number of epochs to use static negative mining.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    # =========================
    # Loss
    # =========================
    parser.add_argument(
        "--loss_type",
        type=str,
        default="infonce",
        choices=[
            "infonce",
            "bce_cosine",
            "queue_infonce"
        ],
    )
    
    parser.add_argument(
        "--loss_temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--loss_sigma",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--loss_pos_weight",
        type=float,
        default=None,
        help="Weight applied to positive pairs in BCE cosine loss.",
    )

    parser.add_argument(
        "--w1",
        type=float,
        default=1.0,
        help="Weight for query-code score.",
    )

    parser.add_argument(
        "--w2",
        type=float,
        default=1.0,
        help="Weight for query-comment score.",
    )

    parser.add_argument(
        "--w3",
        type=float,
        default=1.0,
        help="Weight for code-code score.",
    )

    # =========================
    # Hard Negative Mining
    # =========================
    parser.add_argument(
        "--miner_mode",
        type=str,
        default="topk",
        choices=["topk", "threshold"],
    )

    parser.add_argument(
        "--static_topk",
        type=int,
        default=4
    )

    parser.add_argument(
        "--dynamic_negatives_per_source",
        type=int,
        default=4,
        help="Number of dynamic negatives sampled from each source.",
    )

    parser.add_argument(
        "--dynamic_topk",
        type=int,
        default=4
    )

    parser.add_argument(
        "--miner_threshold",
        type=float,
        default=0.05,
        help="Threshold for selecting hard negatives.",
    )

    # =========================
    # Runtime
    # =========================
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num_logs",
        type=int,
        default=10,
        help="Number of logs to print during hard negative mining.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(__name__)

    args.device = torch.device(args.device)
    args.n_gpu = torch.cuda.device_count()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    for lang in map(str.strip, args.langs.split(",")):
        if not lang:
            continue

        logger.info("=" * 80)
        logger.info("Training language: %s", lang)
        logger.info("=" * 80)

        args.lang = lang

        fit(
            args,
            logger,
        )


if __name__ == "__main__":
    main()