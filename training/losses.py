import torch
import torch.nn.functional as F

from typing import Callable

# =========================
# LOSS FUNCTIONS
# =========================
def infonce_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, mask: torch.Tensor, **kwargs):
    """
    pos_score: Tensor [B]
    neg_score: Tensor [B, N]
    mask: ndarray or Tensor [B, N]
          1 = valid negative
          0 = padding negative
    """

    mask = mask.to(device=pos_score.device, dtype=torch.bool)

    # [B] -> [B, 1]
    pos_score = pos_score.unsqueeze(1)

    # [B, 1+N]
    logits = torch.cat([pos_score, neg_score], dim=1)
    logits = logits / kwargs.get("temperature", 1.0)

    # positive sample
    full_mask = torch.cat([
        torch.ones((mask.shape[0], 1),
                   dtype=torch.bool,
                   device=mask.device),
        mask
    ], dim=1)

    # padding negatives
    logits = logits.masked_fill(~full_mask, -torch.inf)

    # positive is at index 0
    labels = torch.zeros(
        logits.size(0),
        dtype=torch.long,
        device=logits.device
    )

    return F.cross_entropy(logits, labels)


def ranknet_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, mask: torch.Tensor, **kwargs):
    """
    Pairwise ranking loss (RankNet)

    Encourages:
        pos_score > neg_score
    """

    mask = mask.to(device=pos_score.device, dtype=torch.float32)

    pos_score = pos_score.unsqueeze(1)
    diff = pos_score - neg_score

    diff = diff * kwargs.get("sigma", 1.0)

    loss = torch.nn.functional.softplus(-diff)

    loss = loss * mask
    return loss.sum() / mask.sum()

def bce_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, mask: torch.Tensor, **kwargs):
    """
    Binary Cross Entropy loss with logits.

    Encourages:
        pos_score -> 1
        neg_score -> 0

    pos_score: Tensor [B]
    neg_score: Tensor [B, N]
    mask: Tensor [B, N]
          1 = valid negative
          0 = padding negative

    sigma: scale factor applied to scores before BCEWithLogitsLoss.
    """

    mask = mask.to(device=pos_score.device, dtype=torch.float32)

    sigma = kwargs.get("sigma", 1.0)

    # Scale scores similarly to RankNet
    pos_score = pos_score * sigma
    neg_score = neg_score * sigma

    # Positive: [B] -> [B, 1]
    pos_score = pos_score.unsqueeze(1)

    # Concatenate positive and negative logits
    # [B, 1 + N]
    logits = torch.cat([pos_score, neg_score], dim=1)

    # Labels:
    # positive = 1
    # negative = 0
    labels = torch.cat([
        torch.ones(
            (logits.size(0), 1),
            dtype=logits.dtype,
            device=logits.device
        ),
        torch.zeros_like(neg_score)
    ], dim=1)

    pos_weight = kwargs.get("pos_weight", None)

    if pos_weight is None:
        # BCEWithLogitsLoss does sigmoid internally
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none"
        )

    else:
        weight = torch.ones_like(labels)
        weight[labels == 1] = pos_weight

        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            weight=weight,
            reduction="none"
        )

    # Mask padding negatives.
    # Positive is always valid.
    full_mask = torch.cat([
        torch.ones(
            (mask.shape[0], 1),
            dtype=mask.dtype,
            device=mask.device
        ),
        mask
    ], dim=1)

    loss = loss * full_mask

    return loss.sum() / full_mask.sum()


LOSSES: dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "infonce": infonce_loss,
    "ranknet": ranknet_loss,
    "bce": bce_loss,
}