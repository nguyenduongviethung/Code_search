"""
get embeddings of different models
"""

from transformers import AutoTokenizer, AutoModel
import torch
from typing import override

class ModelEmbedding(torch.nn.Module):
    """
    base class
    """
    def __init__(self, n_gpu: int, device: torch.device, model_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)

    def build_inputs(self, device: torch.device, texts: list[str] | tuple[list[str], list[str]], max_length: int = 128):
        if isinstance(texts, list):
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

        else:
            queries, cands = texts
            encoded = self.tokenizer(
                queries,
                cands,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

        return encoded

    def get_embedding(self, device: torch.device, texts: list[str] | tuple[list[str], list[str]], max_length: int = 128) -> torch.Tensor:
        raise NotImplementedError("get_embedding method not implemented in base class")

    def get_token_embedding(self, device: torch.device, texts: list[str] | tuple[list[str], list[str]], max_length: int = 128):
        encoded = self.build_inputs(device, texts, max_length)

        outputs = self.model(**encoded)

        # last hidden state
        token_embeddings = outputs[0]

        # attention mask
        attention_mask = encoded["attention_mask"]

        return token_embeddings, attention_mask

class MeanPoolingEmbedding(ModelEmbedding):
    def __init__(self, n_gpu: int, device: torch.device, model_path: str):
        super().__init__(n_gpu, device, model_path)

    @override
    def get_embedding(self, device: torch.device, texts: list[str] | tuple[list[str], list[str]], max_length: int = 128):
        token_embeddings, attention_mask = self.get_token_embedding(device, texts, max_length)

        mask = attention_mask.unsqueeze(-1).to(
            dtype=token_embeddings.dtype
        )

        embeddings = torch.sum(
            token_embeddings * mask,
            dim=1
        ) / torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )


        # cosine embedding
        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1
        )

        return embeddings

class CLS_PoolingEmbedding(ModelEmbedding):
    def __init__(self, n_gpu: int, device: torch.device, model_path: str):
        super().__init__(n_gpu, device, model_path)

    @override
    def get_embedding(self, device: torch.device, texts: list[str] | tuple[list[str], list[str]], max_length: int = 128):
        token_embeddings, attention_mask = self.get_token_embedding(device, texts, max_length)

        # pooling from CLS
        sentence_embeddings = token_embeddings[:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

class LastTokenPoolingEmbedding(ModelEmbedding):
    def __init__(self, n_gpu: int, device: torch.device, model_path: str):
        super().__init__(n_gpu, device, model_path)

    @override
    def get_embedding(self, device: torch.device, texts: list[str] | tuple[list[str], list[str]], max_length: int = 128):
        token_embeddings, attention_mask = self.get_token_embedding(device, texts, max_length)

        # last token pooling
        sequence_lengths = attention_mask.sum(dim=1) - 1
        sentence_embeddings = token_embeddings[
            torch.arange(
                token_embeddings.shape[0],
                device=token_embeddings.device,
            ),
            sequence_lengths,
        ]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

class UniXCoderEmbedding(MeanPoolingEmbedding):
    def __init__(self, n_gpu: int, device: torch.device, model_path: str):
        super().__init__(n_gpu, device, model_path)

    @override
    def build_inputs(self, device: torch.device, texts: list[str] | tuple[list[str], list[str]], max_length: int = 128): 
        if max_length < 4:
            raise ValueError("max_length must be >= 4")

        # UniXcoder encoder-only format:
        #
        # <s> <encoder-only> </s> text </s> <pad> ...
        #
        # 4 special tokens:
        # <s>
        # <encoder-only>
        # </s>
        # </s>
        #
        text_max_length = max_length - 4

        encoder_only_id = self.tokenizer.convert_tokens_to_ids("<encoder-only>")

        if encoder_only_id == self.tokenizer.unk_token_id:
            raise ValueError("<encoder-only> is not registered in the tokenizer")

        # Tokenize the whole batch.
        #
        # We disable tokenizer special tokens because UniXcoder
        # requires a custom special-token layout.
        if isinstance(texts, list):
            encoded = self.tokenizer(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=text_max_length,
                padding="max_length",
                return_tensors="pt"
            )
        else:
            queries, cands = texts
            encoded = self.tokenizer(
                queries,
                cands,
                add_special_tokens=False,
                truncation=True,
                max_length=text_max_length,
                padding="max_length",
                return_tensors="pt"
            )

        text_ids = encoded["input_ids"]

        batch_size = text_ids.size(0)

        # Start everything as padding.
        input_ids = torch.full(
            (batch_size, max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )

        # =========================
        # Prefix:
        #
        # <s> <encoder-only> </s>
        # =========================
        input_ids[:, 0] = self.tokenizer.cls_token_id
        input_ids[:, 1] = encoder_only_id
        input_ids[:, 2] = self.tokenizer.sep_token_id

        # =========================
        # Text tokens
        # =========================
        
        input_ids[:, 3:3 + text_max_length] = text_ids

        # =========================
        # Final </s>
        # =========================
        
        text_mask = text_ids.ne(self.tokenizer.pad_token_id)

        text_lengths = text_mask.sum(dim=1)
        batch_indices = torch.arange(batch_size)
        input_ids[batch_indices, 3 + text_lengths] = self.tokenizer.sep_token_id

        # =========================
        # Attention mask
        #
        # 1 = actual token
        # 0 = padding
        # =========================
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device)
        }


def get_embedding_model(n_gpu: int, device: torch.device, model_path: str) -> ModelEmbedding:
    model_name = model_path.lower()

    if any(keyword in model_name for keyword in ["unixcoder", "unicor", "cocosoda"]):
        return UniXCoderEmbedding(n_gpu, device, model_path)
    if "bge" in model_name:
        return CLS_PoolingEmbedding(n_gpu, device, model_path)
    if "qwen3" in model_name:
        return LastTokenPoolingEmbedding(n_gpu, device, model_path)
    
    return MeanPoolingEmbedding(n_gpu, device, model_path)