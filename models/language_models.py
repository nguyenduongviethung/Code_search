from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LanguageModel(torch.nn.Module):
    """
    Wrapper for loading and running inference with a causal language model.
    """

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device
        )

        # Configure padding for batch inference (left padding for decoder-only models)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate(
        self,
        contents: str | list[str],
        max_input_tokens: int = 512,
        max_new_tokens: int = 128
    ):
        """
        Generate outputs for a batch of input prompts.

        Args:
            contents: str or list[str]
            max_input_tokens: maximum length for input truncation
            max_new_tokens: maximum number of tokens to generate

        Returns:
            List[str]: decoded outputs
        """

        # =========================
        # Normalize input to list
        # =========================
        if isinstance(contents, str):
            contents = [contents]

        # =========================
        # Build chat-style messages
        # =========================
        messages = [
            [{"role": "user", "content": c}]
            for c in contents
        ]

        # =========================
        # Apply chat template (model-specific formatting)
        # =========================
        text_batch = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # =========================
        # Tokenization
        # =========================
        inputs = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_input_tokens,
        ).to(self.model.device)

        # =========================
        # Generation
        # =========================
        outputs = self.model.generate( # type: ignore
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic decoding
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # =========================
        # Remove prompt tokens from generated output
        # =========================
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

        # =========================
        # Decode tokens into text
        # =========================
        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return decoded