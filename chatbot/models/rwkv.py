"""Defines a PyTorch model for fine-tuning a pre-trained LLM."""

from dataclasses import dataclass
from typing import Iterator, Sequence

import ml.api as ml
import torch
from rwkv.model import pretrained_rwkv
from torch import Tensor


@dataclass
class RwkvChatbotModelConfig(ml.BaseModelConfig):
    lora_rank: int = ml.conf_field(4, help="The rank of the LoRA approximation.")


@ml.register_model("rwkv", RwkvChatbotModelConfig)
class RwkvChatbotModel(ml.BaseModel[RwkvChatbotModelConfig]):
    def __init__(self, config: RwkvChatbotModelConfig) -> None:
        super().__init__(config)

        self.rwkv = pretrained_rwkv("1.5b", lora_rank=config.lora_rank, wkv_impl="eps")
        self.predictor = self.rwkv.predictor()

    def forward(self, tokens: Tensor) -> Tensor:
        preds, _ = self.rwkv.forward(tokens, return_logits=True)
        return preds

    def infer(
        self,
        prompt: str,
        max_len: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.85,
        end_toks: Sequence[int] | None = None,
        end_strs: Sequence[str] | None = None,
    ) -> Iterator[str]:
        yield from self.predictor.generate(prompt, max_len, temperature, top_p, end_toks, end_strs)

    def tokens_to_string(self, tokens: Tensor) -> str:
        return self.predictor.tokenizer.decode(tokens.tolist())

    def string_to_tokens(self, text: str) -> Tensor:
        return torch.tensor(self.predictor.tokenizer.encode(text).ids)
