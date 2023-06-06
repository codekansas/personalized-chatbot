"""Defines a PyTorch model for fine-tuning a pre-trained LLM."""

from dataclasses import dataclass

import ml.api as ml
from pretrained.rwkv import pretrained_rwkv
from torch import Tensor


@dataclass
class ChatbotModelConfig(ml.BaseModelConfig):
    pass


@ml.register_model("chatbot", ChatbotModelConfig)
class ChatbotModel(ml.BaseModel[ChatbotModelConfig]):
    def __init__(self, config: ChatbotModelConfig) -> None:
        super().__init__(config)

        self.rwkv = pretrained_rwkv("430m")

    def forward(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError
