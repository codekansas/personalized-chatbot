"""Defines a PyTorch model for fine-tuning a pre-trained LLM."""

from dataclasses import dataclass

import ml.api as ml
from pretrained.rwkv import pretrained_rwkv
from torch import Tensor


@dataclass
class ChatbotModelConfig(ml.BaseModelConfig):
    model_size: str = ml.conf_field("430m", help="The size of the pre-trained model.")
    lora_rank: int = ml.conf_field(4, help="The rank of the LoRA approximation.")


@ml.register_model("chatbot", ChatbotModelConfig)
class ChatbotModel(ml.BaseModel[ChatbotModelConfig]):
    def __init__(self, config: ChatbotModelConfig) -> None:
        super().__init__(config)

        self.rwkv = pretrained_rwkv("430m", lora_rank=config.lora_rank)
        self.predictor = self.rwkv.predictor()

    def forward(self, tokens: Tensor) -> Tensor:
        preds, _ = self.rwkv.forward(tokens)
        return preds

    def infer(self, prompt: str) -> str:
        return "".join(list(self.predictor.generate(prompt)))
