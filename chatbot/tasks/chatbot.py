"""Defines a task for fine-tuning an LLM on Facebook Messenger conversations."""

from dataclasses import dataclass

import ml.api as ml
from torch import Tensor
from torch.utils.data.dataset import Dataset

from chatbot.models.llm import ChatbotModel


@dataclass
class ChatbotTaskConfig(ml.SupervisedLearningTaskConfig):
    pass


# These types are defined here so that they can be used consistently
# throughout the task and only changed in one location.
Model = ChatbotModel
Batch = tuple[Tensor, Tensor]
Output = Tensor
Loss = Tensor


@ml.register_task("chatbot", ChatbotTaskConfig)
class ChatbotTask(ml.SupervisedLearningTask[ChatbotTaskConfig, Model, Batch, Output, Loss]):
    def run_model(self, model: Model, batch: Batch, state: ml.State) -> Output:
        raise NotImplementedError

    def compute_loss(self, model: Model, batch: Batch, state: ml.State, output: Output) -> Loss:
        raise NotImplementedError

    def get_dataset(self, phase: ml.Phase) -> Dataset:
        raise NotImplementedError
