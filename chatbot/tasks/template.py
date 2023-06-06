"""Defines a simple supervised learning template task.

This task is meant to be used as a template for creating new tasks. Just
change the key from ``template`` to whatever you want to name your task, and
implement the following methods:

- :meth:`run_model`
- :meth:`compute_loss`
- :meth:`get_dataset`
"""

from dataclasses import dataclass

import ml.api as ml
from torch import Tensor
from torch.utils.data.dataset import Dataset


@dataclass
class TemplateTaskConfig(ml.SupervisedLearningTaskConfig):
    pass


# These types are defined here so that they can be used consistently
# throughout the task and only changed in one location.
Model = ml.BaseModel
Batch = tuple[Tensor, Tensor]
Output = Tensor
Loss = Tensor


@ml.register_task("template", TemplateTaskConfig)
class TemplateTask(ml.SupervisedLearningTask[TemplateTaskConfig, Model, Batch, Output, Loss]):
    def run_model(self, model: Model, batch: Batch, state: ml.State) -> Output:
        raise NotImplementedError

    def compute_loss(self, model: Model, batch: Batch, state: ml.State, output: Output) -> Loss:
        raise NotImplementedError

    def get_dataset(self, phase: ml.Phase) -> Dataset:
        raise NotImplementedError
