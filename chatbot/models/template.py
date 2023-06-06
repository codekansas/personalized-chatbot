"""Defines a template mode.

To use this, change the key from ``"template"`` to whatever your project
is called. Next, just override the ``forward`` model to whatever signature
your task expects, and you're good to go!
"""

from dataclasses import dataclass

import ml.api as ml
from torch import Tensor


@dataclass
class TemplateModelConfig(ml.BaseModelConfig):
    pass


@ml.register_model("template", TemplateModelConfig)
class TemplateModel(ml.BaseModel[TemplateModelConfig]):
    def __init__(self, config: TemplateModelConfig) -> None:
        super().__init__(config)

        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
