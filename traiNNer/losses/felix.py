import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.transforms import Compose


def create_weights(
    start_weight: float, end_weight: float, count: int, curve_weight: int
) -> np.ndarray:
    weights = np.linspace(start_weight, end_weight, count)
    for i in range(len(weights)):
        weights[i] = math.pow(weights[i], curve_weight)
    return weights


class FelixExtractor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        hook_instance: type[nn.Module],
        transforms: Compose,
        start_weight: float = 1.0,
        end_weight: float = 1.0,
        curve_weight: int = 1,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        if curve_weight < 1:
            raise Exception("Curve can't be smaller than 1")

        self.activations = []

        def get_activation() -> Callable:
            def hook(model: nn.Module, input: Any, output: Tensor) -> None:
                self.activations.append(output)

            return hook

        if verbose:
            print("Loading model")

        self.model = model.eval()

        count = 0

        def traverse_modules(module: nn.Module) -> None:
            nonlocal count, verbose
            for name, sub_module in module.named_children():
                # full_name = parent_name + '.' + name if parent_name else name
                if isinstance(sub_module, hook_instance):
                    count += 1
                    if verbose:
                        print(f"-> {sub_module}")
                    sub_module.register_forward_hook(get_activation())
                elif isinstance(sub_module, nn.ReLU):
                    setattr(module, name, nn.ReLU(False))
                else:
                    traverse_modules(sub_module)

        traverse_modules(self.model)
        if verbose:
            print(f"Total Layers: {count}")

        self.weights = create_weights(start_weight, end_weight, count, curve_weight)
        self.transforms = transforms

    def forward(self, x: Tensor) -> list:
        x = self.transforms(x)
        self.activations = []
        self.model(x)
        return self.activations


class FelixLoss(torch.nn.Module):
    def __init__(
        self, extractor: FelixExtractor, loss: nn.Module | None = None
    ) -> None:
        super().__init__()
        if loss is None:
            loss = nn.L1Loss()
        self.extractor = extractor
        self.loss = loss

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.extractor(x)
        y = self.extractor(y)
        loss = torch.tensor(0, device=x.device)
        for i in range(len(x)):
            loss += self.loss(x[i], y[i].detach()) * self.extractor.weights[i]

        return loss
