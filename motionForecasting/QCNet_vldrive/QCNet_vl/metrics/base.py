# Minimal stand-in for torchmetrics.Metric: accumulates sum/count on a chosen device.
import torch


class Metric:

    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        self.device = device
        self.sum = torch.tensor(0.0, device=device)
        self.count = torch.tensor(0, device=device)

    def to(self, device: torch.device) -> 'Metric':
        self.device = device
        self.sum = self.sum.to(device)
        self.count = self.count.to(device)
        return self

    def reset(self) -> None:
        self.sum = torch.tensor(0.0, device=self.device)
        self.count = torch.tensor(0, device=self.device)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
