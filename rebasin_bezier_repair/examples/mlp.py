"""Run the shared connectivity pipeline on a small synthetic MLP problem."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rebasin_bezier_repair import MLPAdapter, connect


def make_data() -> tuple[list[tuple[Tensor, Tensor]], tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(42)
    features = torch.randn(96, 4, generator=generator)
    teacher = torch.tensor([
        [1.3, -0.7, 0.4],
        [-0.8, 1.1, 0.2],
        [0.5, 0.3, -1.2],
        [0.2, -0.9, 1.0],
    ])
    labels = (features @ teacher).argmax(dim=1)
    train_batches = [
        (features[start:start + 16], labels[start:start + 16])
        for start in range(0, 64, 16)
    ]
    return train_batches, (features[64:], labels[64:])


def make_model(seed: int) -> nn.Sequential:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
        )


def batch_loss(logits: Tensor, batch: object) -> Tensor:
    _, labels = batch
    return F.cross_entropy(logits, labels)


def train(model: nn.Module, batches: list[tuple[Tensor, Tensor]]) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    for _ in range(50):
        for features, labels in batches:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(features), labels)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def metrics(model: nn.Module, batch: tuple[Tensor, Tensor]) -> tuple[float, float]:
    model.eval()
    features, labels = batch
    logits = model(features)
    loss = F.cross_entropy(logits, labels).item()
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    if not math.isfinite(loss) or not math.isfinite(accuracy):
        raise RuntimeError("example produced nonfinite metrics")
    return loss, accuracy


def main() -> None:
    torch.set_num_threads(1)
    train_batches, test_batch = make_data()
    model_a = make_model(0)
    model_b = make_model(1)
    train(model_a, train_batches)
    train(model_b, train_batches)

    path = connect(
        model_a,
        model_b,
        adapter=MLPAdapter(),
        train_data=train_batches,
        loss_fn=batch_loss,
        calibration_data=train_batches,
    )
    midpoint = path.model_at(0.5)
    repaired_midpoint = path.model_at(
        0.5, repair=True, calibration_data=train_batches,
    )

    evaluated = (
        ("endpoint_a", model_a),
        ("endpoint_b", model_b),
        ("bezier_midpoint", midpoint),
        ("repaired_midpoint", repaired_midpoint),
    )
    for name, model in evaluated:
        loss, accuracy = metrics(model, test_batch)
        print(f"{name} loss={loss:.6f} accuracy={accuracy:.6f}")


if __name__ == "__main__":
    main()
