"""Run the shared connectivity pipeline on a small synthetic graph."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from rebasin_bezier_repair import connect
from rebasin_bezier_repair.adapters.gcn import GCNAdapter


class GCN(nn.Module):
    """Biased, uncached two-layer GCN adapted from gcn_mc/model.py."""

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(4, 8, cached=False, bias=True),
            GCNConv(8, 3, cached=False, bias=True),
        ])

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.convs[0](x, edge_index))
        return self.convs[1](x, edge_index)


def make_graph() -> Data:
    generator = torch.Generator().manual_seed(42)
    features = torch.randn(60, 4, generator=generator)
    teacher = torch.tensor([
        [1.2, -0.5, 0.4],
        [-0.7, 1.0, 0.3],
        [0.6, 0.2, -1.1],
        [0.1, -0.8, 1.0],
    ])
    labels = (features @ teacher).argmax(dim=1)

    nodes = torch.arange(60)
    successors = (nodes + 1) % 60
    skips = (nodes + 5) % 60
    edge_index = torch.stack((
        torch.cat((nodes, successors, nodes, skips)),
        torch.cat((successors, nodes, skips, nodes)),
    ))

    train_mask = torch.zeros(60, dtype=torch.bool)
    validation_mask = torch.zeros(60, dtype=torch.bool)
    test_mask = torch.zeros(60, dtype=torch.bool)
    train_mask[:36] = True
    validation_mask[36:48] = True
    test_mask[48:] = True
    return Data(
        x=features,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
    )


def make_model(seed: int) -> GCN:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return GCN()


def training_loss(logits: Tensor, batch: object) -> Tensor:
    return F.cross_entropy(logits[batch.train_mask], batch.y[batch.train_mask])


def training_nodes(site: object, activation: Tensor, batch: object) -> Tensor:
    return activation[batch.train_mask]


def train(model: GCN, graph: Data) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        logits = model(graph.x, graph.edge_index)
        loss = F.cross_entropy(
            logits[graph.train_mask], graph.y[graph.train_mask],
        )
        loss.backward()
        optimizer.step()


@torch.no_grad()
def metrics(model: GCN, graph: Data) -> tuple[float, float]:
    model.eval()
    logits = model(graph.x, graph.edge_index)[graph.test_mask]
    labels = graph.y[graph.test_mask]
    loss = F.cross_entropy(logits, labels).item()
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    if not math.isfinite(loss) or not math.isfinite(accuracy):
        raise RuntimeError("example produced nonfinite metrics")
    return loss, accuracy


def main() -> None:
    torch.set_num_threads(1)
    graph = make_graph()
    model_a = make_model(0)
    model_b = make_model(1)
    train(model_a, graph)
    train(model_b, graph)

    path = connect(
        model_a,
        model_b,
        adapter=GCNAdapter(),
        train_data=[graph],
        loss_fn=training_loss,
        calibration_data=[graph],
        select_observations=training_nodes,
    )
    midpoint = path.model_at(0.5)
    repaired_midpoint = path.model_at(0.5, repair=True, calibration_data=[graph])

    evaluated = (
        ("endpoint_a", model_a),
        ("endpoint_b", model_b),
        ("bezier_midpoint", midpoint),
        ("repaired_midpoint", repaired_midpoint),
    )
    for name, model in evaluated:
        loss, accuracy = metrics(model, graph)
        print(f"{name} loss={loss:.6f} accuracy={accuracy:.6f}")


if __name__ == "__main__":
    main()
