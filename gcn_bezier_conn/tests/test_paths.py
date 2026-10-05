from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F
from torch_geometric.data import Data

from gcn_mc.model import GCN
from gcn_mc.paths import (
    clone_parameters,
    interpolate,
    make_control,
    path_logits,
    summarize_path,
)


def tiny_graph() -> Data:
    return Data(
        x=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
        ),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
        ),
        y=torch.tensor([0, 1, 0, 1]),
    )


def endpoint_states() -> tuple[GCN, GCN, GCN, Data]:
    graph = tiny_graph()
    torch.manual_seed(1)
    endpoint_a = GCN(2, 5, 2, depth=2, dropout=0.0).eval()
    torch.manual_seed(2)
    endpoint_b = GCN(2, 5, 2, depth=2, dropout=0.0).eval()
    template = GCN(2, 5, 2, depth=2, dropout=0.0).eval()
    return endpoint_a, endpoint_b, template, graph


class PathTests(unittest.TestCase):
    def test_gcn_depth_controls_real_convolution_stack(self) -> None:
        model = GCN(2, 5, 3, depth=4, dropout=0.25).eval()
        graph = tiny_graph()
        logits = model(graph.x, graph.edge_index)

        self.assertEqual(logits.shape, (4, 3))
        self.assertEqual(len(model.convs), 4)
        self.assertTrue(all(conv.cached is False for conv in model.convs))

    def test_path_endpoints_reproduce_independent_model_predictions(self) -> None:
        endpoint_a, endpoint_b, template, graph = endpoint_states()
        state_a = clone_parameters(endpoint_a)
        state_b = clone_parameters(endpoint_b)

        with torch.no_grad():
            expected_a = endpoint_a(graph.x, graph.edge_index)
            expected_b = endpoint_b(graph.x, graph.edge_index)
            actual_a = path_logits(template, graph, state_a, state_b, 0.0)
            actual_b = path_logits(template, graph, state_a, state_b, 1.0)

        torch.testing.assert_close(actual_a, expected_a)
        torch.testing.assert_close(actual_b, expected_b)

    def test_midpoint_control_reproduces_the_straight_parameter_path(self) -> None:
        endpoint_a, endpoint_b, _, _ = endpoint_states()
        state_a = clone_parameters(endpoint_a)
        state_b = clone_parameters(endpoint_b)
        control = make_control(state_a, state_b)

        curved = interpolate(state_a, state_b, 0.37, control)
        straight = interpolate(state_a, state_b, 0.37)

        self.assertTrue(
            all(isinstance(value, torch.nn.Parameter) for value in control.values())
        )
        for name in straight:
            torch.testing.assert_close(curved[name], straight[name])

    def test_control_optimization_changes_path_predictions_but_not_endpoints(self) -> None:
        endpoint_a, endpoint_b, template, graph = endpoint_states()
        state_a = clone_parameters(endpoint_a)
        state_b = clone_parameters(endpoint_b)
        frozen_a = {name: value.clone() for name, value in state_a.items()}
        frozen_b = {name: value.clone() for name, value in state_b.items()}
        control = make_control(state_a, state_b)
        optimizer = torch.optim.SGD(control.values(), lr=0.5)

        before = path_logits(template, graph, state_a, state_b, 0.4, control).detach()
        optimizer.zero_grad(set_to_none=True)
        logits = path_logits(template, graph, state_a, state_b, 0.4, control)
        F.cross_entropy(logits, graph.y).backward()

        self.assertTrue(all(parameter.grad is not None for parameter in control.values()))
        self.assertTrue(any(parameter.grad.norm().item() > 0 for parameter in control.values()))
        optimizer.step()
        after = path_logits(template, graph, state_a, state_b, 0.4, control).detach()

        self.assertFalse(torch.allclose(before, after))
        for name in state_a:
            torch.testing.assert_close(state_a[name], frozen_a[name])
            torch.testing.assert_close(state_b[name], frozen_b[name])

    def test_summarize_path_uses_endpoint_linear_loss_baseline(self) -> None:
        summary = summarize_path([0.0, 0.5, 1.0], [1.0, 2.1, 3.0])

        self.assertAlmostEqual(summary["barrier"], 0.1)
        self.assertEqual(summary["argmax_t"], 0.5)
        self.assertEqual(summary["max_loss"], 3.0)


if __name__ == "__main__":
    unittest.main()
