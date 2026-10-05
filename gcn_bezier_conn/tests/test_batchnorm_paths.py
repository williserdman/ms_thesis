from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from gcn_mc.paths import (
    calibrate_batchnorm,
    clone_parameters,
    make_control,
    path_logits,
)


class TinyBatchNormGraphModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fail_after_norm = False
        self.input = nn.Linear(2, 4, bias=False)
        self.norm = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(0.75)
        self.output = nn.Linear(4, 2, bias=False)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        source, target = edge_index
        neighbors = torch.zeros_like(x)
        neighbors.index_add_(0, target, x[source])
        hidden = self.input(x + neighbors)
        hidden = torch.relu(self.norm(hidden))
        if self.fail_after_norm:
            raise RuntimeError("failure after BatchNorm")
        return self.output(self.dropout(hidden))


def tiny_graph(*, labels: torch.Tensor | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        x=torch.tensor(
            [
                [1.0, -0.5],
                [0.25, 1.5],
                [-1.0, 0.75],
                [2.0, 0.5],
                [-0.25, -1.5],
            ]
        ),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 1, 3], [1, 2, 3, 4, 0, 4, 1]]
        ),
        y=torch.tensor([0, 1, 0, 1, 0]) if labels is None else labels,
    )


def cloned_named_tensors(
    tensors: object,
) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in tensors}


class BatchNormCalibrationTests(unittest.TestCase):
    def test_calibration_is_label_free_deterministic_and_non_accumulating(self) -> None:
        torch.manual_seed(7)
        first = TinyBatchNormGraphModel().train()
        first.dropout.eval()
        first.norm.momentum = 0.23
        second = copy.deepcopy(first)
        graph = tiny_graph()
        relabeled = tiny_graph(labels=torch.tensor([9, 8, 7, 6, 5]))
        parameters = cloned_named_tensors(first.named_parameters())
        modes = {name: module.training for name, module in first.named_modules()}

        calibrate_batchnorm(first, graph)
        once = cloned_named_tensors(first.named_buffers())
        calibrate_batchnorm(first, graph)
        twice = cloned_named_tensors(first.named_buffers())
        calibrate_batchnorm(second, relabeled)
        relabeled_stats = cloned_named_tensors(second.named_buffers())

        self.assertEqual(int(first.norm.num_batches_tracked), 1)
        self.assertEqual(first.norm.momentum, 0.23)
        self.assertEqual(
            {name: module.training for name, module in first.named_modules()}, modes
        )
        for name, value in first.named_parameters():
            torch.testing.assert_close(value, parameters[name])
        for name in once:
            torch.testing.assert_close(twice[name], once[name])
            torch.testing.assert_close(relabeled_stats[name], once[name])

    def test_failed_calibration_restores_buffers_parameters_and_modes(self) -> None:
        torch.manual_seed(11)
        model = TinyBatchNormGraphModel().train()
        model.norm.eval()
        model.dropout.eval()
        model.norm.momentum = 0.37
        with torch.no_grad():
            model.norm.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
            model.norm.running_var.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
            model.norm.num_batches_tracked.fill_(8)
        parameters = cloned_named_tensors(model.named_parameters())
        buffers = cloned_named_tensors(model.named_buffers())
        modes = {name: module.training for name, module in model.named_modules()}
        model.fail_after_norm = True

        with self.assertRaisesRegex(RuntimeError, "failure after BatchNorm"):
            calibrate_batchnorm(model, tiny_graph())

        self.assertEqual(model.norm.momentum, 0.37)
        self.assertEqual(
            {name: module.training for name, module in model.named_modules()}, modes
        )
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, parameters[name])
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name])


class BatchNormPathTests(unittest.TestCase):
    def test_eval_path_replays_calibrated_endpoints_without_mutating_template(self) -> None:
        graph = tiny_graph()
        torch.manual_seed(21)
        endpoint_a = TinyBatchNormGraphModel().eval()
        torch.manual_seed(22)
        endpoint_b = TinyBatchNormGraphModel().eval()
        calibrate_batchnorm(endpoint_a, graph)
        calibrate_batchnorm(endpoint_b, graph)
        endpoint_a.eval()
        endpoint_b.eval()
        with torch.no_grad():
            expected_a = endpoint_a(graph.x, graph.edge_index)
            expected_b = endpoint_b(graph.x, graph.edge_index)

        torch.manual_seed(23)
        template = TinyBatchNormGraphModel().eval()
        template.dropout.train()
        with torch.no_grad():
            template.norm.running_mean.fill_(17.0)
            template.norm.running_var.fill_(3.0)
            template.norm.num_batches_tracked.fill_(12)
        parameters = cloned_named_tensors(template.named_parameters())
        buffers = cloned_named_tensors(template.named_buffers())
        modes = {name: module.training for name, module in template.named_modules()}
        state_a = clone_parameters(endpoint_a)
        state_b = clone_parameters(endpoint_b)

        actual_a = path_logits(template, graph, state_a, state_b, 0.0)
        actual_b = path_logits(template, graph, state_a, state_b, 1.0)
        repeated_a = path_logits(template, graph, state_a, state_b, 0.0)

        torch.testing.assert_close(actual_a, expected_a)
        torch.testing.assert_close(actual_b, expected_b)
        torch.testing.assert_close(repeated_a, actual_a)
        self.assertEqual(
            {name: module.training for name, module in template.named_modules()}, modes
        )
        for name, value in template.named_parameters():
            torch.testing.assert_close(value, parameters[name])
        for name, value in template.named_buffers():
            torch.testing.assert_close(value, buffers[name])

    def test_eval_path_retains_control_gradients_after_calibration(self) -> None:
        graph = tiny_graph()
        torch.manual_seed(26)
        endpoint_a = TinyBatchNormGraphModel()
        torch.manual_seed(27)
        endpoint_b = TinyBatchNormGraphModel()
        template = TinyBatchNormGraphModel().eval()
        control = make_control(
            clone_parameters(endpoint_a), clone_parameters(endpoint_b)
        )
        optimizer = torch.optim.SGD(control.values(), lr=0.4)

        before = path_logits(
            template,
            graph,
            clone_parameters(endpoint_a),
            clone_parameters(endpoint_b),
            0.4,
            control,
        ).detach()
        optimizer.zero_grad(set_to_none=True)
        logits = path_logits(
            template,
            graph,
            clone_parameters(endpoint_a),
            clone_parameters(endpoint_b),
            0.4,
            control,
        )
        F.cross_entropy(logits, graph.y).backward()

        self.assertTrue(all(value.grad is not None for value in control.values()))
        self.assertTrue(any(value.grad.norm().item() > 0 for value in control.values()))
        optimizer.step()
        after = path_logits(
            template,
            graph,
            clone_parameters(endpoint_a),
            clone_parameters(endpoint_b),
            0.4,
            control,
        ).detach()

        self.assertFalse(torch.allclose(after, before))

    def test_training_path_preserves_control_gradients_and_template_state(self) -> None:
        graph = tiny_graph()
        torch.manual_seed(31)
        endpoint_a = TinyBatchNormGraphModel()
        torch.manual_seed(32)
        endpoint_b = TinyBatchNormGraphModel()
        torch.manual_seed(33)
        template = TinyBatchNormGraphModel().train()
        state_a = clone_parameters(endpoint_a)
        state_b = clone_parameters(endpoint_b)
        control = make_control(state_a, state_b)
        optimizer = torch.optim.SGD(control.values(), lr=0.4)
        parameters = cloned_named_tensors(template.named_parameters())
        buffers = cloned_named_tensors(template.named_buffers())
        modes = {name: module.training for name, module in template.named_modules()}

        torch.manual_seed(41)
        before = path_logits(
            template, graph, state_a, state_b, 0.4, control
        ).detach()
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(41)
        logits = path_logits(template, graph, state_a, state_b, 0.4, control)
        F.cross_entropy(logits, graph.y).backward()

        self.assertTrue(all(value.grad is not None for value in control.values()))
        self.assertTrue(any(value.grad.norm().item() > 0 for value in control.values()))
        optimizer.step()
        torch.manual_seed(41)
        after = path_logits(
            template, graph, state_a, state_b, 0.4, control
        ).detach()

        self.assertFalse(torch.allclose(after, before))
        self.assertEqual(
            {name: module.training for name, module in template.named_modules()}, modes
        )
        for name, value in template.named_parameters():
            torch.testing.assert_close(value, parameters[name])
        for name, value in template.named_buffers():
            torch.testing.assert_close(value, buffers[name])

    def test_failed_eval_path_restores_template_buffers_and_modes(self) -> None:
        graph = tiny_graph()
        torch.manual_seed(51)
        endpoint_a = TinyBatchNormGraphModel()
        torch.manual_seed(52)
        endpoint_b = TinyBatchNormGraphModel()
        template = TinyBatchNormGraphModel().eval()
        template.dropout.train()
        template.fail_after_norm = True
        buffers = cloned_named_tensors(template.named_buffers())
        modes = {name: module.training for name, module in template.named_modules()}

        with self.assertRaisesRegex(RuntimeError, "failure after BatchNorm"):
            path_logits(
                template,
                graph,
                clone_parameters(endpoint_a),
                clone_parameters(endpoint_b),
                0.5,
            )

        self.assertEqual(
            {name: module.training for name, module in template.named_modules()}, modes
        )
        for name, value in template.named_buffers():
            torch.testing.assert_close(value, buffers[name])


if __name__ == "__main__":
    unittest.main()
