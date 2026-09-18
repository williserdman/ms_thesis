from __future__ import annotations

import copy
import unittest

import torch
from torch.nn import functional as F
from torch_geometric.data import Data

from gcn_mc.model import GCN
from gcn_mc.repair_adapter import align_gcn, hidden_statistics, repair_gcn


torch.set_num_threads(1)


def unequal_degree_graph() -> Data:
    return Data(
        x=torch.tensor(
            [
                [0.2, 1.1, -0.3, 0.7],
                [1.3, -0.4, 0.8, 0.2],
                [-0.5, 0.6, 1.4, -0.2],
                [0.9, 0.1, -0.7, 1.2],
                [-1.0, 0.4, 0.5, 0.3],
                [0.3, -1.2, 0.2, 1.1],
                [1.5, 0.8, -0.1, -0.6],
            ],
            dtype=torch.float,
        ),
        edge_index=torch.tensor(
            [
                [0, 1, 0, 2, 0, 3, 0, 4, 4, 5, 5, 6],
                [1, 0, 2, 0, 3, 0, 4, 0, 5, 4, 6, 5],
            ]
        ),
        train_mask=torch.tensor([True, True, True, True, False, False, False]),
    )


def configured_gcn(depth: int) -> GCN:
    torch.manual_seed(31 + depth)
    model = GCN(4, 4, 2, depth=depth, dropout=0.4)
    with torch.no_grad():
        for conv in model.convs[:-1]:
            conv.lin.weight.copy_(torch.eye(4))
            conv.bias.copy_(torch.tensor([0.3, 0.5, 0.7, 0.9]))
    return model


def permute_hidden(model: GCN, index: int, permutation: torch.Tensor) -> None:
    conv = model.convs[index]
    following = model.convs[index + 1]
    with torch.no_grad():
        conv.lin.weight.copy_(conv.lin.weight.index_select(0, permutation))
        conv.bias.copy_(conv.bias.index_select(0, permutation))
        following.lin.weight.copy_(following.lin.weight.index_select(1, permutation))


def inverse(permutation: torch.Tensor) -> list[int]:
    return torch.argsort(permutation).tolist()


def preactivation(model: GCN, graph: Data, index: int) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    handle = model.convs[index].register_forward_hook(
        lambda _module, _args, output: captured.append(output.detach().clone())
    )
    flags = [module.training for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            model(graph.x, graph.edge_index)
    finally:
        handle.remove()
        for module, training in zip(model.modules(), flags):
            module.training = training
    return captured[0]


def state_copy(model: GCN) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


class RepairAdapterTests(unittest.TestCase):
    def test_alignment_recovers_known_two_layer_permutation_and_preserves_logits(self) -> None:
        graph = unequal_degree_graph()
        reference = configured_gcn(depth=2).eval()
        candidate = copy.deepcopy(reference).eval()
        permutation = torch.tensor([2, 0, 3, 1])
        permute_hidden(candidate, 0, permutation)
        before_state = state_copy(candidate)

        with torch.no_grad():
            expected = candidate(graph.x, graph.edge_index)
        aligned, diagnostics = align_gcn(reference, candidate, graph)

        torch.testing.assert_close(aligned(graph.x, graph.edge_index), expected)
        self.assertEqual(diagnostics["permutations"]["convs.0"], inverse(permutation))
        self.assertLessEqual(diagnostics["max_logit_error"], 1e-6)
        for name, value in candidate.state_dict().items():
            torch.testing.assert_close(value, before_state[name])

    def test_alignment_preserves_logits_for_every_hidden_layer_of_deeper_gcn(self) -> None:
        graph = unequal_degree_graph()
        reference = configured_gcn(depth=3).eval()
        candidate = copy.deepcopy(reference).eval()
        first = torch.tensor([1, 3, 0, 2])
        second = torch.tensor([2, 0, 3, 1])
        permute_hidden(candidate, 0, first)
        permute_hidden(candidate, 1, second)

        with torch.no_grad():
            expected = candidate(graph.x, graph.edge_index)
        aligned, diagnostics = align_gcn(reference, candidate, graph)

        torch.testing.assert_close(aligned(graph.x, graph.edge_index), expected)
        self.assertEqual(diagnostics["permutations"]["convs.0"], inverse(first))
        self.assertEqual(diagnostics["permutations"]["convs.1"], inverse(second))

    def test_hidden_statistics_use_only_training_nodes_and_preserve_model(self) -> None:
        graph = unequal_degree_graph()
        model = configured_gcn(depth=2)
        model.train()
        model.convs[0].eval()
        flags = [module.training for module in model.modules()]
        before_state = state_copy(model)
        hidden = preactivation(model, graph, 0)

        statistics = hidden_statistics(model, graph)
        mean, std = statistics["convs.0"]
        expected = hidden[graph.train_mask].double()

        torch.testing.assert_close(mean.double(), expected.mean(dim=0))
        torch.testing.assert_close(std.double(), expected.std(dim=0, unbiased=False))
        self.assertFalse(torch.allclose(mean, hidden.mean(dim=0)))
        self.assertEqual([module.training for module in model.modules()], flags)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before_state[name])

    def test_repair_fusion_matches_explicit_post_message_passing_affine(self) -> None:
        graph = unequal_degree_graph()
        reference = configured_gcn(depth=2)
        torch.manual_seed(91)
        aligned = GCN(4, 4, 2, depth=2, dropout=0.4)
        reference.train()
        aligned.train()
        reference_flags = [module.training for module in reference.modules()]
        aligned_flags = [module.training for module in aligned.modules()]
        reference_state = state_copy(reference)
        aligned_state = state_copy(aligned)
        alpha = 0.4

        merged = copy.deepcopy(reference).eval()
        merged.load_state_dict(
            {
                name: (1.0 - alpha) * reference_state[name] + alpha * aligned_state[name]
                for name in reference_state
            }
        )
        source = preactivation(merged, graph, 0)
        endpoint_a = preactivation(reference, graph, 0)[graph.train_mask].double()
        endpoint_b = preactivation(aligned, graph, 0)[graph.train_mask].double()
        source_train = source[graph.train_mask].double()
        target_mean = (1.0 - alpha) * endpoint_a.mean(0) + alpha * endpoint_b.mean(0)
        target_std = (1.0 - alpha) * endpoint_a.std(0, unbiased=False) + alpha * endpoint_b.std(0, unbiased=False)
        source_mean = source_train.mean(0)
        source_std = source_train.std(0, unbiased=False)
        scale = target_std / torch.sqrt(source_std.square() + 1e-5)
        shift = target_mean - scale * source_mean
        self.assertGreater(shift.abs().max().item(), 1e-4)
        expected = source.double() * scale + shift

        repaired, diagnostics = repair_gcn(reference, aligned, graph, alpha)
        actual = preactivation(repaired, graph, 0)

        torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-6)
        self.assertIsNone(repaired.convs[0].lin.bias)
        layer = diagnostics["layers"]["convs.0"]
        for key in (
            "pre_mean_variance",
            "target_mean_variance",
            "post_mean_variance",
            "max_mean_target_residual",
            "max_std_target_residual",
        ):
            self.assertIsInstance(layer[key], float)
        self.assertEqual(diagnostics["epsilon"], 1e-5)
        self.assertEqual([module.training for module in reference.modules()], reference_flags)
        self.assertEqual([module.training for module in aligned.modules()], aligned_flags)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, reference_state[name])
        for name, value in aligned.state_dict().items():
            torch.testing.assert_close(value, aligned_state[name])

        for endpoint_alpha, endpoint in ((0.0, reference), (1.0, aligned)):
            endpoint_copy, _ = repair_gcn(reference, aligned, graph, endpoint_alpha)
            self.assertFalse(endpoint_copy.training)
            for name, value in endpoint.state_dict().items():
                torch.testing.assert_close(endpoint_copy.state_dict()[name], value)


if __name__ == "__main__":
    unittest.main()
