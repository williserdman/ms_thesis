import unittest

import torch
from git_re_basin import apply_permutation
from torch import nn
from torch.nn import functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
except ModuleNotFoundError:
    Data = None
    GCNConv = None

if GCNConv is not None:
    from rebasin_bezier_repair.adapters.gcn import GCNAdapter
    from rebasin_bezier_repair.paths import bezier_state, fit_curve
    from rebasin_bezier_repair.repair import collect_moments, repair_state


if GCNConv is not None:
    class TestGCN(nn.Module):
        def __init__(self, *, cached=False, bias=True):
            super().__init__()
            self.convs = nn.ModuleList([
                GCNConv(4, 5, cached=cached, bias=bias),
                GCNConv(5, 4, cached=cached, bias=bias),
                GCNConv(4, 3, cached=cached, bias=bias),
            ])

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, edge_index))
            return self.convs[-1](x, edge_index)


def graph_case():
    return Data(
        x=torch.tensor([
            [1.0, 0.0, 0.5, -0.5],
            [0.0, 1.0, -0.5, 0.25],
            [1.0, 1.0, 0.25, 0.75],
            [-1.0, 0.5, 1.0, 0.0],
            [2.0, -1.0, 0.0, 1.0],
            [-0.5, 2.0, 1.5, -1.0],
        ]),
        edge_index=torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0, 4, 5],
            [1, 0, 2, 1, 3, 2, 0, 3, 5, 4],
        ]),
        y=torch.tensor([0, 1, 2, 1, 0, 2]),
        train_mask=torch.tensor([True, True, True, True, False, False]),
    )


def model_state(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TestGCN()
    return model, {name: value.detach().clone()
                   for name, value in model.state_dict().items()}


def train_nodes(site, activation, batch):
    return activation[batch.train_mask]


@unittest.skipIf(GCNConv is None, "torch-geometric is unavailable")
class GCNAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gcn_permutation_and_shared_curve(self):
        model, state = model_state(3)
        graph = graph_case()
        adapter = GCNAdapter()
        spec = adapter.permutation_spec(model, state)
        self.assertEqual(set(spec), set(state))
        groups = {group: state[key].shape[axis]
                  for key, axes in spec.items()
                  for axis, group in enumerate(axes) if group is not None}
        permutation = {group: torch.arange(width - 1, -1, -1)
                       for group, width in groups.items()}
        permuted = apply_permutation(spec, permutation, state)
        original_logits = adapter.forward(model, state, graph)
        permuted_logits = adapter.forward(model, permuted, graph)
        torch.testing.assert_close(
            original_logits, permuted_logits, rtol=1e-4, atol=1e-5,
        )
        sites = adapter.repair_sites(model)
        self.assertEqual([site.module_path for site in sites],
                         ["convs.0", "convs.1"])

        _, other = model_state(11)
        control = fit_curve(
            model, adapter, state, other, [graph],
            lambda logits, batch: F.cross_entropy(
                logits[batch.train_mask], batch.y[batch.train_mask],
            ),
            steps=3, seed=7,
        )
        self.assertEqual(set(control), set(dict(model.named_parameters())))
        self.assertTrue(all(torch.isfinite(value).all() for value in control.values()))
        self.assertTrue(any(
            not torch.equal(control[name], (state[name] + other[name]) / 2)
            for name in control
        ))

        unsupported = [TestGCN(cached=True), TestGCN(bias=False)]
        fixed = TestGCN()
        fixed.register_buffer("fixed", torch.tensor([1.0]))
        unsupported.append(fixed)
        for candidate in unsupported:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                adapter.permutation_spec(candidate, candidate.state_dict())

    def test_gcn_training_node_calibration(self):
        model, state = model_state(5)
        graph = graph_case()
        adapter = GCNAdapter()
        measured = collect_moments(
            model, adapter, state, [graph], selector=train_nodes,
        )

        activations = []
        handles = [conv.register_forward_hook(
            lambda module, inputs, output: activations.append(output.detach().clone())
        ) for conv in model.convs[:-1]]
        try:
            model.eval()
            with torch.no_grad():
                model(graph.x, graph.edge_index)
        finally:
            for handle in handles:
                handle.remove()

        for site, activation in zip(adapter.repair_sites(model), activations, strict=True):
            selected = activation[graph.train_mask]
            torch.testing.assert_close(measured[site.name].mean, selected.mean(0))
            torch.testing.assert_close(
                measured[site.name].std, selected.std(0, unbiased=False),
            )

    def test_gcn_repair_and_replay(self):
        template, _ = model_state(7)
        _, endpoint_a = model_state(13)
        _, endpoint_b = model_state(17)
        adapter = GCNAdapter()
        graph = graph_case()
        endpoint_moments = (
            collect_moments(template, adapter, endpoint_a, [graph], selector=train_nodes),
            collect_moments(template, adapter, endpoint_b, [graph], selector=train_nodes),
        )
        parameter_names = tuple(dict(template.named_parameters()))
        control = {name: (endpoint_a[name] + endpoint_b[name]) / 2
                   for name in parameter_names}
        t = 0.35
        sampled = bezier_state(
            endpoint_a, endpoint_b, control, t, parameter_names=parameter_names,
        )
        repaired = repair_state(
            template, adapter, sampled, t, endpoint_moments, [graph],
            selector=train_nodes,
        )
        actual = collect_moments(
            template, adapter, repaired, [graph], selector=train_nodes,
        )
        for site in adapter.repair_sites(template):
            target_mean = ((1 - t) * endpoint_moments[0][site.name].mean
                           + t * endpoint_moments[1][site.name].mean)
            target_std = ((1 - t) * endpoint_moments[0][site.name].std
                          + t * endpoint_moments[1][site.name].std)
            torch.testing.assert_close(
                actual[site.name].mean, target_mean, rtol=1e-4, atol=1e-5,
            )
            torch.testing.assert_close(
                actual[site.name].std, target_std, rtol=1e-4, atol=1e-5,
            )

        expected = adapter.forward(template, repaired, graph)
        replay = TestGCN()
        replay.load_state_dict(repaired)
        replay.eval()
        with torch.no_grad():
            actual_logits = replay(graph.x, graph.edge_index)
        torch.testing.assert_close(actual_logits, expected)


if __name__ == "__main__":
    unittest.main()
