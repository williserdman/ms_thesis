import copy
import importlib.util
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from rebasin_bezier_repair import MLPAdapter, connect
from tests.support import copy_state, mlp_case


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.a, self.batches = mlp_case(0)
        self.b, _ = mlp_case(1)
        self.options = dict(adapter=MLPAdapter(), train_data=self.batches,
                            loss_fn=lambda logits, batch: F.cross_entropy(logits, batch[1]),
                            calibration_data=self.batches, curve_steps=3)

    def test_connect_preserves_endpoints_and_replays_models(self):
        self.a.train()
        self.a[1].eval()
        self.b.eval()
        for model in (self.a, self.b):
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
        before = [copy_state(model) for model in (self.a, self.b)]
        modes = [[m.training for m in model.modules()] for model in (self.a, self.b)]
        path = connect(self.a, self.b, **self.options)
        x = self.batches[0][0]
        with torch.no_grad():
            for t, model in ((0., self.a), (1., self.b)):
                torch.testing.assert_close(path.model_at(t, repair=True)(x), model(x),
                                           rtol=1e-4, atol=1e-5)
        repaired = path.model_at(.5, repair=True, calibration_data=self.batches)
        self.assertFalse(repaired.training)
        restored = copy.deepcopy(self.a).eval()
        restored.load_state_dict(repaired.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(restored(x), repaired(x), rtol=0, atol=0)
        for i, model in enumerate((self.a, self.b)):
            self.assertEqual(modes[i], [m.training for m in model.modules()])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[i][key]))
            for parameter in model.parameters():
                self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter)))
        altered = path.model_at(0)
        with torch.no_grad():
            next(altered.parameters()).zero_()
        for key, value in path.endpoint_a.items():
            self.assertTrue(torch.equal(value, before[0][key]))

    def test_connect_rejects_incompatible_inputs(self):
        wrong = nn.Sequential(nn.Linear(4, 7), nn.ReLU(), nn.Linear(7, 6),
                              nn.ReLU(), nn.Linear(6, 3))
        for candidate in (wrong, copy.deepcopy(self.b).double()):
            with self.subTest(candidate=type(candidate)), self.assertRaises(ValueError):
                connect(self.a, candidate, **self.options)
        for changes in ({"calibration_data": []}, {"train_data": iter(self.batches)}):
            with self.subTest(changes=tuple(changes)), self.assertRaises((ValueError, TypeError)):
                connect(self.a, self.b, **(self.options | changes))
        path = connect(self.a, self.b, **self.options)
        for t in (-.1, 1.1, float("nan")):
            with self.subTest(t=t), self.assertRaises(ValueError):
                path.model_at(t)
        with self.assertRaisesRegex(ValueError, "calibration"):
            path.model_at(.5, repair=True)
        self.a.register_buffer("fixed", torch.tensor([1]))
        self.b.register_buffer("fixed", torch.tensor([2]))
        with self.assertRaisesRegex(ValueError, "fixed"):
            connect(self.a, self.b, **self.options)
        self.b.fixed.fill_(1)
        fixed_path = connect(self.a, self.b, **self.options)
        self.assertTrue(torch.equal(fixed_path.model_at(.5).fixed, torch.tensor([1])))

    @unittest.skipUnless(importlib.util.find_spec("torch_geometric"), "PyG not installed")
    def test_gcn_held_out_labels_do_not_affect_path(self):
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        from rebasin_bezier_repair.adapters.gcn import GCNAdapter

        class TinyGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList([GCNConv(3, 4, cached=False),
                                            GCNConv(4, 2, cached=False)])
                self.dropout = .2

            def forward(self, x, edge_index):
                h = F.relu(self.convs[0](x, edge_index))
                h = F.dropout(h, self.dropout, training=self.training)
                return self.convs[1](h, edge_index)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            a, b = TinyGCN(), TinyGCN()
            x = torch.randn(12, 3)
        nodes = torch.arange(12)
        edges = torch.stack([torch.cat([nodes, nodes.roll(1)]),
                             torch.cat([nodes.roll(1), nodes])])
        graph = Data(x=x, edge_index=edges, y=(x[:, 0] > 0).long(),
                     train_mask=nodes < 6, val_mask=(nodes >= 6) & (nodes < 9),
                     test_mask=nodes >= 9)
        changed = graph.clone()
        changed.y[~graph.train_mask] = 1 - graph.y[~graph.train_mask]
        paths = []
        for batch in (graph, changed):
            paths.append(connect(
                a, b, adapter=GCNAdapter(), train_data=[batch], calibration_data=[batch],
                loss_fn=lambda logits, data: F.cross_entropy(logits[data.train_mask],
                                                             data.y[data.train_mask]),
                select_observations=lambda site, activation, data: activation[data.train_mask],
                curve_steps=3, seed=19,
            ))
        for name in paths[0].control:
            torch.testing.assert_close(paths[0].control[name], paths[1].control[name], rtol=0, atol=0)
        states = [path.model_at(.5, repair=True, calibration_data=[batch]).state_dict()
                  for path, batch in zip(paths, (graph, changed))]
        for name in states[0]:
            torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
