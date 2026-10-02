import unittest

import torch
from torch import nn

from rebasin_bezier_repair import MLPAdapter
from rebasin_bezier_repair.repair import collect_moments, repair_state
from tests.support import copy_state


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def moment_case(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2),
                              nn.ReLU(), nn.Linear(2, 1))
        a = copy_state(model)
        a["0.weight"] = torch.tensor([[1., .2], [.4, 1.]])
        a["0.bias"] = torch.tensor([3., 4.])
        a["2.weight"] = torch.tensor([[.5, .1], [.2, .6]])
        a["2.bias"] = torch.tensor([2., 2.])
        b = {k: v.clone() for k, v in a.items()}
        b["0.weight"] = torch.tensor([[.2, 1.], [1., .3]])
        b["0.bias"] = torch.tensor([4., 3.])
        b["2.weight"] = torch.tensor([[.1, .7], [.8, .2]])
        b["2.bias"] = torch.tensor([1., 3.])
        x = torch.tensor([[-1., 0.], [0., 1.], [1., -1.], [2., 1.], [-.5, 2.], [.2, .6]])
        return model, a, b, [x[:3], x[3:]]

    def test_pooled_population_moments(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(2))
            model[0].bias.zero_()
        values = torch.tensor([[1., 2.], [3., 6.], [8., 9.]])
        result = collect_moments(model, MLPAdapter(), copy_state(model),
                                 [values[:2], values[2:]])["0"]
        torch.testing.assert_close(result.mean, values.mean(0))
        torch.testing.assert_close(result.std, values.std(0, unbiased=False))

    def test_sequential_target_moments(self):
        model, a, b, data = self.moment_case()
        adapter = MLPAdapter()
        stats = tuple(collect_moments(model, adapter, state, data) for state in (a, b))
        merged = {k: (a[k] + b[k]) / 2 for k in a}
        before = {k: v.clone() for k, v in merged.items()}
        repaired = repair_state(model, adapter, merged, .5, stats, data)
        measured = collect_moments(model, adapter, repaired, data)
        for name in measured:
            self.assertTrue((measured[name].std > 1e-6).all())
            torch.testing.assert_close(measured[name].mean,
                                       (stats[0][name].mean + stats[1][name].mean) / 2,
                                       rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(measured[name].std,
                                       (stats[0][name].std + stats[1][name].std) / 2,
                                       rtol=1e-4, atol=1e-5)
        self.assertTrue(torch.equal(repaired["4.weight"], before["4.weight"]))
        for key in merged:
            self.assertTrue(torch.equal(merged[key], before[key]))
        for t in (0., 1.):
            endpoint = repair_state(model, adapter, merged, t, stats, None)
            for key in merged:
                self.assertTrue(torch.equal(endpoint[key], merged[key]))

    def test_constant_channel_is_finite(self):
        model, a, b, data = self.moment_case()
        adapter = MLPAdapter()
        stats = tuple(collect_moments(model, adapter, state, data) for state in (a, b))
        candidate = {k: v.clone() for k, v in a.items()}
        candidate["0.weight"].zero_()
        result = repair_state(model, adapter, candidate, .5, stats, data)
        self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))
        self.assertTrue(torch.equal(result["0.weight"], torch.zeros_like(result["0.weight"])))

    def test_calibration_sources_and_cleanup(self):
        model, a, _, data = self.moment_case()
        adapter = MLPAdapter()
        for source in ([], iter(data), [data[0][:1]]):
            with self.subTest(source=type(source)), self.assertRaises((ValueError, TypeError)):
                collect_moments(model, adapter, a, source)
        model.train()
        model[1].eval()
        modes = [m.training for m in model.modules()]
        hooks = [len(m._forward_hooks) for m in model.modules()]

        def fail_selector(site, activation, batch):
            raise RuntimeError("selector failed")

        with self.assertRaisesRegex(RuntimeError, "selector failed"):
            collect_moments(model, adapter, a, data, selector=fail_selector)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(hooks, [len(m._forward_hooks) for m in model.modules()])


if __name__ == "__main__":
    unittest.main()
