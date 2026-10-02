import unittest

import torch
from torch import nn
from torch.nn import functional as F

from rebasin_bezier_repair import MLPAdapter
from rebasin_bezier_repair.paths import bezier_state, fit_curve
from tests.support import copy_state, mlp_case


class PathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model, self.batches = mlp_case(0)
        other, _ = mlp_case(1)
        self.a, self.b = copy_state(self.model), copy_state(other)
        self.names = tuple(dict(self.model.named_parameters()))
        self.control = {k: nn.Parameter((self.a[k] + self.b[k]) / 2) for k in self.names}

    def test_endpoints_and_control_gradient(self):
        for t, endpoint in ((0, self.a), (1, self.b)):
            result = bezier_state(self.a, self.b, self.control, t, parameter_names=self.names)
            for key in endpoint:
                torch.testing.assert_close(result[key], endpoint[key], rtol=0, atol=0)
        midpoint = bezier_state(self.a, self.b, self.control, 0.5, parameter_names=self.names)
        key = "0.weight"
        midpoint[key].sum().backward()
        torch.testing.assert_close(self.control[key].grad, torch.full_like(self.control[key], 0.5))

    def test_fitting_changes_only_control(self):
        self.model.train()
        self.model[1].eval()
        modes = [module.training for module in self.model.modules()]
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        original = copy_state(self.model)
        a_before, b_before = ({k: v.clone() for k, v in state.items()}
                              for state in (self.a, self.b))
        result = fit_curve(self.model, MLPAdapter(), self.a, self.b, self.batches,
                           lambda logits, batch: F.cross_entropy(logits, batch[1]), steps=3)
        self.assertTrue(any(not torch.equal(result[k], self.control[k]) for k in self.names))
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        for key in self.a:
            self.assertTrue(torch.equal(self.a[key], a_before[key]))
            self.assertTrue(torch.equal(self.b[key], b_before[key]))
            self.assertTrue(torch.equal(self.model.state_dict()[key], original[key]))
        for parameter in self.model.parameters():
            self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter)))

    def test_fixed_buffers(self):
        self.a["fixed"] = torch.tensor([7])
        self.b["fixed"] = torch.tensor([7])
        result = bezier_state(self.a, self.b, self.control, 0.5, parameter_names=self.names)
        self.assertTrue(torch.equal(result["fixed"], torch.tensor([7])))
        self.assertNotEqual(result["fixed"].data_ptr(), self.a["fixed"].data_ptr())
        self.b["fixed"] = torch.tensor([8])
        with self.assertRaisesRegex(ValueError, "fixed"):
            bezier_state(self.a, self.b, self.control, 0.5, parameter_names=self.names)

    def test_invalid_curve_inputs(self):
        for t in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(t=t), self.assertRaises(ValueError):
                bezier_state(self.a, self.b, self.control, t, parameter_names=self.names)
        loss = lambda logits, batch: F.cross_entropy(logits, batch[1])
        for source in ([], iter(self.batches)):
            with self.subTest(source=type(source)), self.assertRaises((ValueError, TypeError)):
                fit_curve(self.model, MLPAdapter(), self.a, self.b, source, loss, steps=1)
        for options in ({"steps": 0}, {"lr": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit_curve(self.model, MLPAdapter(), self.a, self.b, self.batches, loss, **options)
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_curve(self.model, MLPAdapter(), self.a, self.b, self.batches,
                      lambda logits, batch: logits.sum() * float("nan"), steps=1)


if __name__ == "__main__":
    unittest.main()
