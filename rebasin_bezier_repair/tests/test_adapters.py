import subprocess
import sys
import unittest

import torch
from git_re_basin import apply_permutation
from torch import nn

from rebasin_bezier_repair import MLPAdapter
from rebasin_bezier_repair.adapters.base import correct_affine_rows
from tests.support import copy_state, mlp_case


class AdapterTests(unittest.TestCase):
    def test_mlp_permutation_preserves_logits(self):
        model, batches = mlp_case()
        model.eval()
        state = copy_state(model)
        before = copy_state(model)
        adapter = MLPAdapter()
        spec = adapter.permutation_spec(model, state)
        self.assertEqual(set(spec), set(state))
        groups = {group: state[key].shape[axis]
                  for key, axes in spec.items()
                  for axis, group in enumerate(axes) if group is not None}
        permutation = {group: torch.arange(width - 1, -1, -1)
                       for group, width in groups.items()}
        aligned = apply_permutation(spec, permutation, state)
        torch.testing.assert_close(
            adapter.forward(model, state, batches[0]),
            adapter.forward(model, aligned, batches[0]), rtol=1e-4, atol=1e-5,
        )
        self.assertEqual([s.module_path for s in adapter.repair_sites(model)], ["0", "2"])
        for key in state:
            self.assertTrue(torch.equal(before[key], model.state_dict()[key]))

    def test_mlp_rejects_unsupported_structure(self):
        unsupported = (
            nn.Sequential(nn.Linear(4, 8, bias=False), nn.ReLU(), nn.Linear(8, 3)),
            nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 3)),
        )
        for model in unsupported:
            with self.subTest(model=model), self.assertRaises(ValueError):
                MLPAdapter().permutation_spec(model, model.state_dict())

    def test_affine_correction_copies_state(self):
        state = {"weight": torch.arange(6.).reshape(2, 3),
                 "bias": torch.tensor([1., 2.]), "fixed": torch.tensor([7])}
        before = {k: v.clone() for k, v in state.items()}
        scale, shift = torch.tensor([2., 3.]), torch.tensor([4., 5.])
        corrected = correct_affine_rows(state, "weight", "bias", scale, shift)
        torch.testing.assert_close(corrected["weight"], state["weight"] * scale[:, None])
        torch.testing.assert_close(corrected["bias"], state["bias"] * scale + shift)
        for key in state:
            self.assertTrue(torch.equal(state[key], before[key]))
            self.assertNotEqual(state[key].data_ptr(), corrected[key].data_ptr())

    def test_core_import_without_pyg(self):
        source = '''
import importlib.abc
import sys
class BlockPyG(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch_geometric" or fullname.startswith("torch_geometric."):
            raise ModuleNotFoundError("PyG unavailable", name=fullname)
sys.meta_path.insert(0, BlockPyG())
import rebasin_bezier_repair
from rebasin_bezier_repair import MLPAdapter
MLPAdapter()
'''
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
