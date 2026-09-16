import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loading.DatasetInfo import DatasetInfo
from models.fixed_spectral import FixedSpectralModel


class FixedSpectralModelTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.info = DatasetInfo(
            num_classes=2,
            num_features=2,
            name="path",
            class_weights=torch.tensor([1.0, 2.0]),
            N=6,
        )
        self.data = Data(
            x=torch.tensor(
                [[1.0, -0.5], [0.2, 0.7], [-0.3, 1.1],
                 [0.8, 0.4], [-0.6, 0.9], [0.1, -1.0]]
            ),
            # One direction only: the model must explicitly symmetrize it.
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
            ),
            y=torch.tensor([0, 1, 0, 1, 0, 1]),
            train_mask=torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool),
            val_mask=torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool),
            test_mask=torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool),
        )

    def test_filters_match_dense_spectral_responses_on_directed_path(self):
        model = FixedSpectralModel(self.info, K=10, dropout_rate=0.0)
        model.eval()
        with torch.no_grad():
            logits, inner_loss = model(self.data)

        adjacency = torch.zeros(6, 6, dtype=torch.float64)
        for source, target in self.data.edge_index.t().tolist():
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
        degree = adjacency.sum(dim=1)
        inv_sqrt_degree = degree.rsqrt()
        laplacian = torch.eye(6, dtype=torch.float64) - (
            inv_sqrt_degree[:, None]
            * adjacency
            * inv_sqrt_degree[None, :]
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        x = self.data.x.double()
        expected_low = eigenvectors @ (
            torch.exp(-10.0 * eigenvalues.square())[:, None]
            * (eigenvectors.T @ x)
        )
        expected_high = x - expected_low
        expected = torch.cat((expected_low, expected_high), dim=1).float()

        self.assertEqual(logits.shape, (6, 2))
        self.assertEqual(inner_loss.shape, torch.Size([]))
        self.assertEqual(inner_loss.item(), 0.0)
        torch.testing.assert_close(
            model._cached_features, expected, atol=6e-3, rtol=6e-3
        )

    def test_training_updates_only_predictor_and_keeps_fixed_cache(self):
        model = FixedSpectralModel(
            self.info,
            hidden_dim=8,
            learning_rate=0.05,
            dropout_rate=0.0,
            weight_decay=0.0,
            K=4,
        )
        model.log = lambda *args, **kwargs: None
        model.train()
        logits, _ = model(self.data)
        expected_loss = F.cross_entropy(
            logits[self.data.train_mask],
            self.data.y[self.data.train_mask],
            weight=self.info.class_weights,
        )
        actual_loss = model.training_step(self.data)
        torch.testing.assert_close(actual_loss, expected_loss)

        self.assertTrue(
            all(name.startswith("predictor.") for name, _ in model.named_parameters())
        )
        fixed_before = {
            name: value.clone()
            for name, value in model.named_buffers()
            if name in {"arnoldi_h", "filter_coefficients"}
        }
        cache_before = model._cached_features.clone()
        params_before = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }

        optimizer = model.configure_optimizers()["optimizer"]
        optimizer.zero_grad()
        actual_loss.backward()
        optimizer.step()

        self.assertTrue(
            any(
                not torch.equal(params_before[name], value)
                for name, value in model.named_parameters()
            )
        )
        for name, before in fixed_before.items():
            torch.testing.assert_close(dict(model.named_buffers())[name], before)
        torch.testing.assert_close(model._cached_features, cache_before)
        self.assertFalse(model._cached_features.requires_grad)
        self.assertNotIn("_cached_features", model.state_dict())

    def test_forward_accepts_inference_tensors_from_lightning_evaluation(self):
        model = FixedSpectralModel(self.info, K=2, dropout_rate=0.0)
        model.eval()
        with torch.inference_mode():
            inference_data = self.data.clone()
            logits, _ = model(inference_data)
        self.assertEqual(logits.shape, (6, 2))


if __name__ == "__main__":
    unittest.main()
