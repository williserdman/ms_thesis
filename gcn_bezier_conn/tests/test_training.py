"""Focused checks that held-out labels cannot affect training decisions."""

import unittest

import torch
from torch_geometric.data import Data

from gcn_mc.experiment import Config, fit_curve, train_endpoint
from gcn_mc.model import GCN
from gcn_mc.paths import clone_parameters


class TrainingIsolationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.graph = Data(
            x=torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.], [2., 0.], [0., 2.]]),
            edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]]),
            y=torch.tensor([0, 1, 0, 1, 0, 1]),
            train_mask=torch.tensor([True, True, False, False, False, False]),
            val_mask=torch.tensor([False, False, True, True, False, False]),
            test_mask=torch.tensor([False, False, False, False, True, True]),
        )
        self.model_config = dict(in_channels=2, hidden_channels=4, out_channels=2, depth=2, dropout=0.0)
        self.config = Config(epochs=3, curve_epochs=3, dropout=0.0)

    def test_test_labels_do_not_select_or_train_endpoint(self):
        other = self.graph.clone()
        other.y[other.test_mask] = 1 - other.y[other.test_mask]
        first, first_info = train_endpoint(self.graph, self.model_config, self.config, 5)
        second, second_info = train_endpoint(other, self.model_config, self.config, 5)
        self.assertEqual(first_info["selected_epoch"], second_info["selected_epoch"])
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    def test_control_fitting_ignores_all_held_out_labels(self):
        torch.manual_seed(7)
        first = GCN(**self.model_config)
        second = GCN(**self.model_config)
        a, b = clone_parameters(first), clone_parameters(second)
        other = self.graph.clone()
        other.y[~other.train_mask] = 1 - other.y[~other.train_mask]
        control, history = fit_curve(first, self.graph, a, b, self.config, 11)
        other_control, other_history = fit_curve(second, other, a, b, self.config, 11)
        self.assertEqual(history, other_history)
        for key, value in control.items():
            torch.testing.assert_close(value, other_control[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
