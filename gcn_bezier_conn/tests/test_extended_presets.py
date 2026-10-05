"""Coverage for the reference preset's full thesis-loader dataset set."""

from __future__ import annotations

import unittest

from gcn_mc.presets import reference_profile


def _values(profile):
    model = profile["model"]
    training = profile["training"]
    return (
        model["hidden_channels"],
        model["depth"],
        model["dropout"],
        model["normalization"],
        model["residual"],
        model["pre_linear"],
        training["epochs"],
        training["lr"],
        training["weight_decay"],
        training["selection"],
    )


class ExtendedReferenceProfileTests(unittest.TestCase):
    def test_every_loader_dataset_resolves_for_every_architecture(self):
        datasets = (
            "Questions",
            "Cora",
            "Roman-empire",
            "computers",
            "photo",
            "Citeseer",
            "Pubmed",
            "squirrel",
            "chameleon",
            "actor",
            "texas",
            "cornell",
            "Amazon-ratings",
            "Minesweeper",
            "Tolokers",
        )
        for dataset in datasets:
            for architecture in ("gcn", "mlp", "graphsage", "gat"):
                with self.subTest(dataset=dataset, architecture=architecture):
                    profile = reference_profile(dataset, architecture)
                    self.assertEqual(set(profile), {"model", "training", "source"})
                    self.assertEqual(profile["model"]["heads"], 1)

    def test_source_dataset_aliases_match_loader_names(self):
        aliases = (
            ("amazon-computer", "computers"),
            ("amazon_computer", "computers"),
            ("amazon-photo", "photo"),
            ("amazon_photo", "photo"),
        )
        for alias, loader_name in aliases:
            with self.subTest(alias=alias):
                self.assertEqual(
                    reference_profile(alias, "gcn"),
                    reference_profile(loader_name, "gcn"),
                )

    def test_new_script_profiles_preserve_architecture_specific_commands(self):
        expected = {
            ("computers", "gcn"): (512, 3, 0.5, "layer", False, False, 1000, 0.001, 5e-5, "val_accuracy"),
            ("computers", "graphsage"): (64, 4, 0.3, "layer", False, False, 1000, 0.001, 5e-5, "val_accuracy"),
            ("computers", "gat"): (64, 2, 0.5, "layer", False, False, 1000, 0.001, 5e-5, "val_accuracy"),
            ("photo", "gcn"): (256, 6, 0.5, "layer", True, False, 1000, 0.001, 5e-5, "val_accuracy"),
            ("photo", "graphsage"): (64, 6, 0.2, "layer", True, False, 1000, 0.001, 5e-5, "val_accuracy"),
            ("photo", "gat"): (64, 3, 0.5, "layer", True, False, 1000, 0.001, 5e-5, "val_accuracy"),
            ("Citeseer", "gcn"): (512, 2, 0.5, "none", False, False, 500, 0.001, 0.01, "val_accuracy"),
            ("Citeseer", "graphsage"): (512, 3, 0.2, "none", False, False, 500, 0.001, 0.01, "val_accuracy"),
            ("Citeseer", "gat"): (256, 3, 0.5, "none", True, False, 500, 0.001, 0.01, "val_accuracy"),
            ("Pubmed", "gcn"): (256, 2, 0.7, "none", False, False, 500, 0.005, 5e-4, "val_accuracy"),
            ("Pubmed", "graphsage"): (512, 4, 0.7, "none", False, False, 500, 0.005, 5e-4, "val_accuracy"),
            ("Pubmed", "gat"): (512, 2, 0.5, "none", False, False, 500, 0.01, 5e-4, "val_accuracy"),
            ("Amazon-ratings", "gcn"): (512, 4, 0.5, "batch", True, False, 2500, 0.001, 0.0, "val_accuracy"),
            ("Amazon-ratings", "graphsage"): (512, 9, 0.5, "batch", True, False, 2500, 0.001, 0.0, "val_accuracy"),
            ("Amazon-ratings", "gat"): (512, 4, 0.5, "batch", True, False, 2500, 0.001, 0.0, "val_accuracy"),
            ("Minesweeper", "gcn"): (64, 12, 0.2, "batch", True, False, 2000, 0.01, 0.0, "val_accuracy"),
            ("Minesweeper", "graphsage"): (64, 15, 0.2, "batch", True, False, 2000, 0.01, 0.0, "val_accuracy"),
            ("Minesweeper", "gat"): (64, 15, 0.2, "batch", True, False, 2000, 0.01, 0.0, "val_accuracy"),
            ("Questions", "gcn"): (512, 10, 0.3, "none", True, True, 1500, 3e-5, 0.0, "val_accuracy"),
            ("Questions", "graphsage"): (512, 6, 0.2, "layer", False, True, 1500, 3e-5, 0.0, "val_accuracy"),
            ("Questions", "gat"): (512, 3, 0.2, "layer", True, True, 1500, 3e-5, 0.0, "val_accuracy"),
        }
        for (dataset, architecture), values in expected.items():
            with self.subTest(dataset=dataset, architecture=architecture):
                profile = reference_profile(dataset, architecture)
                self.assertEqual(_values(profile), values)
                self.assertFalse(profile["source"]["local_fallback"])

    def test_local_fallbacks_are_explicit_and_marked(self):
        expected = (64, 2, 0.5, "none", False, False, 200, 0.01, 5e-4, "val_loss")
        for dataset in ("actor", "texas", "cornell", "Tolokers"):
            for architecture in ("gcn", "mlp", "graphsage", "gat"):
                with self.subTest(dataset=dataset, architecture=architecture):
                    profile = reference_profile(dataset, architecture)
                    self.assertEqual(_values(profile), expected)
                    self.assertTrue(profile["source"]["local_fallback"])
                    self.assertEqual(profile["source"]["profile_origin"], "local_fallback")
                    if architecture == "mlp":
                        self.assertIn(
                            "local GCN fallback", profile["source"]["profile_adaptation"]
                        )

    def test_mlp_explicitly_inherits_each_gcn_profile(self):
        for dataset in (
            "Questions", "Cora", "Roman-empire", "computers", "photo",
            "Citeseer", "Pubmed", "squirrel", "chameleon", "actor",
            "texas", "cornell", "Amazon-ratings", "Minesweeper", "Tolokers",
        ):
            with self.subTest(dataset=dataset):
                mlp = reference_profile(dataset, "mlp")
                gcn = reference_profile(dataset, "gcn")
                self.assertEqual(mlp["model"], gcn["model"])
                self.assertEqual(mlp["training"], gcn["training"])
                self.assertIn("profile_adaptation", mlp["source"])

    def test_existing_profiles_keep_their_values(self):
        expected = {
            ("Cora", "gcn"): (512, 3, 0.7, "none", False, False, 500, 0.001, 5e-4, "val_accuracy"),
            ("Squirrel", "graphsage"): (256, 3, 0.7, "batch", True, False, 500, 0.01, 5e-4, "val_accuracy"),
            ("Roman-Empire", "gat"): (512, 10, 0.3, "batch", True, True, 2500, 0.001, 0.0, "val_accuracy"),
            ("Chameleon", "graphsage"): (256, 4, 0.7, "batch", True, False, 200, 0.01, 0.001, "val_accuracy"),
        }
        for key, values in expected.items():
            with self.subTest(dataset=key[0], architecture=key[1]):
                self.assertEqual(_values(reference_profile(*key)), values)


if __name__ == "__main__":
    unittest.main()
