"""Persistent-budget tests for the shared thesis Optuna hook."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import optuna


THESIS_SRC = Path(__file__).resolve().parents[2] / "src"
if str(THESIS_SRC) not in sys.path:
    sys.path.insert(0, str(THESIS_SRC))

from optuna_trainer import OptunaTrainer, suggest_common_parameters


optuna.logging.set_verbosity(optuna.logging.WARNING)


def _storage(path: Path) -> str:
    return f"sqlite:///{path}"


class OptunaHookTests(unittest.TestCase):
    def setUp(self):
        self.trainer = OptunaTrainer("cpu")

    def test_common_parameters_keep_original_ranges_and_skip_fixed_dimensions(self):
        def objective(trial):
            values = suggest_common_parameters(trial, fixed={"hidden_dim": 8})
            self.assertEqual(values["hidden_dim"], 8)
            return values["learning_rate"] + values["dropout_rate"]

        study = self.trainer.run_optimization("unused", 1, objective=objective, seed=2)
        trial = study.trials[0]

        self.assertEqual(set(trial.params), {"learning_rate", "dropout_rate"})
        self.assertEqual(trial.distributions["learning_rate"].low, 1e-4)
        self.assertEqual(trial.distributions["learning_rate"].high, 1e-2)
        self.assertEqual(trial.distributions["dropout_rate"].low, 0.0)
        self.assertEqual(trial.distributions["dropout_rate"].high, 0.7)

    def test_completed_budget_does_not_call_objective_again(self):
        with tempfile.TemporaryDirectory() as folder:
            storage = _storage(Path(folder) / "study.sqlite3")
            calls = []

            def objective(trial):
                calls.append(trial.number)
                return float(trial.number)

            self.trainer.run_optimization(
                "unused", 2, objective=objective, storage=storage, study_name="same"
            )
            self.trainer.run_optimization(
                "unused", 2, objective=objective, storage=storage, study_name="same"
            )

            self.assertEqual(calls, [0, 1])

    def test_larger_total_budget_runs_only_the_difference(self):
        with tempfile.TemporaryDirectory() as folder:
            storage = _storage(Path(folder) / "study.sqlite3")
            calls = []

            def objective(trial):
                calls.append(trial.number)
                return float(trial.number)

            self.trainer.run_optimization(
                "unused", 2, objective=objective, storage=storage, study_name="resume"
            )
            study = self.trainer.run_optimization(
                "unused", 5, objective=objective, storage=storage, study_name="resume"
            )

            self.assertEqual(calls, [0, 1, 2, 3, 4])
            self.assertEqual(len(study.trials), 5)

    def test_sqlite_study_reopens_with_the_same_best_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            storage = _storage(Path(folder) / "study.sqlite3")

            def objective(trial):
                value = trial.suggest_int("value", 1, 4)
                return float(value)

            first = self.trainer.run_optimization(
                "unused",
                3,
                objective=objective,
                storage=storage,
                study_name="reload",
                seed=17,
            )

            def unexpected(_trial):
                raise AssertionError("satisfied persistent budget ran another trial")

            reopened = OptunaTrainer("cpu").run_optimization(
                "unused",
                3,
                objective=unexpected,
                storage=storage,
                study_name="reload",
                seed=999,
            )

            self.assertEqual(reopened.best_trial.number, first.best_trial.number)
            self.assertEqual(reopened.best_value, first.best_value)
            self.assertEqual(reopened.best_params, first.best_params)

    def test_study_names_are_independent_in_one_database(self):
        with tempfile.TemporaryDirectory() as folder:
            storage = _storage(Path(folder) / "study.sqlite3")
            calls = []

            def objective(trial):
                calls.append(trial.study.study_name)
                return 0.0

            first = self.trainer.run_optimization(
                "unused", 1, objective=objective, storage=storage, study_name="first"
            )
            second = self.trainer.run_optimization(
                "unused", 1, objective=objective, storage=storage, study_name="second"
            )

            self.assertEqual(calls, ["first", "second"])
            self.assertEqual(first.study_name, "first")
            self.assertEqual(second.study_name, "second")

    def test_initial_parameters_are_enqueued_only_for_a_new_empty_study(self):
        with tempfile.TemporaryDirectory() as folder:
            storage = _storage(Path(folder) / "study.sqlite3")
            baseline = {
                "learning_rate": 0.002,
                "hidden_dim": 128,
                "dropout_rate": 0.25,
            }

            def objective(trial):
                values = suggest_common_parameters(trial)
                return values["learning_rate"]

            first = self.trainer.run_optimization(
                "unused",
                1,
                objective=objective,
                storage=storage,
                study_name="baseline",
                initial_params=baseline,
            )
            second = self.trainer.run_optimization(
                "unused",
                2,
                objective=objective,
                storage=storage,
                study_name="baseline",
                initial_params={
                    "learning_rate": 0.009,
                    "hidden_dim": 16,
                    "dropout_rate": 0.6,
                },
            )

            self.assertEqual(first.trials[0].params, baseline)
            self.assertNotEqual(second.trials[1].params, baseline)
            self.assertEqual(len(second.trials), 2)


if __name__ == "__main__":
    unittest.main()
