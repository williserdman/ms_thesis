import torch
import collections
import numpy as np
from .model_manager import ModelManager, DataBundle
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ModeConnectAnalyzer:
    ### PRIVATE
    def _get_od_shape(self, od: collections.OrderedDict) -> dict[str, int]:
        out = {}
        for el in od:
            out[el] = od[el].shape
        return out

    def _flatten_params(self, theta: collections.OrderedDict) -> torch.Tensor:
        return torch.cat([v.flatten() for v in theta.values()])

    def _unflatten_params(self, vals: torch.Tensor) -> collections.OrderedDict:
        start = 0
        out = collections.OrderedDict()
        for x in self.theta_shape:
            end = start + self.theta_shape[x].numel()
            out[x] = (
                vals[start:end].reshape(self.theta_shape[x]).to(dtype=torch.float32)
            )
            start = end
        return out

    def _batch_get_bezier(
        self,
        alphas: list,
        theta_flat: torch.Tensor,
    ) -> list:
        out = []
        for a in alphas:
            out.append(self.interpolate_bezier_flat(theta_flat, a))

        return out

    ### PUBLIC
    def __init__(
        self,
        theta_a: collections.OrderedDict,
        theta_b: collections.OrderedDict,
        mm: ModelManager,
        bezier_data: DataBundle,
    ):
        self.bezier_data = bezier_data
        self.mm = mm
        self.theta_a = theta_a
        self.theta_b = theta_b
        self.theta_a_flat = self._flatten_params(theta_a)
        self.theta_b_flat = self._flatten_params(theta_b)
        self.theta_shape = self._get_od_shape(theta_a)
        self.bezier_theta = None
        assert self.theta_shape == self._get_od_shape(
            theta_b
        ), "theta a and b must be the same shape"

    def all_close_test(self) -> bool:
        return torch.allclose(
            self._flatten_params(self.theta_a),
            self._flatten_params(self.theta_b),
        )

    def euclidian_distance(self) -> float:
        return torch.linalg.norm(
            self._flatten_params(self.theta_a) - self._flatten_params(self.theta_b)
        )

    def interpolate_linear(self, alpha: int) -> collections.OrderedDict:
        assert 0 <= alpha <= 1, " alpha must be in [0, 1] "

        wa = self.theta_a_flat
        wb = self.theta_b_flat

        new_w = (1 - alpha) * wa + alpha * wb

        return self._unflatten_params(new_w)

    def barrier(self, losses, L_a=None, L_b=None):
        losses = np.asarray(losses, dtype=float)
        if L_a is None:
            L_a = losses[0]
        if L_b is None:
            L_b = losses[-1]
        alphas = np.linspace(
            0.0, 1.0, len(losses)
        )  # finding the values of alpha by linearly interpolating between 0 and 1 for the ammount of losses we have in our array
        baseline = (
            1 - alphas
        ) * L_a + alphas * L_b  # how much of each loss we should get at each point
        return float(np.max(losses - baseline))

    def eval_linear_path(self, steps: int):
        original_state = self.mm.get_model_state()
        alphas = np.linspace(0, 1, steps)
        out = {}
        for a in alphas:
            a_state = self.interpolate_linear(a)
            self.mm.set_model_state(a_state)
            mets = self.mm.evaluate()
            out[a] = mets
        self.mm.set_model_state(original_state)
        return out

    def interpolate_bezier_flat(
        self,
        theta_flat: torch.Tensor,
        a,
    ):
        assert 0 <= a <= 1, " alpha must be in [0, 1] "

        return (
            (1 - a) ** 2 * self.theta_a_flat
            + 2 * a * (1 - a) * theta_flat
            + a**2 * self.theta_b_flat
        )

    def interpolate_bezier(self, theta, a):
        return self._unflatten_params(
            self.interpolate_bezier_flat(self._flatten_params(theta), a)
        )

    def smooth_max_barrier(
        self,
        losses: torch.Tensor,
        L_a: float,
        L_b: float,
        alphas: torch.Tensor,
        tau: float = 0.03,
    ) -> torch.Tensor:
        baseline = (
            1 - alphas
        ) * L_a + alphas * L_b  # expected loss from linear interpolation function
        baseline = torch.tensor(baseline)
        barriers = losses - baseline  # normalizing our loss
        return tau * torch.logsumexp(
            barriers / tau, dim=0
        )  # softmax over grid (in linear we just return max)
        # why this instead? i'm not sure # TODO

    def eval_bezier_path(
        self,
        theta: collections.OrderedDict,
        steps: int,
    ):
        alphas = np.linspace(0, 1, steps)
        out = {}
        original_state = self.mm.get_model_state()
        for a in alphas:
            a_state = self.interpolate_bezier(theta, a)
            self.mm.set_model_state(a_state)
            mets = self.mm.evaluate()
            out[a] = mets
        self.mm.set_model_state(original_state)
        return out

    def train_bezier(
        self,
        L_a: float,
        L_b: float,
        lr=0.01,
        wd=0,
        a_steps=10,
        epochs=200,
        verbose=False,
    ) -> torch.Tensor:
        theta_c = nn.Parameter(0.5 * (self.theta_a_flat + self.theta_b_flat))
        # original_state = model.state_dict()

        optimizer = torch.optim.Adam([theta_c], lr=lr, weight_decay=wd)
        alphas = np.linspace(0, 1, a_steps)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            thetas = self._batch_get_bezier(alphas, theta_c)
            losses = []
            for theta in thetas:

                s = self._unflatten_params(theta)
                # update_model_state(model, s)
                self.bezier_data.x = self.bezier_data.x.to(dtype=torch.float32)
                logits = self.mm.no_touch_get_logits(s)
                losses.append(
                    F.cross_entropy(
                        logits[self.bezier_data.train_mask],
                        self.bezier_data.y[self.bezier_data.train_mask],
                    )
                )
            losses = torch.stack(losses)

            objective = self.smooth_max_barrier(losses, L_a, L_b, alphas)
            objective.backward()
            nn.utils.clip_grad_norm_([theta_c], 10.0)
            optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == 1 or epoch == epochs):
                with torch.no_grad():
                    baseline = (1 - alphas) * L_a + alphas * L_b
                    hard_max = (losses - baseline).max().item()
                    where = alphas[(losses - baseline).argmax()].item()
                    print(
                        f"[{epoch:03d}] smooth={objective.item():.4f} hard={hard_max:.4f} @ α={where:.2f}"
                    )

        self.bezier_theta = self._unflatten_params(theta_c)
        return theta_c

    def linear_analysis_and_plot(self, steps, metrics: set = {"train", "val", "test"}):
        linear_interp = self.eval_linear_path(steps)
        losses = [linear_interp[a]["val_loss"] for a in linear_interp]
        self.barrier(losses)
        import matplotlib.pyplot as plt

        alphas = list(linear_interp.keys())
        val_losses = [linear_interp[a]["val_loss"] for a in alphas]
        train_losses = [linear_interp[a]["train_loss"] for a in alphas]
        test_losses = [linear_interp[a]["test_loss"] for a in alphas]

        val_accs = [linear_interp[a]["val_acc"] for a in alphas]
        train_accs = [linear_interp[a]["train_acc"] for a in alphas]
        test_accs = [linear_interp[a]["test_acc"] for a in alphas]

        ax2 = plt.gca()
        ax1 = ax2.twinx()
        ax1.set_ylim(bottom=-1.0, top=2.5)
        ax2.set_ylim(bottom=-0.1, top=1.1)

        ax1.scatter(alphas, train_losses, color="tab:green", s=20)
        ax1.scatter(alphas, test_losses, color="tab:red", s=20)
        ax2.scatter(alphas, train_accs, color="tab:blue", s=20)
        ax2.scatter(alphas, test_accs, color="tab:orange", s=20)

        if "train" in metrics:
            ax1.plot(alphas, train_losses, label="Train Loss", color="tab:green")
        if "val" in metrics:
            ax1.plot(alphas, val_losses, label="Val Loss", color="tab:orange")
        if "test" in metrics:
            ax1.plot(alphas, test_losses, label="Test Loss", color="tab:red")
        ax1.set_xlabel("Alpha")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="lower right")

        if "train" in metrics:
            ax2.plot(alphas, train_accs, label="Train Acc", color="tab:blue")
        if "val" in metrics:
            ax2.plot(alphas, val_accs, label="Val Acc", color="tab:red", linestyle="--")
        if "test" in metrics:
            ax2.plot(alphas, test_accs, label="Test Acc", color="tab:orange")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower left")

        plt.title("Losses and Accuracies along linear interpolation path")
        plt.show()

    def bezier_train_and_plot(
        self, steps, L_a, L_b, metrics: set = {"train", "val", "test"}
    ):
        theta_flat = self.train_bezier(L_a, L_b)
        theta = self._unflatten_params(theta_flat)
        b_interp = self.eval_bezier_path(theta, steps)
        losses = [b_interp[a]["val_loss"] for a in b_interp]
        self.barrier(losses)

        alphas = list(b_interp.keys())
        val_losses = [b_interp[a]["val_loss"] for a in alphas]
        train_losses = [b_interp[a]["train_loss"] for a in alphas]
        test_losses = [b_interp[a]["test_loss"] for a in alphas]

        val_accs = [b_interp[a]["val_acc"] for a in alphas]
        train_accs = [b_interp[a]["train_acc"] for a in alphas]
        test_accs = [b_interp[a]["test_acc"] for a in alphas]

        ax2 = plt.gca()
        ax1 = ax2.twinx()
        ax1.set_ylim(bottom=-1.0, top=2.5)
        ax2.set_ylim(bottom=-0.1, top=1.1)

        ax1.scatter(alphas, train_losses, color="tab:green", s=20)
        ax1.scatter(alphas, test_losses, color="tab:red", s=20)
        ax2.scatter(alphas, train_accs, color="tab:blue", s=20)
        ax2.scatter(alphas, test_accs, color="tab:orange", s=20)

        if "train" in metrics:
            ax1.plot(alphas, train_losses, label="Train Loss", color="tab:green")
        if "val" in metrics:
            ax1.plot(alphas, val_losses, label="Val Loss", color="tab:orange")
        if "test" in metrics:
            ax1.plot(alphas, test_losses, label="Test Loss", color="tab:red")
        ax1.set_xlabel("Alpha")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="lower right")

        if "train" in metrics:
            ax2.plot(alphas, train_accs, label="Train Acc", color="tab:blue")
        if "val" in metrics:
            ax2.plot(alphas, val_accs, label="Val Acc", color="tab:red", linestyle="--")
        if "test" in metrics:
            ax2.plot(alphas, test_accs, label="Test Acc", color="tab:orange")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower left")

        plt.title("Losses and Accuracies along bezier interpolation path")
        plt.show()

    def linear_vs_bezier_plot(self, steps: int):
        assert self.bezier_theta is not None, " please call .train_bezier() first"
        linear_interp = self.eval_linear_path(steps)
        bezier_interp = self.eval_bezier_path(self.bezier_theta, steps)

        ax2 = plt.gca()
        ax1 = ax2.twinx()
        ax1.set_ylim(bottom=-1.0, top=2.5)
        ax2.set_ylim(bottom=-0.1, top=1.1)

        # Linear interpolation
        linear_alphas = list(linear_interp.keys())
        linear_train_losses = [linear_interp[a]["train_loss"] for a in linear_alphas]
        linear_test_losses = [linear_interp[a]["test_loss"] for a in linear_alphas]
        linear_train_accs = [linear_interp[a]["train_acc"] for a in linear_alphas]
        linear_test_accs = [linear_interp[a]["test_acc"] for a in linear_alphas]
        linear_val_accs = [linear_interp[a]["val_acc"] for a in linear_alphas]
        linear_val_accs = [linear_interp[a]["val_loss"] for a in linear_alphas]

        alphas = list(bezier_interp.keys())
        train_losses = [bezier_interp[a]["train_loss"] for a in alphas]
        test_losses = [bezier_interp[a]["test_loss"] for a in alphas]
        val_losses = [bezier_interp[a]["val_loss"] for a in alphas]
        train_accs = [bezier_interp[a]["train_acc"] for a in alphas]
        val_accs = [bezier_interp[a]["val_acc"] for a in alphas]
        test_accs = [bezier_interp[a]["test_acc"] for a in alphas]

        ax1.plot(
            linear_alphas,
            linear_train_losses,
            label="Linear Train Loss",
            color="tab:green",
        )
        ax1.plot(
            linear_alphas, linear_test_losses, label="Linear Test Loss", color="tab:red"
        )
        ax2.plot(
            linear_alphas, linear_train_accs, label="Linear Train Acc", color="tab:blue"
        )
        ax2.plot(
            linear_alphas, linear_test_accs, label="Linear Test Acc", color="tab:orange"
        )

        ax1.scatter(linear_alphas, linear_train_losses, color="tab:green", s=20)
        ax1.scatter(linear_alphas, linear_test_losses, color="tab:red", s=20)
        ax2.scatter(linear_alphas, linear_train_accs, color="tab:blue", s=20)
        ax2.scatter(linear_alphas, linear_test_accs, color="tab:orange", s=20)

        # Bezier interpolation (dashed)
        ax1.plot(
            alphas,
            train_losses,
            label="Bézier Train Loss",
            color="tab:green",
            linestyle="--",
        )
        ax1.plot(
            alphas,
            test_losses,
            label="Bézier Test Loss",
            color="tab:red",
            linestyle="--",
        )
        ax2.plot(
            alphas,
            train_accs,
            label="Bézier Train Acc",
            color="tab:blue",
            linestyle="--",
        )
        ax2.plot(
            alphas,
            test_accs,
            label="Bézier Test Acc",
            color="tab:orange",
            linestyle="--",
        )

        ax1.set_xlabel("Alpha")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        ax1.legend(loc="lower right")
        ax2.legend(loc="lower left")
        plt.title("Linear vs Bezier Interpolation: Losses and Accuracies")
        plt.show()

    def get_bezier_theta(self) -> collections.OrderedDict:
        return self.bezier_theta
