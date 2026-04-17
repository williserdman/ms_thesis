import collections
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModeConnectAnalyzer:
    def _get_od_shape(self, od: collections.OrderedDict) -> dict[str, torch.Size]:
        out: dict[str, torch.Size] = {}
        for key in od:
            out[key] = od[key].shape
        return out

    def _flatten_params(self, theta: collections.OrderedDict) -> torch.Tensor:
        return torch.cat([value.flatten() for value in theta.values()])

    def _unflatten_params(self, values: torch.Tensor) -> collections.OrderedDict:
        start = 0
        out = collections.OrderedDict()
        for key in self.theta_shape:
            end = start + self.theta_shape[key].numel()
            out[key] = values[start:end].reshape(self.theta_shape[key]).to(dtype=torch.float32)
            start = end
        return out

    def _batch_get_bezier(self, alphas: Iterable[float], theta_flat: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for alpha in alphas:
            out.append(self.interpolate_bezier_flat(theta_flat, alpha))
        return out

    def __init__(self, theta_a: collections.OrderedDict, theta_b: collections.OrderedDict, mm, bezier_data):
        self.bezier_data = bezier_data
        self.mm = mm
        self.theta_a = theta_a
        self.theta_b = theta_b
        self.theta_a_flat = self._flatten_params(theta_a)
        self.theta_b_flat = self._flatten_params(theta_b)
        self.theta_shape = self._get_od_shape(theta_a)
        self.bezier_theta = None
        assert self.theta_shape == self._get_od_shape(theta_b), "theta a and b must be the same shape"

    def interpolate_linear(self, alpha: float) -> collections.OrderedDict:
        assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
        new_w = (1 - alpha) * self.theta_a_flat + alpha * self.theta_b_flat
        return self._unflatten_params(new_w)

    def barrier(self, losses, L_a=None, L_b=None) -> float:
        loss_arr = np.asarray(losses, dtype=float)
        if L_a is None:
            L_a = loss_arr[0]
        if L_b is None:
            L_b = loss_arr[-1]
        alphas = np.linspace(0.0, 1.0, len(loss_arr))
        baseline = (1 - alphas) * L_a + alphas * L_b
        return float(np.max(loss_arr - baseline))

    def eval_linear_path(self, steps: int) -> dict[float, dict[str, float]]:
        original_state = self.mm.get_model_state()
        alphas = np.linspace(0, 1, steps)
        out: dict[float, dict[str, float]] = {}
        for alpha in alphas:
            a_state = self.interpolate_linear(float(alpha))
            self.mm.set_model_state(a_state)
            out[float(alpha)] = self.mm.evaluate()
        self.mm.set_model_state(original_state)
        return out

    def interpolate_bezier_flat(self, theta_flat: torch.Tensor, alpha: float) -> torch.Tensor:
        assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
        return (
            (1 - alpha) ** 2 * self.theta_a_flat
            + 2 * alpha * (1 - alpha) * theta_flat
            + alpha**2 * self.theta_b_flat
        )

    def interpolate_bezier(self, theta: collections.OrderedDict, alpha: float) -> collections.OrderedDict:
        return self._unflatten_params(self.interpolate_bezier_flat(self._flatten_params(theta), alpha))

    def smooth_max_barrier(
        self,
        losses: torch.Tensor,
        L_a: float,
        L_b: float,
        alphas: np.ndarray,
        tau: float = 0.03,
    ) -> torch.Tensor:
        alpha_t = torch.as_tensor(alphas, device=losses.device, dtype=losses.dtype)
        baseline = (1 - alpha_t) * L_a + alpha_t * L_b
        barriers = losses - baseline
        return tau * torch.logsumexp(barriers / tau, dim=0)

    def eval_bezier_path(self, theta: collections.OrderedDict, steps: int) -> dict[float, dict[str, float]]:
        alphas = np.linspace(0, 1, steps)
        out: dict[float, dict[str, float]] = {}
        original_state = self.mm.get_model_state()
        for alpha in alphas:
            a_state = self.interpolate_bezier(theta, float(alpha))
            self.mm.set_model_state(a_state)
            out[float(alpha)] = self.mm.evaluate()
        self.mm.set_model_state(original_state)
        return out

    def train_bezier(
        self,
        L_a: float,
        L_b: float,
        lr: float = 0.01,
        wd: float = 0,
        a_steps: int = 10,
        epochs: int = 200,
        verbose: bool = False,
    ) -> torch.Tensor:
        theta_c = nn.Parameter(0.5 * (self.theta_a_flat + self.theta_b_flat))
        optimizer = torch.optim.Adam([theta_c], lr=lr, weight_decay=wd)
        alphas = np.linspace(0, 1, a_steps)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            thetas = self._batch_get_bezier(alphas, theta_c)
            losses = []
            for theta in thetas:
                state = self._unflatten_params(theta)
                logits = self.mm.no_touch_get_logits(state)
                losses.append(
                    F.cross_entropy(
                        logits[self.bezier_data.train_mask],
                        self.bezier_data.y[self.bezier_data.train_mask],
                    )
                )
            loss_stack = torch.stack(losses)
            objective = self.smooth_max_barrier(loss_stack, L_a, L_b, alphas)
            objective.backward()
            nn.utils.clip_grad_norm_([theta_c], 10.0)
            optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == 1 or epoch == epochs):
                with torch.no_grad():
                    alpha_t = torch.as_tensor(alphas, device=loss_stack.device, dtype=loss_stack.dtype)
                    baseline = (1 - alpha_t) * L_a + alpha_t * L_b
                    hard_barriers = loss_stack - baseline
                    hard_max = float(hard_barriers.max().item())
                    idx = int(hard_barriers.argmax().item())
                    where = float(alpha_t[idx].item())
                    print(f"[{epoch:03d}] smooth={objective.item():.4f} hard={hard_max:.4f} @ alpha={where:.2f}")

        self.bezier_theta = self._unflatten_params(theta_c.detach())
        return theta_c.detach()

    def get_bezier_theta(self) -> collections.OrderedDict:
        return self.bezier_theta
