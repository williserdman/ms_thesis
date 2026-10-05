"""idea18_filtercurve: a trainable Bezier curve of spectral filters with
per-node arc-length selection.

Headline question (research brief sec. 4, the spectral<->mode-connectivity
bridge): the LinearSpectralGNN exposes a single shared (K+1)-dim coefficient
vector gamma -- one filter response g(lambda) for the whole graph. But nodes are
not interchangeable: a high-degree hub and a leaf may want different amounts of
low/high-pass mixing. Idea-18 replaces the single point gamma with a *curve* of
filters -- a quadratic Bezier in coefficient space defined by two learnable
endpoints (g0, g1) and a learnable control (gc) -- and lets every node pick its
own arc-length position t_i in [0,1] from its features. Node i then filters with
its own coefficient vector

    gamma_i = bezier(g0, g1, gc, t_i) = (1-t_i)^2 g0 + 2(1-t_i)t_i gc + t_i^2 g1.

This is a strict generalization of the baseline: if t_i collapses to a constant
the curve degenerates to a single shared filter. The two questions we answer per
dataset are:

  (1) does the per-node curve beat a single shared filter on TEST accuracy?
  (2) do the learned positions t_i organize by node structure -- here probed
      cheaply by Pearson correlation of t_i with node degree (a stand-in for
      local homophily / over-smoothing pressure)?

Design / reuse:
  - FilterCurveGNN SUBCLASSES models.linear_spectral.LinearSpectralGNN and only
    adds the three coefficient vectors + a per-node position head, then overrides
    _propagate (per-node coefficients) and forward (compute t_i / gamma_i first).
    training_step / validation_step / test_step / configure_optimizers / _loss
    are inherited unchanged -- they only call forward() + masks.
  - The fixed-gamma baseline is the unmodified LinearSpectralGNN trained via
    common.train_model. Because train_model hard-codes LinearSpectralGNN, the
    FilterCurveGNN is trained with a small local copy of that train loop (same
    Trainer config, same EarlyStopping, same seeding via common.set_seed). No
    barrier / vectorization / dataset code is re-implemented.
  - per-node bezier uses mode_connectivity.paths.bezier_interp (pure arithmetic,
    so it broadcasts over a (N,1) torch t against (K+1,) coefficient vectors).

Runtime guard: per dataset we train exactly ONE baseline and ONE FilterCurve
model per seed, over args.seeds (default 3). No sweep. The extra cost over the
baseline is one Linear(F->1) head and a per-node weighting of the K+1 basis
features -- negligible vs the MLP. With 4 datasets x 3 seeds x 2 models and the
shared 200-epoch / patience-50 budget this stays well under 1hr on 1 GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import common
from mode_connectivity import paths
from models.linear_spectral import LinearSpectralGNN


EPS = 1e-8


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class FilterCurveGNN(LinearSpectralGNN):
    """LinearSpectralGNN with a per-node Bezier curve of spectral filters.

    Replaces the single shared coefficient vector gamma with three learnable
    (K+1,) vectors -- curve endpoints g0, g1 and control gc -- plus a per-node
    position head t_i = sigmoid(Linear(node_features)) in [0,1]. Node i filters
    with gamma_i = bezier(g0, g1, gc, t_i). The inherited gamma Parameter is kept
    (so the parent __init__ contract is untouched) but is NOT used by this
    subclass's _propagate; g0 is initialized from it so the curve starts at the
    baseline init.
    """

    def __init__(self, ds_info, **kwargs):
        super().__init__(ds_info, **kwargs)
        # Seed the curve from the parent's gamma init so a collapsed curve
        # (t_i constant) reproduces a baseline-like filter; all three start equal.
        g_init = self.gamma.detach().clone()
        self.g0 = nn.Parameter(g_init.clone())
        self.g1 = nn.Parameter(g_init.clone())
        self.gc = nn.Parameter(g_init.clone())
        # per-node arc-length position head: features -> scalar logit -> sigmoid.
        self.t_head = nn.Linear(self.num_features, 1)
        nn.init.zeros_(self.t_head.weight)
        # bias 0 -> sigmoid(0) = 0.5: every node starts at the curve midpoint.
        nn.init.zeros_(self.t_head.bias)
        self._last_t = None  # cache of the most recent per-node positions (detached use)

    # ---- per-node positions / coefficients ----
    def node_positions(self, batch) -> torch.Tensor:
        """t_i in [0,1] per node, shape (N, 1)."""
        return torch.sigmoid(self.t_head(batch.x))

    def node_gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Per-node coefficient matrix gamma_i, shape (N, K+1).

        t is (N,1); g0/g1/gc are (K+1,). bezier_interp is pure arithmetic so the
        broadcast (N,1) x (K+1,) -> (N, K+1) is exactly the per-node curve point.
        """
        return paths.bezier_interp(self.g0, self.g1, self.gc, t)

    # ---- propagation with per-node coefficients ----
    def _propagate(self, adj: torch.Tensor, H0: torch.Tensor) -> torch.Tensor:
        """Z[i] = sum_k gamma_i[k] * phi_k(op)[i] using the cached per-node gamma.

        Computes each per-order basis feature H_k (shape (N, C)) exactly as the
        parent does, but weights node i's H_k by its own gamma_i[:, k:k+1] before
        summing -- so every node applies its own filter. self._last_t / the
        per-node gamma matrix are set by forward() right before this is called.
        """
        gamma_i = self._cur_gamma_i  # (N, K+1), set in forward()

        def weight(Hk, k):
            return gamma_i[:, k:k + 1] * Hk

        if self.basis == "mono":
            Z = weight(H0, 0)
            Hk = H0
            for k in range(1, self.K + 1):
                Hk = self._apply_op(adj, Hk)
                Z = Z + weight(Hk, k)
            return Z
        if self.basis == "cheb":
            T0 = H0
            Z = weight(T0, 0)
            if self.K >= 1:
                T1 = self._apply_op(adj, H0)
                Z = Z + weight(T1, 1)
                Tprev, Tcur = T0, T1
                for k in range(2, self.K + 1):
                    Tnext = 2.0 * self._apply_op(adj, Tcur) - Tprev
                    Z = Z + weight(Tnext, k)
                    Tprev, Tcur = Tcur, Tnext
            return Z
        raise ValueError(f"unknown basis {self.basis!r}")

    def forward(self, batch):
        adj = self._get_op(batch)
        # per-node positions are read from the RAW node features (pre-dropout) so
        # the arc-length selection is a stable property of the node, not noised.
        t = self.node_positions(batch)            # (N, 1)
        self._cur_gamma_i = self.node_gamma(t)    # (N, K+1)
        self._last_t = t.detach()

        h = F.dropout(batch.x, p=self.dropout_rate, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        H0 = self.lin2(h)
        logits = self._propagate(adj, H0)
        inner_loss = logits.new_zeros(())
        return logits, inner_loss


# ----------------------------------------------------------------------------
# training (mirrors common.train_model but instantiates FilterCurveGNN)
# ----------------------------------------------------------------------------
def _train_filtercurve(prepared, model_kwargs, max_epochs, patience, gpus, seed, verbose=False):
    """Train a FilterCurveGNN with the exact Trainer config common.train_model uses."""
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    common.set_seed(seed)
    accelerator, devices, _ = common.resolve_device(gpus)
    model = FilterCurveGNN(prepared.ds_info, **model_kwargs)
    callbacks = [EarlyStopping(monitor="val_loss", patience=patience, mode="min")]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_progress_bar=verbose,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )
    trainer.fit(
        model,
        train_dataloaders=prepared.datamodule.train_dataloader(),
        val_dataloaders=prepared.datamodule.val_dataloader(),
    )
    return model


# ----------------------------------------------------------------------------
# stats helpers (cheap, numpy-free where possible)
# ----------------------------------------------------------------------------
def _node_degrees(batch) -> torch.Tensor:
    """In-degree per node from edge_index (no self-loops added), shape (N,)."""
    n = batch.x.size(0)
    row = batch.edge_index[0]
    deg = torch.zeros(n, device=batch.x.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=batch.x.device))
    return deg


def _quantiles(t: torch.Tensor, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    flat = t.flatten().float()
    return {f"q{int(q * 100):02d}": float(torch.quantile(flat, q).item()) for q in qs}


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors (0.0 if a side is constant)."""
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < EPS:
        return 0.0
    return float((a @ b).item() / denom)


@torch.no_grad()
def _t_stats(model, batch) -> dict:
    """Distribution of learned t_i and its correlation with node degree."""
    was_training = model.training
    model.eval()
    t = model.node_positions(batch).flatten()  # (N,)
    deg = _node_degrees(batch)
    stats = {
        "t_mean": float(t.mean().item()),
        "t_std": float(t.std(unbiased=False).item()),
        "t_quantiles": _quantiles(t),
        "t_corr_degree": _pearson(t, deg),
        "t_corr_log_degree": _pearson(t, torch.log(deg + 1.0)),
        "t_spread": float((t.max() - t.min()).item()),
        "n_nodes": int(t.numel()),
    }
    if was_training:
        model.train()
    return stats


# ----------------------------------------------------------------------------
# per-dataset run
# ----------------------------------------------------------------------------
def _run_dataset(args, name):
    prepared = common.prepare_dataset(name)
    _, _, tdev = common.resolve_device(args.gpus)
    batch = common.move_batch(prepared.batch, tdev)

    model_kwargs = dict(
        hidden_dim=args.hidden_dim,
        K=args.K,
        basis=args.basis,
        domain=args.domain,
        learning_rate=args.lr,
        dropout_rate=args.dropout,
    )

    base_test, curve_test = [], []
    t_stats_per_seed = []
    for seed in args.seeds:
        # baseline: unmodified LinearSpectralGNN via the shared train loop.
        base = common.train_model(
            prepared, model_kwargs, max_epochs=args.epochs, patience=args.patience,
            gpus=args.gpus, seed=seed, verbose=False,
        ).to(tdev)
        base.eval()
        _, base_acc = common.eval_loss_acc(base, batch, mask_name="test_mask")

        # FilterCurve model.
        curve = _train_filtercurve(
            prepared, model_kwargs, max_epochs=args.epochs, patience=args.patience,
            gpus=args.gpus, seed=seed, verbose=False,
        ).to(tdev)
        curve.eval()
        _, curve_acc = common.eval_loss_acc(curve, batch, mask_name="test_mask")

        base_test.append(float(base_acc))
        curve_test.append(float(curve_acc))
        t_stats_per_seed.append(_t_stats(curve, batch))

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else None

    def _std(xs):
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return float((sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5)

    # aggregate t-distribution stats across seeds (mean of per-seed summaries).
    def _avg_t(key):
        return _mean([s[key] for s in t_stats_per_seed])

    agg_quantiles = {
        qk: _mean([s["t_quantiles"][qk] for s in t_stats_per_seed])
        for qk in t_stats_per_seed[0]["t_quantiles"]
    }

    mean_base, mean_curve = _mean(base_test), _mean(curve_test)
    return {
        "dataset": name,
        "n_seeds": len(args.seeds),
        "test_acc_baseline": {"mean": mean_base, "std": _std(base_test), "per_seed": base_test},
        "test_acc_filtercurve": {"mean": mean_curve, "std": _std(curve_test), "per_seed": curve_test},
        "delta_curve_minus_baseline": float(mean_curve - mean_base),
        "curve_beats_baseline": bool(mean_curve > mean_base + EPS),
        "t_distribution": {
            "t_mean": _avg_t("t_mean"),
            "t_std": _avg_t("t_std"),
            "t_spread": _avg_t("t_spread"),
            "t_quantiles": agg_quantiles,
        },
        "t_corr_degree": _avg_t("t_corr_degree"),
        "t_corr_log_degree": _avg_t("t_corr_log_degree"),
        "t_stats_per_seed": t_stats_per_seed,
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    parser = common.base_argparser(
        "idea18_filtercurve: a trainable Bezier curve of spectral filters with "
        "per-node arc-length selection (FilterCurveGNN) vs a fixed-gamma "
        "LinearSpectralGNN baseline. Reports test acc of both and whether the "
        "learned per-node positions t_i organize by node degree."
    )
    args = parser.parse_args()
    args = common.apply_smoke(args)

    results = []
    for name in args.datasets:
        results.append(_run_dataset(args, name))

    payload = {
        "experiment": "idea18_filtercurve",
        "config": {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "K": args.K,
            "basis": args.basis,
            "domain": args.domain,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "patience": args.patience,
            "smoke": bool(args.smoke),
            "model_note": (
                "FilterCurveGNN subclasses LinearSpectralGNN; replaces shared gamma "
                "with curve endpoints g0,g1 + control gc and a per-node position "
                "head t_i=sigmoid(Linear(x)); gamma_i=bezier(g0,g1,gc,t_i). t_i "
                "starts at 0.5 (head zero-init); g0=g1=gc=parent gamma init."
            ),
        },
        "results": results,
    }

    common.write_results(args, "idea18_filtercurve", payload)

    # human summary
    print("\n=== idea18_filtercurve: per-node Bezier filter curve vs shared filter ===")
    print("headline: does per-node curve position beat a single shared filter, "
          "and do t_i organize by node degree?")
    print(f"  {'dataset':>14s} {'base_acc':>9s} {'curve_acc':>9s} {'delta':>8s} "
          f"{'win?':>5s} {'t_mean':>7s} {'t_std':>7s} {'corr(deg)':>10s}")
    for r in results:
        td = r["t_distribution"]
        print(
            f"  {r['dataset']:>14s} "
            f"{r['test_acc_baseline']['mean']:>9.4f} "
            f"{r['test_acc_filtercurve']['mean']:>9.4f} "
            f"{r['delta_curve_minus_baseline']:>+8.4f} "
            f"{('yes' if r['curve_beats_baseline'] else 'no'):>5s} "
            f"{td['t_mean']:>7.3f} {td['t_std']:>7.3f} "
            f"{r['t_corr_degree']:>+10.3f}"
        )


if __name__ == "__main__":
    main()
