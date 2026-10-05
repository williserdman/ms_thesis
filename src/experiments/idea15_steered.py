"""idea15_steered: spectrally-steered Bezier bends in coefficient space.

Headline question: a structure-blind Bezier bend uses the straight-line midpoint
as its control point. Can a tiny *steering net* that reads only a cheap summary of
the graph spectrum produce a better coefficient-space control point -- one that
lowers the mode-connectivity barrier between two trained solutions below the
midpoint bend -- and does that steering *transfer* to a held-out graph?

The bridge framing (research brief, sec. 4): the explicit linear coefficient axis
gamma is a tiny, transferable handle on a GNN solution. A bend in gamma-space is a
morph of the filter response g(lambda). If the *right* bend is predictable from the
graph spectrum alone, then mode connectivity in coefficient space is itself a
spectral property -- steerable without ever touching the data again.

Setup (per dataset):
  - train 2 endpoint LinearSpectralGNN models (seeds[0], seeds[1]);
  - extract their (K+1)-dim coefficient vectors gamma_a, gamma_b;
  - compute three coefficient-space barriers (names = COEFF_NAMES = ['gamma'],
    so ONLY gamma is interpolated; the MLP is held at endpoint a along the path),
    all on the validation mask via common.barrier_along_path:
       (a) linear          -- straight gamma path (control=None);
       (b) midpoint-Bezier  -- control = 0.5*(gamma_a + gamma_b) (structure-blind);
       (c) steered-Bezier   -- control = midpoint + SteerNet(spectrum_summary).

SteerNet (a tiny torch nn.Module): maps a fixed-length spectrum summary of the
graph Laplacian (cheap moments + quantiles + count, see _spectrum_summary) to a
(K+1) control *offset* added to the midpoint. It is kept sub-100 parameters (the
idea spec's constraint): a single Linear(F_summary -> K+1) with a tanh-bounded,
scaled output. With F_summary ~= 7 and K+1 small this is ~ (7+1)*(K+1) params.

PoC fit (documented limitation): for part (c) we FIT SteerNet by directly
minimizing the *sampled coefficient-path loss* of THIS dataset's endpoint pair
(a differentiable inner loop that injects the Bezier-interpolated gamma into the
trained model and backprops the mean path cross-entropy w.r.t. SteerNet weights).
This is a proof-of-concept, per-pair fit -- it shows steering *can* beat the
midpoint, not that one SteerNet generalizes. Full cross-dataset transfer is future
work; we include a first probe of it below.

Held-out transfer probe: we additionally fit ONE SteerNet on the path losses of
datasets[0..n-2] (their endpoint pairs, summaries computed per dataset), then
APPLY it -- frozen, no further fitting -- to datasets[-1], and report the
resulting steered barrier vs that dataset's own midpoint / linear / per-pair-fit
barriers. With >=2 datasets this answers "does steering transfer?" for one held-out
graph (a single point, not a generalization claim).

Everything that can be reused is reused: training (common.train_model),
vectorization (common.get_vector/set_vector on COEFF_NAMES), barrier math
(common.barrier_along_path), the spectrum (common.laplacian_eigs). The only new
torch code is SteerNet and the differentiable path-loss fit; barrier *reporting*
always goes back through common.barrier_along_path for an apples-to-apples number.

Runtime guard: exactly 2 endpoint models per dataset; the SteerNet fit is a short
fixed-step Adam loop (FIT_STEPS) over the already-trained model (no GNN retraining).
Both are tiny relative to training; nothing else is swept.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from experiments import common
from mode_connectivity import paths


EPS = 1e-8
# number of quantiles in the spectrum summary (q=0..1). Summary length = NQ + 4.
QUANTILES = [0.0, 0.25, 0.5, 0.75, 1.0]
FIT_STEPS = 200          # Adam steps for the SteerNet path-loss fit (per fit call)
FIT_LR = 5e-2
STEER_SCALE = 0.5        # tanh-bounded max magnitude of each control offset entry


# ----------------------------------------------------------------------------
# spectrum summary (cheap, fixed-length, framework-light)
# ----------------------------------------------------------------------------
def _spectrum_summary(eigs) -> "list[float]":
    """Cheap fixed-length summary of a Laplacian spectrum: moments + quantiles + count.

    Returns a python list: [mean, var, q0, q25, q50, q75, q100, log10(n)].
    Sub-trivial and dataset-size-robust (count enters as log10 so SteerNet stays
    small and the magnitudes are comparable across graphs).
    """
    t = torch.as_tensor(eigs, dtype=torch.float).flatten()
    n = t.numel()
    if n == 0:
        return [0.0, 0.0] + [0.0] * len(QUANTILES) + [0.0]
    mean = float(t.mean().item())
    var = float(t.var(unbiased=False).item()) if n > 1 else 0.0
    qs = [float(torch.quantile(t, q).item()) for q in QUANTILES]
    cnt = float(torch.log10(torch.tensor(float(n))).item())
    return [mean, var] + qs + [cnt]


SUMMARY_DIM = 2 + len(QUANTILES) + 1  # mean, var, |quantiles|, log10(n)


# ----------------------------------------------------------------------------
# SteerNet: spectrum summary -> (K+1) coefficient-space control offset
# ----------------------------------------------------------------------------
class SteerNet(nn.Module):
    """Tiny steering net: spectrum summary -> bounded (K+1) control offset.

    Parameter count = (SUMMARY_DIM + 1) * (K+1). For SUMMARY_DIM=8, K=10 that is
    8*11 + 11 = 99 params (<=100, the idea spec's budget). For larger K we drop the
    bias to stay under budget; this is logged in the payload as `steer_params`.
    """

    def __init__(self, in_dim: int, k_plus_1: int, scale: float = STEER_SCALE):
        super().__init__()
        use_bias = (in_dim + 1) * k_plus_1 <= 100
        self.lin = nn.Linear(in_dim, k_plus_1, bias=use_bias)
        # start near zero so the steered control begins at the midpoint (matching
        # the midpoint-Bezier path) and can only improve from there.
        nn.init.zeros_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)
        self.scale = scale

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.lin(summary))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------
# differentiable gamma injection (no substrate edits)
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def _temp_gamma(model, gamma_tensor):
    """Temporarily replace model.gamma with a (grad-carrying) tensor.

    nn.Module forbids assigning a plain tensor onto a name registered as a
    Parameter, so we pop it from _parameters, set a plain attribute, run, then
    restore the original Parameter exactly. The model's forward uses self.gamma,
    so this routes gradients from the loss back through gamma_tensor.
    """
    orig = model._parameters.pop("gamma")
    try:
        object.__setattr__(model, "gamma", gamma_tensor)
        yield
    finally:
        if hasattr(model, "gamma"):
            object.__delattr__(model, "gamma")
        model._parameters["gamma"] = orig


def _path_loss_mean(model, batch, gamma_a, gamma_b, control, ts, mask_name):
    """Mean cross-entropy over Bezier-sampled gamma points (differentiable).

    gamma_a, gamma_b are detached constants; `control` carries grad (it depends on
    SteerNet). Returns a scalar tensor. The MLP is whatever is currently loaded in
    `model` (set to endpoint a by the caller) -- matching the names=['gamma']
    coefficient-space convention used for the reported barrier.
    """
    mask = getattr(batch, mask_name)
    y = batch.y
    total = 0.0
    for t in ts:
        gamma_t = paths.bezier_interp(gamma_a, gamma_b, control, t)
        with _temp_gamma(model, gamma_t):
            logits, _ = model.forward(batch)
        total = total + model._loss(logits, y, mask)
    return total / len(ts)


def _fit_steernet(steer, fit_jobs, ts, mask_name, steps=FIT_STEPS, lr=FIT_LR):
    """Fit SteerNet by minimizing summed sampled coeff-path loss over fit_jobs.

    fit_jobs: list of dicts each with keys model, batch, gamma_a, gamma_b, summary.
    The control for a job is midpoint(gamma_a, gamma_b) + steer(summary). All
    models are kept in eval() and their MLPs frozen at endpoint a (caller's
    responsibility); only SteerNet weights receive gradients. Returns the list of
    final per-job mean path losses (floats) for logging.
    """
    opt = torch.optim.Adam(steer.parameters(), lr=lr)
    for job in fit_jobs:
        job["model"].eval()
        for p in job["model"].parameters():
            p.requires_grad_(False)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for job in fit_jobs:
            mid = paths.bezier_midpoint_init(job["gamma_a"], job["gamma_b"])
            control = mid + steer(job["summary"])
            loss = loss + _path_loss_mean(
                job["model"], job["batch"], job["gamma_a"], job["gamma_b"],
                control, ts, mask_name,
            )
        loss.backward()
        opt.step()
    # final per-job losses (no grad)
    finals = []
    with torch.no_grad():
        for job in fit_jobs:
            mid = paths.bezier_midpoint_init(job["gamma_a"], job["gamma_b"])
            control = mid + steer(job["summary"])
            finals.append(float(_path_loss_mean(
                job["model"], job["batch"], job["gamma_a"], job["gamma_b"],
                control, ts, mask_name).item()))
    return finals


# ----------------------------------------------------------------------------
# per-dataset endpoint training + bundle
# ----------------------------------------------------------------------------
def _build_bundle(args, name, mask_name="val_mask"):
    """Train 2 endpoint models on `name`; return everything the barriers/fit need.

    Returns a dict with: prepared, batch (on device), model (harness on device,
    MLP loaded at endpoint a), gamma_a, gamma_b (detached, on device), summary
    tensor, n_eigs, val accuracies of the two endpoints.
    """
    prepared = common.prepare_dataset(name)
    _, _, tdev = common.resolve_device(args.gpus)

    model_kwargs = dict(
        hidden_dim=args.hidden_dim,
        K=args.K,
        basis=args.basis,
        domain=args.domain,
        learning_rate=args.lr,
        dropout_rate=args.dropout,
    )

    seed_a, seed_b = args.seeds[0], args.seeds[1]

    model_a = common.train_model(prepared, model_kwargs, max_epochs=args.epochs,
                                 patience=args.patience, gpus=args.gpus,
                                 seed=seed_a, verbose=False).to(tdev)
    model_b = common.train_model(prepared, model_kwargs, max_epochs=args.epochs,
                                 patience=args.patience, gpus=args.gpus,
                                 seed=seed_b, verbose=False).to(tdev)
    model_a.eval()
    model_b.eval()

    batch = common.move_batch(prepared.batch, tdev)
    eigs = common.laplacian_eigs(batch)
    summary = torch.as_tensor(_spectrum_summary(eigs), dtype=torch.float, device=tdev)

    gamma_a = common.gamma_vector(model_a).to(tdev).detach()
    gamma_b = common.gamma_vector(model_b).to(tdev).detach()

    _, acc_a = common.eval_loss_acc(model_a, batch, mask_name)
    _, acc_b = common.eval_loss_acc(model_b, batch, mask_name)

    # Harness: model_a is our path-evaluation model. Its MLP stays at endpoint a;
    # barrier_along_path / the fit only vary gamma (names=COEFF_NAMES).
    common.set_vector(model_a, gamma_a, common.COEFF_NAMES)

    return {
        "name": name,
        "prepared": prepared,
        "batch": batch,
        "model": model_a,
        "gamma_a": gamma_a,
        "gamma_b": gamma_b,
        "summary": summary,
        "n_eigs": len(eigs),
        "acc_a": float(acc_a),
        "acc_b": float(acc_b),
        "tdev": tdev,
    }


def _barrier(bundle, control, n_points, mask_name):
    """Reported coefficient-space barrier via the shared helper (control=None=linear)."""
    res = common.barrier_along_path(
        bundle["model"], bundle["batch"],
        bundle["gamma_a"], bundle["gamma_b"],
        names=common.COEFF_NAMES,
        control=control,
        n_points=n_points,
        mask_name=mask_name,
    )
    return float(res["barrier"]), float(res["argmax_t"])


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    parser = common.base_argparser(
        "idea15_steered: spectrally-steered Bezier bends -- a tiny steering net "
        "maps a graph spectrum summary to a coefficient-space Bezier control "
        "point, vs a structure-blind midpoint control; with a held-out transfer "
        "probe."
    )
    parser.add_argument("--fit_steps", type=int, default=FIT_STEPS,
                        help="Adam steps for each SteerNet path-loss fit")
    parser.add_argument("--fit_lr", type=float, default=FIT_LR,
                        help="learning rate for the SteerNet fit")
    args = parser.parse_args()
    args = common.apply_smoke(args)
    if args.smoke:
        args.fit_steps = 20
    if len(args.seeds) < 2:
        args.seeds = list(args.seeds) + [args.seeds[-1] + 1]

    mask_name = "val_mask"
    ts = paths.linspace(args.n_points)

    # ---- build per-dataset bundles (train 2 endpoints each) ----
    bundles = [_build_bundle(args, name, mask_name) for name in args.datasets]

    # ---- per-dataset barriers: linear, midpoint, per-pair-fit steered ----
    results = []
    for b in bundles:
        lin_barrier, lin_t = _barrier(b, None, args.n_points, mask_name)
        midpoint = paths.bezier_midpoint_init(b["gamma_a"], b["gamma_b"])
        mid_barrier, mid_t = _barrier(b, midpoint, args.n_points, mask_name)

        # per-pair PoC fit: a fresh SteerNet fit ONLY on this dataset's pair.
        steer = SteerNet(SUMMARY_DIM, args.K + 1).to(b["tdev"])
        fit_job = dict(model=b["model"], batch=b["batch"],
                       gamma_a=b["gamma_a"], gamma_b=b["gamma_b"],
                       summary=b["summary"])
        _fit_steernet(steer, [fit_job], ts, mask_name,
                      steps=args.fit_steps, lr=args.fit_lr)
        with torch.no_grad():
            steered_ctrl = midpoint + steer(b["summary"])
        steer_barrier, steer_t = _barrier(b, steered_ctrl, args.n_points, mask_name)

        results.append({
            "dataset": b["name"],
            "n_eigs": b["n_eigs"],
            "endpoint_val_acc": [b["acc_a"], b["acc_b"]],
            "steer_params": steer.num_params(),
            "barrier_linear": lin_barrier,
            "barrier_midpoint_bezier": mid_barrier,
            "barrier_steered_bezier": steer_barrier,
            "argmax_t": {"linear": lin_t, "midpoint": mid_t, "steered": steer_t},
            # headline deltas: positive => steering helped vs that baseline
            "delta_steered_vs_midpoint": mid_barrier - steer_barrier,
            "delta_steered_vs_linear": lin_barrier - steer_barrier,
            "steered_below_midpoint": bool(steer_barrier < mid_barrier - EPS),
        })

    # ---- held-out transfer probe: fit ONE SteerNet on datasets[0..n-2],
    #      apply frozen to datasets[-1]. Requires >= 2 datasets. ----
    transfer = None
    if len(bundles) >= 2:
        train_bundles, held = bundles[:-1], bundles[-1]
        shared = SteerNet(SUMMARY_DIM, args.K + 1).to(held["tdev"])
        fit_jobs = [dict(model=tb["model"], batch=tb["batch"],
                         gamma_a=tb["gamma_a"], gamma_b=tb["gamma_b"],
                         summary=tb["summary"]) for tb in train_bundles]
        _fit_steernet(shared, fit_jobs, ts, mask_name,
                      steps=args.fit_steps, lr=args.fit_lr)
        held_mid = paths.bezier_midpoint_init(held["gamma_a"], held["gamma_b"])
        with torch.no_grad():
            transfer_ctrl = held_mid + shared(held["summary"])
        transfer_barrier, transfer_t = _barrier(
            held, transfer_ctrl, args.n_points, mask_name)
        # baselines on the held-out dataset for comparison
        held_rec = next(r for r in results if r["dataset"] == held["name"])
        transfer = {
            "train_datasets": [tb["name"] for tb in train_bundles],
            "held_out_dataset": held["name"],
            "barrier_transfer_steered": transfer_barrier,
            "argmax_t": transfer_t,
            "barrier_linear": held_rec["barrier_linear"],
            "barrier_midpoint_bezier": held_rec["barrier_midpoint_bezier"],
            "barrier_perpair_steered": held_rec["barrier_steered_bezier"],
            "delta_transfer_vs_midpoint":
                held_rec["barrier_midpoint_bezier"] - transfer_barrier,
            "transfer_below_midpoint":
                bool(transfer_barrier < held_rec["barrier_midpoint_bezier"] - EPS),
            "shared_steer_params": shared.num_params(),
        }
    else:
        transfer = {"note": "transfer probe needs >= 2 datasets; skipped"}

    payload = {
        "experiment": "idea15_steered",
        "config": {
            "datasets": args.datasets,
            "seeds_used": [args.seeds[0], args.seeds[1]],
            "K": args.K,
            "basis": args.basis,
            "domain": args.domain,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "patience": args.patience,
            "n_points": args.n_points,
            "fit_steps": args.fit_steps,
            "fit_lr": args.fit_lr,
            "summary_dim": SUMMARY_DIM,
            "steer_scale": STEER_SCALE,
            "mask": mask_name,
            "smoke": bool(args.smoke),
            "poc_fit_note": (
                "steered barrier per dataset uses a PER-PAIR PoC fit of SteerNet "
                "on that dataset's own path loss; transfer probe fits one SteerNet "
                "on all-but-last datasets and applies it frozen to the last."
            ),
        },
        "results": results,
        "transfer": transfer,
    }

    common.write_results(args, "idea15_steered", payload)

    # ---- human summary ----
    print("\n=== idea15_steered: spectrally-steered Bezier bends (coeff space) ===")
    print("barrier (val): linear vs midpoint-Bezier vs steered-Bezier "
          "(per-pair PoC fit)")
    print(f"  {'dataset':>14s} {'params':>6s} {'linear':>9s} {'midpt':>9s} "
          f"{'steered':>9s} {'d(mid)':>9s} {'win?':>5s}")
    for r in results:
        print(f"  {r['dataset']:>14s} {r['steer_params']:>6d} "
              f"{r['barrier_linear']:>9.4f} {r['barrier_midpoint_bezier']:>9.4f} "
              f"{r['barrier_steered_bezier']:>9.4f} "
              f"{r['delta_steered_vs_midpoint']:>+9.4f} "
              f"{('yes' if r['steered_below_midpoint'] else 'no'):>5s}")
    if transfer and "held_out_dataset" in transfer:
        print(f"\n  transfer: fit on {transfer['train_datasets']} -> "
              f"apply (frozen) to '{transfer['held_out_dataset']}'")
        print(f"    midpoint={transfer['barrier_midpoint_bezier']:.4f}  "
              f"perpair-steered={transfer['barrier_perpair_steered']:.4f}  "
              f"TRANSFER-steered={transfer['barrier_transfer_steered']:.4f}  "
              f"d(mid)={transfer['delta_transfer_vs_midpoint']:+.4f}  "
              f"transfers={'yes' if transfer['transfer_below_midpoint'] else 'no'}")
    else:
        print(f"\n  transfer: {transfer.get('note', 'n/a')}")


if __name__ == "__main__":
    main()
