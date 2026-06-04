"""B1 gating pilot: does a loss barrier exist in the low-dim coefficient axis?

For each dataset we train ``args.seeds`` (>=3) independent ``LinearSpectralGNN``
models from different seeds. For every unordered seed PAIR (a, b) we compute three
val-mask loss barriers via ``common.barrier_along_path``:

  (1) full-weight LINEAR  -- interpolate ALL params (names=None).
  (2) coeff-only LINEAR   -- interpolate ONLY gamma (names=COEFF_NAMES) on top of
                             endpoint A's full weights (A's MLP held fixed).
  (3) coeff-only BEZIER    -- ONLY gamma along a quadratic Bezier whose control is
                             the straight-line midpoint (paths.bezier_midpoint_init),
                             a sanity floor that can only reduce the barrier vs (2).

Per dataset we report mean/max of each barrier across pairs and the boolean
``coeff_barrier_nontrivial`` (mean coeff-linear barrier > 0.02). That boolean is THE
gate the downstream experiments depend on. Output JSON holds raw + aggregated.

Run from src/:  python -m experiments.b1_barrier --smoke
"""

from __future__ import annotations

from itertools import combinations

from experiments import common
from mode_connectivity import paths

NAME = "b1_barrier"

# Gate threshold: mean coeff-only LINEAR barrier above this => a real barrier
# exists in the coefficient axis, so the downstream connectivity experiments
# have something to study.
COEFF_BARRIER_GATE = 0.02


def _summ(values):
    """mean/max/min of a list of floats (empty -> zeros)."""
    if not values:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "n": 0}
    return {
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
        "min": float(min(values)),
        "n": len(values),
    }


def run_dataset(prepared, args):
    """Train one model per seed, then barriers over all unordered seed pairs."""
    model_kwargs = dict(
        hidden_dim=args.hidden_dim,
        K=args.K,
        basis=args.basis,
        domain=args.domain,
        dropout_rate=args.dropout,
        learning_rate=args.lr,
    )

    # --- train one independent model per seed ---
    models = {}
    val_metrics = {}
    for seed in args.seeds:
        model = common.train_model(
            prepared,
            model_kwargs,
            max_epochs=args.epochs,
            patience=args.patience,
            gpus=args.gpus,
            seed=seed,
            verbose=False,
        )
        _, _, torch_device = common.resolve_device(args.gpus)
        model = model.to(torch_device)
        batch = common.move_batch(prepared.batch, torch_device)
        vloss, vacc = common.eval_loss_acc(model, batch, mask_name="val_mask")
        models[seed] = model
        val_metrics[seed] = {"val_loss": float(vloss), "val_acc": float(vacc)}

    # --- barriers over every unordered seed pair ---
    pairs = []
    full_lin, coeff_lin, coeff_bez = [], [], []
    for sa, sb in combinations(args.seeds, 2):
        model_a = models[sa]
        model_b = models[sb]
        _, _, torch_device = common.resolve_device(args.gpus)
        batch = common.move_batch(prepared.batch, torch_device)

        # full param vectors (all params, consistent ordering)
        full_a = common.get_vector(model_a, names=None)
        full_b = common.get_vector(model_b, names=None)
        # coefficient-only (gamma) vectors
        gamma_a = common.get_vector(model_a, names=common.COEFF_NAMES)
        gamma_b = common.get_vector(model_b, names=common.COEFF_NAMES)

        # (1) full-weight LINEAR barrier (interpolate ALL params on model_a)
        res_full = common.barrier_along_path(
            model_a, batch, full_a, full_b,
            names=None, control=None,
            n_points=args.n_points, mask_name="val_mask",
        )

        # (2) coeff-only LINEAR barrier: hold A's full weights live, vary only gamma.
        # Ensure model_a carries A's own weights (barrier_along_path restores what it
        # touches, so A's MLP is already its own; we interpolate only gamma).
        common.set_vector(model_a, full_a, names=None)
        res_coeff = common.barrier_along_path(
            model_a, batch, gamma_a, gamma_b,
            names=common.COEFF_NAMES, control=None,
            n_points=args.n_points, mask_name="val_mask",
        )

        # (3) coeff-only BEZIER barrier (sanity floor): control = straight midpoint.
        control = paths.bezier_midpoint_init(gamma_a, gamma_b)
        common.set_vector(model_a, full_a, names=None)
        res_bez = common.barrier_along_path(
            model_a, batch, gamma_a, gamma_b,
            names=common.COEFF_NAMES, control=control,
            n_points=args.n_points, mask_name="val_mask",
        )

        full_lin.append(res_full["barrier"])
        coeff_lin.append(res_coeff["barrier"])
        coeff_bez.append(res_bez["barrier"])
        pairs.append({
            "seed_a": sa,
            "seed_b": sb,
            "full_linear": {
                "barrier": res_full["barrier"],
                "argmax_t": res_full["argmax_t"],
                "losses": res_full["losses"],
            },
            "coeff_linear": {
                "barrier": res_coeff["barrier"],
                "argmax_t": res_coeff["argmax_t"],
                "losses": res_coeff["losses"],
            },
            "coeff_bezier": {
                "barrier": res_bez["barrier"],
                "argmax_t": res_bez["argmax_t"],
                "losses": res_bez["losses"],
            },
        })

    agg = {
        "full_linear": _summ(full_lin),
        "coeff_linear": _summ(coeff_lin),
        "coeff_bezier": _summ(coeff_bez),
    }
    coeff_barrier_nontrivial = bool(agg["coeff_linear"]["mean"] > COEFF_BARRIER_GATE)

    return {
        "seeds": list(args.seeds),
        "ts": paths.linspace(args.n_points),
        "val_metrics": val_metrics,
        "pairs": pairs,
        "aggregated": agg,
        "coeff_barrier_nontrivial": coeff_barrier_nontrivial,
        "gate_threshold": COEFF_BARRIER_GATE,
    }


def main():
    parser = common.base_argparser(
        "B1 gating pilot: barrier existence in the coefficient (gamma) axis."
    )
    args = parser.parse_args()
    args = common.apply_smoke(args)

    if len(args.seeds) < 3 and not args.smoke:
        print(f"[{NAME}] WARNING: spec wants >=3 seeds for pairs; got "
              f"{len(args.seeds)} -> only {max(0, len(args.seeds) - 1)} pairs.")

    results = {}
    for ds_name in args.datasets:
        print(f"[{NAME}] === dataset: {ds_name} ===")
        prepared = common.prepare_dataset(ds_name)
        results[ds_name] = run_dataset(prepared, args)

    payload = {
        "experiment": NAME,
        "config": {
            "datasets": list(args.datasets),
            "seeds": list(args.seeds),
            "K": args.K,
            "basis": args.basis,
            "domain": args.domain,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "patience": args.patience,
            "n_points": args.n_points,
            "gate_threshold": COEFF_BARRIER_GATE,
            "smoke": args.smoke,
        },
        "results": results,
    }
    common.write_results(args, NAME, payload)

    # --- concise human summary: the gate metric per dataset ---
    print(f"\n[{NAME}] summary (gate = mean coeff-linear barrier > {COEFF_BARRIER_GATE}):")
    for ds_name, r in results.items():
        agg = r["aggregated"]
        print(
            f"  {ds_name:>14}: coeff-lin mean={agg['coeff_linear']['mean']:.4f} "
            f"max={agg['coeff_linear']['max']:.4f} | "
            f"full-lin mean={agg['full_linear']['mean']:.4f} | "
            f"coeff-bezier mean={agg['coeff_bezier']['mean']:.4f} | "
            f"NONTRIVIAL={r['coeff_barrier_nontrivial']}"
        )


if __name__ == "__main__":
    main()
