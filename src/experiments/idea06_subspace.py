"""idea06_subspace: subspace-restricted curve finding.

Question (headline): is the full-weight LINEAR mode-connectivity barrier between
two trained LinearSpectralGNN solutions removable by bending only the *filter*
axis (the (K+1)-dim spectral coefficient vector gamma), or only by bending the
*mixer* (the feature MLP)?

For each dataset and a few seed pairs (a, b) we measure three barriers, all on
the validation mask, using common.py helpers (no re-implemented training /
barrier math):

  (a) B_full_linear  -- barrier of the straight (linear) path over ALL weights.
  (b) B_coef_bent    -- barrier of a quadratic-Bezier path that bends ONLY gamma;
                        the MLP is held identical at the two endpoints' shared
                        values along the path (names = COEFF_NAMES). The Bezier
                        control point is chosen by a cheap 1-D scalar grid search
                        (see _best_coef_control) to lower the barrier.
  (c) B_bulk_bent    -- barrier of a quadratic-Bezier path that bends ONLY the MLP
                        (names = mlp_names(model)); same cheap scalar search.

  rho_coef = (B_full_linear - B_coef_bent) / max(B_full_linear, eps)
  rho_bulk = (B_full_linear - B_bulk_bent) / max(B_full_linear, eps)

rho in [~0, 1]: a value near 1 means bending that single subspace removes (most
of) the barrier; near 0 means that subspace is not where the obstruction lives.

NOTE on the subset-barrier convention: barrier_along_path(..., names=S) only
interpolates the parameters in S; parameters NOT in S keep their *current* model
state (which we set to endpoint a before each call). So B_coef_bent / B_bulk_bent
are barriers within the chosen subspace, with the complementary weights frozen at
a. B_full_linear interpolates everything (names=None). This matches the spec's
"bend only gamma" / "bend only the mlp" framing.

Bezier control search (cheap PoC method, documented):
  We never run an optimizer. For the subset S we take the linear-path endpoints
  (a_S, b_S), form the straight midpoint m = 0.5*(a_S + b_S), and grid-search a
  scalar bend magnitude s over a fixed symmetric grid. The search *direction* d
  is the off-chord direction that points from the midpoint toward whichever
  endpoint subspace-vector yields the lower endpoint loss when its complement is
  frozen at a -- i.e. d = (target_S - m). control(s) = m + s * d. Because a
  quadratic Bezier's t=0.5 point is 0.25*a + 0.5*control + 0.25*b, moving control
  directly lowers the interior of the path. s=0 reproduces the linear path
  (barrier can only stay equal or drop). We keep the control giving the smallest
  barrier. This is intentionally a 1-D scalar search (cheap, <= len(grid) extra
  path evaluations per subset) rather than a full Bezier fit.
"""

from __future__ import annotations

from itertools import combinations

import torch

from experiments import common


# fixed, small scalar bend grid (s=0 == linear path; symmetric so the search can
# bend toward or away from the chosen target subspace vector).
BEND_GRID = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
EPS = 1e-8


def _endpoint_loss_with_frozen_complement(model, batch, vec_sub, names, mask_name):
    """Loss when the subset `names` is set to vec_sub and the complement stays put."""
    saved = common.get_vector(model, names)
    try:
        common.set_vector(model, vec_sub, names)
        loss, _ = common.eval_loss_acc(model, batch, mask_name)
    finally:
        common.set_vector(model, saved, names)
    return loss


def _best_coef_control(model, batch, a_sub, b_sub, names, n_points, mask_name):
    """Grid-search a scalar bend magnitude for the Bezier control on subset `names`.

    Returns (best_control, best_barrier, best_s, n_evals). The complement weights
    are assumed already frozen at endpoint a by the caller; we restore the subset
    after each trial inside barrier_along_path (it restores internally).
    """
    midpoint = 0.5 * (a_sub + b_sub)

    # pick the off-chord search target = whichever endpoint subset-vector gives the
    # lower endpoint loss (complement frozen at a). Direction d = target - midpoint.
    la = _endpoint_loss_with_frozen_complement(model, batch, a_sub, names, mask_name)
    lb = _endpoint_loss_with_frozen_complement(model, batch, b_sub, names, mask_name)
    target = a_sub if la <= lb else b_sub
    direction = target - midpoint

    best_control = None
    best_barrier = float("inf")
    best_s = 0.0
    n_evals = 0
    for s in BEND_GRID:
        control = midpoint + s * direction
        res = common.barrier_along_path(
            model, batch, a_sub, b_sub, names=names, control=control,
            n_points=n_points, mask_name=mask_name,
        )
        n_evals += 1
        if res["barrier"] < best_barrier:
            best_barrier = res["barrier"]
            best_control = control
            best_s = s
    return best_control, best_barrier, best_s, n_evals


def _subspace_bent_barrier(model, batch, vec_a_full, vec_b_full, names, n_points, mask_name):
    """Best Bezier barrier when bending ONLY `names`, complement frozen at a.

    Restores the full model to vec_a_full on exit.
    """
    saved_full = common.get_vector(model, None)
    try:
        # freeze the complement at endpoint a by loading the full a-vector first.
        common.set_vector(model, vec_a_full, None)
        a_sub = common.get_vector(model, names)
        # b on the subset comes from the b-endpoint full vector.
        common.set_vector(model, vec_b_full, None)
        b_sub = common.get_vector(model, names)
        # restore complement to a before searching (subset will be swept).
        common.set_vector(model, vec_a_full, None)

        control, barrier, s, n_evals = _best_coef_control(
            model, batch, a_sub, b_sub, names, n_points, mask_name,
        )
    finally:
        common.set_vector(model, saved_full, None)
    return {"barrier": float(barrier), "bend_s": float(s), "n_path_evals": int(n_evals)}


def _run_dataset(args, name, mask_name="val_mask"):
    prepared = common.prepare_dataset(name)
    model_kwargs = dict(
        hidden_dim=args.hidden_dim,
        K=args.K,
        basis=args.basis,
        domain=args.domain,
        learning_rate=args.lr,
        dropout_rate=args.dropout,
    )

    # train one model per seed
    trained_vecs = {}
    for seed in args.seeds:
        model = common.train_model(
            prepared, model_kwargs,
            max_epochs=args.epochs, patience=args.patience,
            gpus=args.gpus, seed=seed, verbose=False,
        )
        trained_vecs[seed] = common.get_vector(model, None).clone()

    # build a fresh model as the evaluation harness (params get swapped in/out)
    common.set_seed(args.seeds[0])
    harness = common.LinearSpectralGNN(prepared.ds_info, **model_kwargs)
    _, _, tdev = common.resolve_device(args.gpus)
    harness = harness.to(tdev)
    batch = common.move_batch(prepared.batch, tdev)
    harness.eval()

    mlp = common.mlp_names(harness)

    # seed pairs: cap to keep runtime bounded
    all_pairs = list(combinations(args.seeds, 2))
    max_pairs = getattr(args, "max_pairs", 3)
    pairs = all_pairs[:max_pairs]
    capped_pairs = len(all_pairs) - len(pairs)

    pair_records = []
    for (sa, sb) in pairs:
        vec_a = trained_vecs[sa].to(tdev)
        vec_b = trained_vecs[sb].to(tdev)

        # (a) full-weight LINEAR barrier (names=None, control=None)
        common.set_vector(harness, vec_a, None)
        full = common.barrier_along_path(
            harness, batch, vec_a, vec_b, names=None, control=None,
            n_points=args.n_points, mask_name=mask_name,
        )
        B_full_linear = float(full["barrier"])

        # (b) coefficient-axis bent barrier (bend only gamma)
        coef = _subspace_bent_barrier(
            harness, batch, vec_a, vec_b, common.COEFF_NAMES,
            args.n_points, mask_name,
        )
        B_coef_bent = coef["barrier"]

        # (c) bulk/mixer bent barrier (bend only the MLP)
        bulk = _subspace_bent_barrier(
            harness, batch, vec_a, vec_b, mlp,
            args.n_points, mask_name,
        )
        B_bulk_bent = bulk["barrier"]

        denom = max(B_full_linear, EPS)
        rho_coef = (B_full_linear - B_coef_bent) / denom
        rho_bulk = (B_full_linear - B_bulk_bent) / denom

        pair_records.append({
            "seed_a": int(sa),
            "seed_b": int(sb),
            "B_full_linear": B_full_linear,
            "B_coef_bent": B_coef_bent,
            "B_bulk_bent": B_bulk_bent,
            "rho_coef": float(rho_coef),
            "rho_bulk": float(rho_bulk),
            "coef_bend_s": coef["bend_s"],
            "bulk_bend_s": bulk["bend_s"],
        })

    def _mean(key):
        vals = [r[key] for r in pair_records]
        return float(sum(vals) / len(vals)) if vals else None

    return {
        "dataset": name,
        "n_seeds": len(args.seeds),
        "n_pairs_used": len(pairs),
        "n_pairs_capped": capped_pairs,
        "mask": mask_name,
        "pairs": pair_records,
        "mean_B_full_linear": _mean("B_full_linear"),
        "mean_B_coef_bent": _mean("B_coef_bent"),
        "mean_B_bulk_bent": _mean("B_bulk_bent"),
        "mean_rho_coef": _mean("rho_coef"),
        "mean_rho_bulk": _mean("rho_bulk"),
    }


def main():
    parser = common.base_argparser(
        "idea06_subspace: fraction of the linear weight barrier removable by "
        "bending only the spectral coefficient axis (gamma) vs only the MLP."
    )
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="cap on number of seed pairs per dataset (runtime guard)")
    args = parser.parse_args()
    args = common.apply_smoke(args)
    if args.smoke:
        args.max_pairs = 1

    results = []
    for name in args.datasets:
        rec = _run_dataset(args, name)
        results.append(rec)

    payload = {
        "experiment": "idea06_subspace",
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
            "n_points": args.n_points,
            "max_pairs": args.max_pairs,
            "bend_grid": BEND_GRID,
            "smoke": bool(args.smoke),
        },
        "results": results,
    }

    common.write_results(args, "idea06_subspace", payload)

    # human summary
    print("\n=== idea06_subspace: barrier removability by subspace ===")
    print("headline rho_coef -> bend only gamma (filter axis); "
          "rho_bulk -> bend only MLP (mixer). higher = more barrier removed.")
    for rec in results:
        capnote = (f" [capped {rec['n_pairs_capped']} pairs]"
                   if rec["n_pairs_capped"] else "")
        print(
            f"  {rec['dataset']:>14s}: "
            f"B_full_lin={rec['mean_B_full_linear']:.4f}  "
            f"rho_coef={rec['mean_rho_coef']:.3f}  "
            f"rho_bulk={rec['mean_rho_bulk']:.3f}"
            f"  (pairs={rec['n_pairs_used']}{capnote})"
        )
        winner = ("filter/gamma" if (rec["mean_rho_coef"] or 0) >= (rec["mean_rho_bulk"] or 0)
                  else "mixer/MLP")
        print(f"  {'':>14s}  -> barrier more removable from: {winner}")


if __name__ == "__main__":
    main()
