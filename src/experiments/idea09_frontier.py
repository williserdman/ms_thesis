"""idea09_frontier: the expressivity-connectivity frontier.

Headline question: as we increase the spectral polynomial order K (the length of
the linear coefficient axis gamma), test accuracy buys more expressivity -- but at
what point (K*) does the weight-space mode-connectivity barrier start to bend
upward, and does the basis-design conditioning kappa_K predict that barrier?

For each dataset and each K in a sweep we:
  1. train `--seeds` LinearSpectralGNN models, record mean/std TEST accuracy;
  2. measure the WEIGHT-space linear barrier (names=None, straight path over ALL
     params) between trained seed pairs, on the validation mask, via
     common.barrier_along_path -- mean over capped seed pairs;
  3. measure a FILTER-RESPONSE-space barrier: sample gamma along the *coefficient*
     linear path a->b, map each sampled gamma to its realized filter response
     g(lambda) via spectral.filter_response on common.laplacian_eigs(batch), and
     compute the barrier of a response-based loss proxy. The proxy at fraction t
     is the mean-squared response deviation from the straight-line interpolation
     of the two ENDPOINT responses:
         proxy(t) = mean_i ( g_t(lambda_i) - [(1-t) g_a + t g_b](lambda_i) )^2
     This is exactly the geometric "bulge" of the realized filter away from a
     straight morph in response space; for a LINEAR basis the coefficient path is
     linear in gamma so the response is also linear in t and the proxy is ~0 by
     construction -- a sanity floor. We therefore ALSO report the raw endpoint
     response_distance (spectral.response_distance) as the headline response-space
     scalar, plus the max along-path response distance to either endpoint.
  4. compute kappa_K = spectral.condition_number(eigs, K, basis, domain).

Across K (per dataset) we report a numpy Pearson correlation between the weight
barrier and log10(kappa_K): the idea-09 hypothesis is that ill-conditioned bases
(high kappa, e.g. monomial / large K) inflate the weight barrier.

All training / vectorization / barrier math is reused from common.py and the
numpy spectral helpers; nothing is re-implemented here.

Runtime guard: K is swept over a capped list and seed pairs are capped; both caps
are logged in the payload.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from experiments import common
from mode_connectivity import spectral, paths

# default expressivity sweep (spec: [2,4,8,12,16]); capped for runtime if needed.
K_SWEEP = [2, 4, 8, 12, 16]
EPS = 1e-12


def _pearson(x, y):
    """Pearson correlation of two python lists (numpy); None if degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) < EPS or np.std(y) < EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _response_space_barrier(gamma_a, gamma_b, eigs, basis, domain, n_points):
    """Barrier of a filter-response loss proxy along the coefficient linear path.

    Samples gamma(t) = (1-t) gamma_a + t gamma_b, maps each to its realized
    response g_t = filter_response(gamma(t)), and measures how far g_t bulges from
    the straight-line interpolation of the endpoint responses g_a, g_b.

    Returns a dict with:
      proxy_barrier   -- max_t mean-squared response bulge (>=0; ~0 for linear bases)
      endpoint_dist   -- response_distance(gamma_a, gamma_b)  (headline scalar)
      max_path_dist   -- max over interior t of min(dist(g_t,g_a), dist(g_t,g_b))
      argmax_t        -- t of the proxy barrier
    """
    ga = np.asarray(spectral.filter_response(gamma_a, eigs, basis, domain), dtype=float)
    gb = np.asarray(spectral.filter_response(gamma_b, eigs, basis, domain), dtype=float)
    n_eig = max(len(eigs), 1)

    ts = paths.linspace(n_points)
    proxy = []
    path_dists = []
    for t in ts:
        gamma_t = paths.linear_interp(np.asarray(gamma_a, float),
                                      np.asarray(gamma_b, float), t)
        g_t = np.asarray(spectral.filter_response(gamma_t, eigs, basis, domain), dtype=float)
        baseline = (1.0 - t) * ga + t * gb
        proxy.append(float(np.mean((g_t - baseline) ** 2)))
        da = float(np.linalg.norm(g_t - ga) / np.sqrt(n_eig))
        db = float(np.linalg.norm(g_t - gb) / np.sqrt(n_eig))
        path_dists.append(min(da, db))

    proxy_barrier = paths.barrier_from_losses(proxy, ts)
    argmax_t = paths.argmax_barrier_t(proxy, ts)
    # interior max distance to the nearer endpoint (exclude the two endpoints)
    interior = path_dists[1:-1] if len(path_dists) > 2 else path_dists
    max_path_dist = float(max(interior)) if interior else 0.0
    endpoint_dist = spectral.response_distance(gamma_a, gamma_b, eigs, basis, domain)
    return {
        "proxy_barrier": float(proxy_barrier),
        "endpoint_dist": float(endpoint_dist),
        "max_path_dist": float(max_path_dist),
        "argmax_t": float(argmax_t),
    }


def _run_dataset(args, name, mask_name="val_mask"):
    prepared = common.prepare_dataset(name)
    _, _, tdev = common.resolve_device(args.gpus)
    eigs = None  # computed lazily once we have a batch on device

    # cap the K sweep if requested (runtime guard); always log what was used.
    k_list = [k for k in args.k_sweep if k >= 1]
    n_k_capped = len(args.k_sweep) - len(k_list)

    all_pairs = list(combinations(args.seeds, 2))
    pairs = all_pairs[:args.max_pairs]
    n_pairs_capped = len(all_pairs) - len(pairs)

    per_k = []
    for K in k_list:
        model_kwargs = dict(
            hidden_dim=args.hidden_dim,
            K=K,
            basis=args.basis,
            domain=args.domain,
            learning_rate=args.lr,
            dropout_rate=args.dropout,
        )

        # train one model per seed; record test acc + trained full vectors.
        trained_vecs = {}
        trained_gammas = {}
        test_accs = []
        for seed in args.seeds:
            model = common.train_model(
                prepared, model_kwargs,
                max_epochs=args.epochs, patience=args.patience,
                gpus=args.gpus, seed=seed, verbose=False,
            )
            model = model.to(tdev)
            model.eval()
            batch = common.move_batch(prepared.batch, tdev)
            if eigs is None:
                eigs = common.laplacian_eigs(batch)
            _, acc = common.eval_loss_acc(model, batch, "test_mask")
            test_accs.append(float(acc))
            trained_vecs[seed] = common.get_vector(model, None).clone()
            trained_gammas[seed] = (
                common.gamma_vector(model).detach().cpu().numpy().astype(float)
            )

        # evaluation harness: swap trained weights in/out of one fresh model.
        common.set_seed(args.seeds[0])
        harness = common.LinearSpectralGNN(prepared.ds_info, **model_kwargs).to(tdev)
        harness.eval()
        batch = common.move_batch(prepared.batch, tdev)

        weight_barriers = []
        response_barriers = []
        endpoint_dists = []
        max_path_dists = []
        for (sa, sb) in pairs:
            vec_a = trained_vecs[sa].to(tdev)
            vec_b = trained_vecs[sb].to(tdev)

            # (2) weight-space LINEAR barrier over ALL params (names=None).
            common.set_vector(harness, vec_a, None)
            wb = common.barrier_along_path(
                harness, batch, vec_a, vec_b, names=None, control=None,
                n_points=args.n_points, mask_name=mask_name,
            )
            weight_barriers.append(float(wb["barrier"]))

            # (3) filter-response-space barrier along the coefficient path.
            rb = _response_space_barrier(
                trained_gammas[sa], trained_gammas[sb], eigs,
                args.basis, args.domain, args.n_points,
            )
            response_barriers.append(rb["proxy_barrier"])
            endpoint_dists.append(rb["endpoint_dist"])
            max_path_dists.append(rb["max_path_dist"])

        # (4) basis-design conditioning at this K.
        kappa_K = spectral.condition_number(eigs, K, args.basis, args.domain)

        def _mean(v):
            return float(np.mean(v)) if v else None

        def _std(v):
            return float(np.std(v)) if v else None

        per_k.append({
            "K": int(K),
            "test_acc_mean": _mean(test_accs),
            "test_acc_std": _std(test_accs),
            "weight_barrier_mean": _mean(weight_barriers),
            "weight_barrier_std": _std(weight_barriers),
            "response_barrier_mean": _mean(response_barriers),
            "response_endpoint_dist_mean": _mean(endpoint_dists),
            "response_max_path_dist_mean": _mean(max_path_dists),
            "kappa_K": float(kappa_K),
            "log10_kappa_K": (float(np.log10(kappa_K))
                              if np.isfinite(kappa_K) and kappa_K > 0 else None),
            "n_pairs_used": len(pairs),
        })

    # ---- across-K analysis (per dataset) ----
    Ks = [r["K"] for r in per_k]
    accs = [r["test_acc_mean"] for r in per_k]
    wbars = [r["weight_barrier_mean"] for r in per_k]
    logkappas = [r["log10_kappa_K"] for r in per_k]

    # correlation between weight barrier and log10(kappa) across K.
    pair_kw = [(w, lk) for w, lk in zip(wbars, logkappas)
               if w is not None and lk is not None]
    corr_wbar_logkappa = (_pearson([p[0] for p in pair_kw], [p[1] for p in pair_kw])
                          if len(pair_kw) >= 2 else None)
    # also raw-kappa correlation for reference.
    kappas = [r["kappa_K"] for r in per_k]
    pair_kk = [(w, k) for w, k in zip(wbars, kappas)
               if w is not None and np.isfinite(k)]
    corr_wbar_kappa = (_pearson([p[0] for p in pair_kk], [p[1] for p in pair_kk])
                       if len(pair_kk) >= 2 else None)

    # K* = the K where accuracy gains start to flatten / barrier starts bending up.
    # Heuristic: largest K whose accuracy is within 0.5% of the best accuracy AND
    # whose weight barrier is below the max barrier -- i.e. the "elbow" of the
    # frontier. Falls back to argmax accuracy.
    k_star = _frontier_kstar(Ks, accs, wbars)

    return {
        "dataset": name,
        "n_seeds": len(args.seeds),
        "n_pairs_used": len(pairs),
        "n_pairs_capped": n_pairs_capped,
        "n_k_capped": n_k_capped,
        "k_sweep_used": Ks,
        "mask": mask_name,
        "n_eigs": (len(eigs) if eigs is not None else 0),
        "per_K": per_k,
        "corr_weight_barrier_vs_log10kappa": corr_wbar_logkappa,
        "corr_weight_barrier_vs_kappa": corr_wbar_kappa,
        "k_star": k_star,
    }


def _frontier_kstar(Ks, accs, wbars):
    """Pick the frontier elbow K*: best accuracy with the smallest weight barrier.

    We score each K by (accuracy - lambda * normalized_barrier) and take the
    argmax; the smallest such K on ties. Pure-python/numpy, robust to None.
    """
    valid = [(K, a, w) for K, a, w in zip(Ks, accs, wbars)
             if a is not None and w is not None]
    if not valid:
        # fall back to argmax accuracy ignoring barrier
        acc_valid = [(K, a) for K, a in zip(Ks, accs) if a is not None]
        if not acc_valid:
            return None
        return int(max(acc_valid, key=lambda t: t[1])[0])
    ws = np.array([w for _, _, w in valid], dtype=float)
    wrange = float(ws.max() - ws.min())
    best_k, best_score = None, -float("inf")
    for K, a, w in valid:
        wn = (w - ws.min()) / wrange if wrange > EPS else 0.0
        score = a - 0.5 * wn  # accuracy traded off against normalized barrier
        if score > best_score + EPS or (abs(score - best_score) <= EPS
                                        and (best_k is None or K < best_k)):
            best_score, best_k = score, K
    return int(best_k)


def main():
    parser = common.base_argparser(
        "idea09_frontier: expressivity (K) vs mode-connectivity barrier frontier; "
        "weight-space barrier, filter-response-space barrier, and basis "
        "conditioning kappa_K, with a kappa-vs-barrier correlation across K."
    )
    parser.add_argument("--k_sweep", nargs="+", type=int, default=K_SWEEP,
                        help="expressivity sweep over polynomial order K")
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="cap on seed pairs per (K,dataset) (runtime guard)")
    args = parser.parse_args()
    args = common.apply_smoke(args)
    if args.smoke:
        args.k_sweep = [2, 4]
        args.max_pairs = 1

    results = []
    for name in args.datasets:
        rec = _run_dataset(args, name)
        results.append(rec)

    payload = {
        "experiment": "idea09_frontier",
        "config": {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "k_sweep": args.k_sweep,
            "basis": args.basis,
            "domain": args.domain,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "patience": args.patience,
            "n_points": args.n_points,
            "max_pairs": args.max_pairs,
            "smoke": bool(args.smoke),
        },
        "results": results,
    }

    common.write_results(args, "idea09_frontier", payload)

    # human summary
    print("\n=== idea09_frontier: expressivity-connectivity frontier ===")
    print("per-K: test acc, weight-space barrier, response-space barrier, kappa_K")
    for rec in results:
        caps = []
        if rec["n_k_capped"]:
            caps.append(f"{rec['n_k_capped']} K capped")
        if rec["n_pairs_capped"]:
            caps.append(f"{rec['n_pairs_capped']} pairs capped")
        capnote = f"  [{', '.join(caps)}]" if caps else ""
        print(f"\n  {rec['dataset']} (basis={args.basis}, domain={args.domain}, "
              f"pairs={rec['n_pairs_used']}){capnote}")
        print(f"    {'K':>4s} {'test_acc':>9s} {'w_barrier':>10s} "
              f"{'r_barrier':>10s} {'kappa_K':>12s}")
        for r in rec["per_K"]:
            acc = r["test_acc_mean"]
            wb = r["weight_barrier_mean"]
            rb = r["response_barrier_mean"]
            kap = r["kappa_K"]
            print(f"    {r['K']:>4d} "
                  f"{(acc if acc is not None else float('nan')):>9.4f} "
                  f"{(wb if wb is not None else float('nan')):>10.4f} "
                  f"{(rb if rb is not None else float('nan')):>10.4g} "
                  f"{kap:>12.4g}")
        cw = rec["corr_weight_barrier_vs_log10kappa"]
        cw_s = f"{cw:+.3f}" if cw is not None else "n/a"
        print(f"    -> K* (frontier elbow) = {rec['k_star']}  |  "
              f"corr(weight_barrier, log10 kappa_K) = {cw_s}")


if __name__ == "__main__":
    main()
