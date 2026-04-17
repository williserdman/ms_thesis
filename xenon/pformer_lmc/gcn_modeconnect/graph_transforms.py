from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected


@dataclass
class GraphCondition:
    condition_id: str = "baseline"
    homophily_target: float | None = None
    sparsity_keep: float = 1.0
    degree_gamma: float = 1.0
    synthetic: bool = False
    synthetic_type: str = "label_sbm"
    synthetic_p_in: float = 0.10
    synthetic_p_out: float = 0.01
    synthetic_target_edges: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_undirected_simple(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def _edge_set_from_undirected(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    rows = edge_index[0].tolist()
    cols = edge_index[1].tolist()
    out: set[tuple[int, int]] = set()
    for u, v in zip(rows, cols):
        a, b = (u, v) if u < v else (v, u)
        if a != b:
            out.add((a, b))
    return out


def _edge_index_from_set(edges: set[tuple[int, int]], num_nodes: int) -> torch.Tensor:
    undirected = []
    for u, v in edges:
        undirected.append((u, v))
        undirected.append((v, u))
    if not undirected:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(undirected, dtype=torch.long).t().contiguous()
    return _make_undirected_simple(edge_index, num_nodes)


def edge_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    if edge_index.numel() == 0:
        return 0.0
    src = edge_index[0]
    dst = edge_index[1]
    same = (y[src] == y[dst]).float()
    return float(same.mean().item())


def _sample_non_edge(
    rng: random.Random,
    num_nodes: int,
    existing: set[tuple[int, int]],
    same_label: bool,
    y: torch.Tensor,
    max_tries: int = 1000,
) -> tuple[int, int] | None:
    for _ in range(max_tries):
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing:
            continue
        if bool((y[a] == y[b]).item()) != same_label:
            continue
        return (a, b)
    return None


def _rewire_homophily(edges: set[tuple[int, int]], y: torch.Tensor, target: float, seed: int) -> set[tuple[int, int]]:
    rng = random.Random(seed)
    current = set(edges)

    def ratio(es: set[tuple[int, int]]) -> float:
        if not es:
            return 0.0
        same = 0
        for u, v in es:
            if int(y[u]) == int(y[v]):
                same += 1
        return same / len(es)

    desired_increase = target > ratio(current)
    max_steps = len(current) * 2
    for _ in range(max_steps):
        now = ratio(current)
        if abs(now - target) < 1e-3:
            break

        candidates = list(current)
        rng.shuffle(candidates)
        swapped = False
        for e in candidates:
            u, v = e
            is_same = int(y[u]) == int(y[v])
            if desired_increase and is_same:
                continue
            if (not desired_increase) and (not is_same):
                continue
            replacement = _sample_non_edge(
                rng=rng,
                num_nodes=y.numel(),
                existing=current,
                same_label=desired_increase,
                y=y,
            )
            if replacement is None:
                continue
            current.remove(e)
            current.add(replacement)
            swapped = True
            break

        if not swapped:
            break
        desired_increase = target > ratio(current)

    return current


def _apply_sparsity(edges: set[tuple[int, int]], keep: float, seed: int) -> set[tuple[int, int]]:
    keep = float(max(0.05, min(1.0, keep)))
    if keep >= 1.0:
        return set(edges)
    rng = random.Random(seed)
    edges_list = list(edges)
    rng.shuffle(edges_list)
    kept = int(round(len(edges_list) * keep))
    kept = max(1, kept)
    return set(edges_list[:kept])


def _apply_degree_rewire(edges: set[tuple[int, int]], y: torch.Tensor, gamma: float, seed: int) -> set[tuple[int, int]]:
    if abs(gamma - 1.0) < 1e-6:
        return set(edges)

    rng = random.Random(seed)
    num_nodes = y.numel()
    deg = [0 for _ in range(num_nodes)]
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    weights = [(d + 1) ** gamma for d in deg]
    nodes = list(range(num_nodes))
    total_edges = len(edges)
    out: set[tuple[int, int]] = set()

    tries = 0
    max_tries = total_edges * 50
    while len(out) < total_edges and tries < max_tries:
        tries += 1
        u = rng.choices(nodes, weights=weights, k=1)[0]
        v = rng.choices(nodes, weights=weights, k=1)[0]
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        out.add((a, b))

    if len(out) < max(1, total_edges // 2):
        return set(edges)
    return out


def _synthetic_from_labels(data: Data, p_in: float, p_out: float, seed: int) -> set[tuple[int, int]]:
    rng = random.Random(seed)
    y = data.y.cpu()
    n = y.numel()
    out: set[tuple[int, int]] = set()
    for u in range(n):
        for v in range(u + 1, n):
            p = p_in if int(y[u]) == int(y[v]) else p_out
            if rng.random() < p:
                out.add((u, v))
    if not out:
        for i in range(max(1, n - 1)):
            out.add((i, i + 1))
    return out


def _weighted_same_label_probability(y: torch.Tensor, weights: list[float]) -> float:
    total_w = float(sum(weights))
    if total_w <= 0:
        return 0.0
    class_sum: dict[int, float] = {}
    for idx, cls in enumerate(y.tolist()):
        class_sum[int(cls)] = class_sum.get(int(cls), 0.0) + float(weights[idx])
    return float(sum((v / total_w) ** 2 for v in class_sum.values()))


def _solve_pin_pout_for_homophily(
    h_target: float | None,
    r_same: float,
    p_in_default: float,
    p_out_default: float,
) -> tuple[float, float]:
    if h_target is None:
        return p_in_default, p_out_default

    h = float(max(0.001, min(0.999, h_target)))
    r = float(max(1e-6, min(1 - 1e-6, r_same)))

    p_out = float(max(1e-5, min(1.0, p_out_default)))
    p_in = (h * (1.0 - r) * p_out) / (r * (1.0 - h))

    if p_in > 1.0:
        p_in = 1.0
        p_out = (r * (1.0 - h) * p_in) / (h * (1.0 - r))
    p_in = float(max(1e-5, min(1.0, p_in)))
    p_out = float(max(1e-5, min(1.0, p_out)))
    return p_in, p_out


def _sample_weighted_pair(rng: random.Random, num_nodes: int, weights: list[float]) -> tuple[int, int]:
    nodes = list(range(num_nodes))
    u = rng.choices(nodes, weights=weights, k=1)[0]
    v = rng.choices(nodes, weights=weights, k=1)[0]
    return u, v


def _synthetic_dcsbm(
    y: torch.Tensor,
    num_nodes: int,
    target_edges: int,
    degree_gamma: float,
    p_in_default: float,
    p_out_default: float,
    homophily_target: float | None,
    seed: int,
) -> tuple[set[tuple[int, int]], dict[str, float]]:
    rng = random.Random(seed)
    deg_proxy = [1.0 for _ in range(num_nodes)]
    weights = [d**degree_gamma for d in deg_proxy]
    r_same = _weighted_same_label_probability(y, weights)
    p_in, p_out = _solve_pin_pout_for_homophily(homophily_target, r_same, p_in_default, p_out_default)

    out: set[tuple[int, int]] = set()
    max_tries = max(1000, target_edges * 200)
    tries = 0
    while len(out) < target_edges and tries < max_tries:
        tries += 1
        u, v = _sample_weighted_pair(rng, num_nodes=num_nodes, weights=weights)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in out:
            continue
        same = int(y[a]) == int(y[b])
        p = p_in if same else p_out
        if rng.random() < p:
            out.add((a, b))

    if len(out) < max(1, target_edges // 4):
        out = _synthetic_from_labels(
            data=Data(y=y),
            p_in=max(0.05, p_in),
            p_out=max(0.005, p_out),
            seed=seed + 31,
        )

    return out, {"used_p_in": p_in, "used_p_out": p_out, "same_pair_mass": r_same}


def _synthetic_config_model(
    base_edges: set[tuple[int, int]],
    num_nodes: int,
    target_edges: int,
    seed: int,
) -> set[tuple[int, int]]:
    rng = random.Random(seed)
    deg = [0 for _ in range(num_nodes)]
    for u, v in base_edges:
        deg[u] += 1
        deg[v] += 1

    stubs: list[int] = []
    for i, d in enumerate(deg):
        stubs.extend([i] * d)
    if len(stubs) < 2:
        return set(base_edges)

    rng.shuffle(stubs)
    out: set[tuple[int, int]] = set()
    max_trials = max(1000, len(stubs) * 5)
    trials = 0
    ptr = 0
    while ptr + 1 < len(stubs) and len(out) < target_edges and trials < max_trials:
        trials += 1
        u = stubs[ptr]
        v = stubs[ptr + 1]
        ptr += 2
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        out.add((a, b))

    if len(out) < max(1, target_edges // 4):
        edges_list = list(base_edges)
        rng.shuffle(edges_list)
        out = set(edges_list[:max(1, min(target_edges, len(edges_list)))])

    return out


def apply_graph_condition(data: Data, condition: GraphCondition, seed: int = 42) -> tuple[Data, dict[str, float]]:
    out = data.clone()
    out.edge_attr = None

    y_cpu = out.y.detach().cpu()
    num_nodes = out.num_nodes
    base_edge_index = _make_undirected_simple(out.edge_index.cpu(), num_nodes)
    base_edges = _edge_set_from_undirected(base_edge_index)
    edges = set(base_edges)
    synthetic_meta: dict[str, float] = {}

    if condition.synthetic:
        target_edges = int(condition.synthetic_target_edges or len(base_edges))
        target_edges = max(1, target_edges)
        synthetic_type = str(condition.synthetic_type).lower()
        if synthetic_type == "label_sbm":
            edges = _synthetic_from_labels(
                out,
                p_in=condition.synthetic_p_in,
                p_out=condition.synthetic_p_out,
                seed=seed,
            )
        elif synthetic_type == "dcsbm":
            edges, synthetic_meta = _synthetic_dcsbm(
                y=y_cpu,
                num_nodes=num_nodes,
                target_edges=target_edges,
                degree_gamma=condition.degree_gamma,
                p_in_default=condition.synthetic_p_in,
                p_out_default=condition.synthetic_p_out,
                homophily_target=condition.homophily_target,
                seed=seed,
            )
        elif synthetic_type == "config_model":
            edges = _synthetic_config_model(
                base_edges=base_edges,
                num_nodes=num_nodes,
                target_edges=target_edges,
                seed=seed,
            )
        else:
            raise ValueError(
                f"Unsupported synthetic_type '{condition.synthetic_type}'. "
                "Use one of: label_sbm, dcsbm, config_model"
            )

    skip_homophily_rewire = bool(condition.synthetic and str(condition.synthetic_type).lower() in {"label_sbm", "dcsbm"})
    if condition.homophily_target is not None and not skip_homophily_rewire:
        target = float(max(0.0, min(1.0, condition.homophily_target)))
        edges = _rewire_homophily(edges, y_cpu, target=target, seed=seed + 7)

    edges = _apply_sparsity(edges, keep=condition.sparsity_keep, seed=seed + 13)
    edges = _apply_degree_rewire(edges, y_cpu, gamma=condition.degree_gamma, seed=seed + 17)

    out.edge_index = _edge_index_from_set(edges, num_nodes=num_nodes)

    deg = torch.bincount(out.edge_index[0], minlength=num_nodes).float() if out.edge_index.numel() else torch.zeros(num_nodes)
    stats = {
        "num_nodes": float(num_nodes),
        "num_directed_edges": float(out.edge_index.size(1)),
        "num_undirected_edges": float(len(edges)),
        "edge_density": float((2.0 * len(edges)) / max(1.0, num_nodes * (num_nodes - 1))),
        "edge_homophily": edge_homophily(out.edge_index, out.y),
        "degree_mean": float(deg.mean().item()),
        "degree_std": float(deg.std(unbiased=False).item()),
        "synthetic": float(1.0 if condition.synthetic else 0.0),
    }
    if condition.synthetic:
        stats["synthetic_type"] = float({"label_sbm": 0.0, "dcsbm": 1.0, "config_model": 2.0}.get(str(condition.synthetic_type).lower(), -1.0))
        if "used_p_in" in synthetic_meta:
            stats["used_p_in"] = float(synthetic_meta["used_p_in"])
        if "used_p_out" in synthetic_meta:
            stats["used_p_out"] = float(synthetic_meta["used_p_out"])
        if "same_pair_mass" in synthetic_meta:
            stats["same_pair_mass"] = float(synthetic_meta["same_pair_mass"])
    return out, stats
