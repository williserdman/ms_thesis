# Adding an architecture

Add one explicit adapter for each supported model family. Matching, curve
fitting, and REPAIR consume the `ArchitectureAdapter` protocol in
[adapters/base.py](../src/rebasin_bezier_repair/adapters/base.py); they do not need architecture branches or a registry.

Use `MLPAdapter` and `GCNAdapter` as the working references. An adapter has four
responsibilities.

## 1. Describe every state axis

`permutation_spec(model, state)` returns one tuple for every parameter and
buffer in `state`. Each tuple has one entry per tensor axis. Use the same hidden
group name wherever two axes must receive the same permutation. Use `None` for
fixed axes such as input features, output classes, scalar values, and fixed
buffers.

Call `git_re_basin.validate_spec(spec, state)` before returning. Reject model
structures whose coupled axes are unknown. Residual paths, attention, tied
weights, normalization, and cached graph layers need architecture-specific
rules rather than guesses based on module names.

## 2. Execute a supplied state

`forward(model, state, batch)` returns logits from the supplied complete state.
Use `torch.func.functional_call(..., strict=True)` so missing or extra state
fails immediately. This method also owns batch unpacking. For example, the MLP
extracts an input tensor, while the GCN extracts node features and edge indices.

## 3. Name preactivation repair sites

`repair_sites(model)` returns `RepairSite` objects in forward order. Each site
identifies a module output and its channel axis. Exclude the classifier. REPAIR
hooks the named module output, moves the channel axis last, and pools population
moments over all other axes.

Keep observation selection outside the adapter. A caller-provided
`ObservationSelector` receives `(site, output, batch)` before pooling. It can
select training nodes in a full-graph GCN batch or selected examples without
putting dataset splits into architecture code. Selection must preserve the
output rank and channel axis.

## 4. Fold a correction into state

`apply_correction(state, site, scale, shift)` returns a copied state with the
channelwise affine correction folded into the parameters that produce that
site. For a biased affine layer, use `correct_affine_rows`:

```text
W' = scale * W
b' = scale * b + shift
```

Override this mapping only when the architecture has another valid way to
apply the correction. Native BatchNorm is outside the current contract. Fixed
buffers must match at both aligned endpoints and remain unchanged.

## Start from a working adapter

Copy the closest implementation and change its four methods:

- [MLPAdapter](../src/rebasin_bezier_repair/adapters/mlp.py) is the template for
  biased affine layers with ReLU activations.
- [GCNAdapter](../src/rebasin_bezier_repair/adapters/gcn.py) shows nested weight
  keys, complete graph-layer hooks, and graph batch unpacking.

Keep the shared pipeline unchanged. Add focused tests alongside
[test_adapters.py](../tests/test_adapters.py) or
[test_gcn.py](../tests/test_gcn.py), then add a small runnable example.

Keep optional dependencies local to their adapter module. In particular, an
adapter that imports PyTorch Geometric must not be imported from the package
root. Core and MLP imports must continue to work without PyG installed.

## Completion checks

A new adapter is complete when focused tests establish all of these contracts:

- The permutation spec covers the complete state, and applying a hidden-channel
  permutation preserves endpoint logits within the configured tolerance.
- Curve fitting produces a nonzero control gradient, updates the control, and
  leaves caller-owned endpoint state, gradients, and training flags unchanged.
- The selector measures the intended observations. Sequential REPAIR reaches
  the requested nondegenerate population means and standard deviations.
- A materialized repaired model can load through the architecture's ordinary
  `state_dict` path and replay the same predictions.
- Unsupported structures and unequal fixed buffers fail with clear errors.

REPAIR runs separately for each requested interior sample and does not
differentiate through correction. Once corrected, the family is generally no
longer an exact quadratic parameter path. Keep broader benchmarks, a registry,
a command-line interface, and general plugin machinery out of an adapter change.
