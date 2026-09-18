# Architecture source check

Source and implementation checkpoint: 2026-09-18.

## Available reference code

The mode-connectivity paper's Appendix B refers to Luo et al., *Classic GNNs are
Strong Baselines*. That paper links the official
[`LUOyk1999/tunedGNN`](https://github.com/LUOyk1999/tunedGNN) repository.
The inspected main-branch commit was
[`23f9604e8b13a9a6d3faa2f691cd844006979153`](https://github.com/LUOyk1999/tunedGNN/tree/23f9604e8b13a9a6d3faa2f691cd844006979153).
The repository uses the MIT License. The pinned reference files and license are
preserved under [`upstream/tunedGNN`](../upstream/tunedGNN/PROVENANCE.md).
Source paper: [arXiv 2406.08993v2](https://arxiv.org/abs/2406.08993v2).

This is reference architecture and training code, not an implementation of the
mode-connectivity study. A bounded search found no public author repository in
the checked first-party sources. The [official arXiv record](https://arxiv.org/abs/2502.12608)
and [full paper HTML](https://arxiv.org/html/2502.12608) provide no author code
link. Coauthor Haoyu Han's [official publication list](https://hhy.one/) links
only the paper, although nearby entries have separate code links. Exact-title,
arXiv-ID, and author-name GitHub searches also found no matching repository.
This evidence does not prove that no author code exists.

## Architecture and training settings

The reference supports GCN, GraphSAGE, and GAT through PyG operators. The exact
forward order is an optional input linear projection, dropout, then repeated
convolution, optional learned linear residual addition, LayerNorm or BatchNorm,
ReLU, and dropout. Jumping knowledge, when enabled, sums each layer's output
after dropout. A final linear layer produces logits. LayerNorm takes precedence
if both normalization flags are set. The `pre_ln` option allocates and resets
LayerNorm modules but the forward method never applies them.

`reset_parameters()` resets every convolution, residual linear, LayerNorm,
BatchNorm, optional pre-LayerNorm, input projection, and output classifier,
including modules disabled by the selected profile. GCN uses `cached=False` and
`normalize=True`. GAT uses `concat=True`, `add_self_loops=False`, and
`bias=False`. The parser's attention-head default is one, and every target GAT
command uses that default. No MLP implementation exists in this reference. An
MLP formed by substituting linear layers for graph convolutions is therefore an
explicit local design choice rather than copied tunedGNN behavior.

The inspected dataset scripts use substantially different settings from our
two-layer, width-64 GCN. The table transcribes the pinned commands and parser
defaults. All listed GAT profiles use one attention head.

| Dataset | Model | Layers | Width | Epochs | Learning rate | Weight decay | Dropout | Normalization | Residual | Input projection |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| Cora | GCN | 3 | 512 | 500 | .001 | .0005 | .7 | none | no | no |
| Cora | GraphSAGE | 3 | 256 | 500 | .001 | .0005 | .7 | none | no | no |
| Cora | GAT | 3 | 512 | 500 | .001 | .0005 | .2 | none | yes | no |
| Roman-empire | GCN | 9 | 512 | 2,500 | .001 | 0 | .5 | BatchNorm | yes | yes |
| Roman-empire | GraphSAGE | 9 | 256 | 2,500 | .001 | 0 | .3 | BatchNorm | no | yes |
| Roman-empire | GAT | 10 | 512 | 2,500 | .001 | 0 | .3 | BatchNorm | yes | yes |
| squirrel | GCN | 4 | 256 | 500 | .01 | .0005 | .7 | BatchNorm | yes | no |
| squirrel | GraphSAGE | 3 | 256 | 500 | .01 | .0005 | .7 | BatchNorm | yes | no |
| squirrel | GAT | 7 | 512 | 500 | .005 | .0005 | .5 | BatchNorm | yes | no |
| chameleon | GCN | 5 | 512 | 200 | .005 | .001 | .2 | none | no | no |
| chameleon | GraphSAGE | 4 | 256 | 200 | .01 | .001 | .7 | BatchNorm | yes | no |
| chameleon | GAT | 2 | 256 | 200 | .01 | .001 | .7 | BatchNorm | yes | no |

The reference uses Adam and validation-based model selection. Its preprocessing
makes graphs undirected, removes self-loops, then adds one self-loop per node.
Its split policies also differ: Cora uses a seeded class-balanced split, Roman
uses published splits per run, and the filtered Wikipedia datasets use supplied
splits. Our user explicitly requires the existing thesis pipeline. Keep that
pipeline and document deviations; do not silently replace it with the reference
loader or graph preprocessing.

The runner recalibrates BatchNorm statistics separately at each evaluated path
point and selected endpoint, using one full-graph, label-free forward with
dropout disabled. Curve training uses temporary buffers. This policy is an
implementation choice: the mode-connectivity paper does not resolve this detail.
Its exact Bézier training settings also remain unspecified in the inspected
materials. The implementation supports the target profiles' single GAT head;
multiple heads require a separate width convention and are currently rejected.

## Scope for continuation

Use this source to recover justified architecture settings, while keeping the
existing thesis loader and baseline workflow. Preserve the current simple GCN
results as a separate experiment. First run linear/Bézier baselines with the four
model families; defer new REPAIR adapters until the user reevaluates those results.
