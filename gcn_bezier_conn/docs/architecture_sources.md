# Architecture source check

Research checkpoint: 2026-09-18. Implementation has not started.

## Available reference code

The mode-connectivity paper's Appendix B refers to Luo et al., *Classic GNNs are
Strong Baselines*. That paper links the official
[`LUOyk1999/tunedGNN`](https://github.com/LUOyk1999/tunedGNN) repository.
The inspected main-branch commit was
[`23f9604e8b13a9a6d3faa2f691cd844006979153`](https://github.com/LUOyk1999/tunedGNN/tree/23f9604e8b13a9a6d3faa2f691cd844006979153).
The repository reports an MIT license. Verify and preserve the license when
copying source. Source paper: [arXiv 2406.08993v2](https://arxiv.org/abs/2406.08993v2).

This is reference architecture/training code, not an identified implementation
of the mode-connectivity study. The earlier search found no official
mode-connectivity implementation; the fresh search was interrupted for handoff
before that question was resolved. Do not turn this into a claim that no author
code exists.

## Findings to use in the next design

The reference supports GCN, GraphSAGE, and GAT through PyG operators. Its models
can include input projection, learned residual projections, BatchNorm or
LayerNorm, ReLU, dropout, and a final linear classifier. Its GAT default uses one
head, disables convolution-added self-loops, and disables convolution bias.
No MLP implementation was identified in this reference repository.

The inspected dataset scripts use substantially different settings from our
two-layer, width-64 GCN. These are research notes from the script inspection;
recheck the pinned scripts and parser defaults before encoding configurations:

| Dataset | Model | Layers | Width | Learning rate | Weight decay | Dropout | Other settings |
|---|---|---:|---:|---:|---:|---:|---|
| Cora | GCN | 3 | 512 | .001 | .0005 | .7 | 5 runs |
| Cora | GraphSAGE | 3 | 256 | .001 | .0005 | .7 | 5 runs |
| Cora | GAT | 3 | 512 | .001 | .0005 | .2 | residual; 5 runs |
| Roman-empire | GCN | 9 | 512 | .001 | 0 | .5 | BatchNorm, residual, input projection; 2,500 epochs |
| Roman-empire | GraphSAGE | 9 | 256 | .001 | 0 | .3 | BatchNorm, input projection; 2,500 epochs |
| Roman-empire | GAT | 10 | 512 | .001 | 0 | .3 | BatchNorm, residual, input projection; 2,500 epochs |
| squirrel | GCN | 4 | 256 | .01 | .0005 | .7 | BatchNorm, residual |
| squirrel | GraphSAGE | 3 | 256 | .01 | .0005 | .7 | BatchNorm, residual |
| squirrel | GAT | 7 | 512 | .005 | .0005 | .5 | BatchNorm, residual |
| chameleon | GCN | 5 | 512 | .005 | .001 | .2 | no normalization or residual; 200 epochs |
| chameleon | GraphSAGE | 4 | 256 | .01 | .001 | .7 | BatchNorm, residual; 200 epochs |
| chameleon | GAT | 2 | 256 | .01 | .001 | .7 | BatchNorm, residual; 200 epochs |

The reference uses Adam and validation-based model selection. Its preprocessing
makes graphs undirected, removes self-loops, then adds one self-loop per node.
Its split policies also differ: Cora uses a seeded class-balanced split, Roman
uses published splits per run, and the filtered Wikipedia datasets use supplied
splits. Our user explicitly requires the existing thesis pipeline. Keep that
pipeline and document deviations; do not silently replace it with the reference
loader or graph preprocessing.

Before using these models in Bézier paths, define treatment of normalization
buffers and statistics. The present plain GCN has no running-statistic buffers;
adding BatchNorm is not a model-class substitution alone. The mode-connectivity
paper does not resolve this detail. Its exact Bézier training settings also
remain unspecified in the materials inspected so far.

## Scope for continuation

Use this source to recover justified architecture settings, while keeping the
existing thesis loader and baseline workflow. Preserve the current simple GCN
results as a separate experiment. First run linear/Bézier baselines with the four
model families; defer new REPAIR adapters until the user reevaluates those results.
