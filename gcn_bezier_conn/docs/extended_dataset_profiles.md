# Extended dataset profiles

The `reference` preset now resolves every dataset accepted by the thesis loader
for GCN, MLP, GraphSAGE, and GAT. The original Cora, Roman-empire, squirrel,
and chameleon values remain unchanged.

## Pinned script profiles

Seven added datasets use commands from
[`upstream/tunedGNN/medium_graph/run_gnn.sh`](../upstream/tunedGNN/medium_graph/run_gnn.sh)
at commit `23f9604e8b13a9a6d3faa2f691cd844006979153`. The parser in
[`parse.py`](../upstream/tunedGNN/medium_graph/parse.py) supplies 500 epochs
when a command omits `--epochs`, one GAT head, disabled residual and input
projection flags, and no normalization unless the command enables them.

The thesis loader names `amazon-computer` as `computers` and `amazon-photo` as
`photo`. The preset accepts both forms and records the upstream name in its
source metadata.

| Loader dataset | Model | Width | Depth | Dropout | Norm | Residual | Input projection | Epochs | Learning rate | Weight decay |
|---|---|---:|---:|---:|---|---|---|---:|---:|---:|
| Questions | GCN | 512 | 10 | .3 | none | yes | yes | 1,500 | .00003 | 0 |
| Questions | GraphSAGE | 512 | 6 | .2 | layer | no | yes | 1,500 | .00003 | 0 |
| Questions | GAT | 512 | 3 | .2 | layer | yes | yes | 1,500 | .00003 | 0 |
| computers | GCN | 512 | 3 | .5 | layer | no | no | 1,000 | .001 | .00005 |
| computers | GraphSAGE | 64 | 4 | .3 | layer | no | no | 1,000 | .001 | .00005 |
| computers | GAT | 64 | 2 | .5 | layer | no | no | 1,000 | .001 | .00005 |
| photo | GCN | 256 | 6 | .5 | layer | yes | no | 1,000 | .001 | .00005 |
| photo | GraphSAGE | 64 | 6 | .2 | layer | yes | no | 1,000 | .001 | .00005 |
| photo | GAT | 64 | 3 | .5 | layer | yes | no | 1,000 | .001 | .00005 |
| Citeseer | GCN | 512 | 2 | .5 | none | no | no | 500 | .001 | .01 |
| Citeseer | GraphSAGE | 512 | 3 | .2 | none | no | no | 500 | .001 | .01 |
| Citeseer | GAT | 256 | 3 | .5 | none | yes | no | 500 | .001 | .01 |
| Pubmed | GCN | 256 | 2 | .7 | none | no | no | 500 | .005 | .0005 |
| Pubmed | GraphSAGE | 512 | 4 | .7 | none | no | no | 500 | .005 | .0005 |
| Pubmed | GAT | 512 | 2 | .5 | none | no | no | 500 | .01 | .0005 |
| Amazon-ratings | GCN | 512 | 4 | .5 | batch | yes | no | 2,500 | .001 | 0 |
| Amazon-ratings | GraphSAGE | 512 | 9 | .5 | batch | yes | no | 2,500 | .001 | 0 |
| Amazon-ratings | GAT | 512 | 4 | .5 | batch | yes | no | 2,500 | .001 | 0 |
| Minesweeper | GCN | 64 | 12 | .2 | batch | yes | no | 2,000 | .01 | 0 |
| Minesweeper | GraphSAGE | 64 | 15 | .2 | batch | yes | no | 2,000 | .01 | 0 |
| Minesweeper | GAT | 64 | 15 | .2 | batch | yes | no | 2,000 | .01 | 0 |

These profiles select endpoints by validation accuracy. Questions and
Minesweeper are binary tasks, and the pinned commands request ROC AUC. The local
runner instead trains two logits with cross entropy and selects by validation
accuracy. The preset retains that runner behavior. It does not reproduce the
pinned script's ROC AUC evaluation.

The pinned repository has no MLP recipe. Each MLP profile explicitly inherits
the same dataset's GCN width, depth, dropout, normalization, residual, input
projection, and training settings. Its source metadata records this adaptation.

## Local fallback profiles

The pinned script has no commands for actor, texas, cornell, or Tolokers. These
four datasets use the following local choice for each selected operator:

| Datasets | Models | Width | Depth | Dropout | Norm | Residual | Input projection | Heads | Epochs | Learning rate | Weight decay | Selection |
|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|
| actor, texas, cornell, Tolokers | GCN, GraphSAGE, GAT | 64 | 2 | .5 | none | no | no | 1 | 200 | .01 | .0005 | validation loss |

This is the earlier compact GCN budget applied to the requested operator. It is
not derived from tunedGNN. Returned source metadata sets `local_fallback` to
`true` and `profile_origin` to `local_fallback`; pinned script profiles set
`local_fallback` to `false`. MLP again inherits the fallback GCN profile and
records that adaptation.
