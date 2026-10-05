# Six-method GNN connectivity matrix

Audit status: **complete**. 60 of 60 configurations passed; 60 reports were available.

The intended matrix mixes 16 reused tuned configurations, 28 pinned source-profile configurations, and 16 explicit local-fallback configurations. These groups must be reported separately because their endpoint budgets and provenance differ.

Methods: raw linear, aligned linear, aligned linear + REPAIR, raw Bézier, aligned Bézier, and aligned Bézier + REPAIR. Repaired paths are post-hoc activation corrections and are not straight or quadratic parameter-space paths.

## Overview figures

- GCN: [test loss](gcn_test_loss_overview.png), [test accuracy](gcn_test_accuracy_overview.png)
- MLP: [test loss](mlp_test_loss_overview.png), [test accuracy](mlp_test_accuracy_overview.png)
- GraphSAGE: [test loss](graphsage_test_loss_overview.png), [test accuracy](graphsage_test_accuracy_overview.png)
- GAT: [test loss](gat_test_loss_overview.png), [test accuracy](gat_test_accuracy_overview.png)

## Provenance groups

| Group | Intended | Available | Passed |
|---|---:|---:|---:|
| Reused tuned endpoints | 16 | 16 | 16 |
| Pinned source profiles | 28 | 28 | 28 |
| Local fallback profiles | 16 | 16 | 16 |

## Configuration audit

| Dataset | Model | Group | Status | Errors |
|---|---|---|---|---|
| Cora | GCN | Reused tuned endpoints | passed |  |
| Cora | MLP | Reused tuned endpoints | passed |  |
| Cora | GraphSAGE | Reused tuned endpoints | passed |  |
| Cora | GAT | Reused tuned endpoints | passed |  |
| Roman-empire | GCN | Reused tuned endpoints | passed |  |
| Roman-empire | MLP | Reused tuned endpoints | passed |  |
| Roman-empire | GraphSAGE | Reused tuned endpoints | passed |  |
| Roman-empire | GAT | Reused tuned endpoints | passed |  |
| squirrel | GCN | Reused tuned endpoints | passed |  |
| squirrel | MLP | Reused tuned endpoints | passed |  |
| squirrel | GraphSAGE | Reused tuned endpoints | passed |  |
| squirrel | GAT | Reused tuned endpoints | passed |  |
| chameleon | GCN | Reused tuned endpoints | passed |  |
| chameleon | MLP | Reused tuned endpoints | passed |  |
| chameleon | GraphSAGE | Reused tuned endpoints | passed |  |
| chameleon | GAT | Reused tuned endpoints | passed |  |
| Questions | GCN | Pinned source profiles | passed |  |
| Questions | MLP | Pinned source profiles | passed |  |
| Questions | GraphSAGE | Pinned source profiles | passed |  |
| Questions | GAT | Pinned source profiles | passed |  |
| computers | GCN | Pinned source profiles | passed |  |
| computers | MLP | Pinned source profiles | passed |  |
| computers | GraphSAGE | Pinned source profiles | passed |  |
| computers | GAT | Pinned source profiles | passed |  |
| photo | GCN | Pinned source profiles | passed |  |
| photo | MLP | Pinned source profiles | passed |  |
| photo | GraphSAGE | Pinned source profiles | passed |  |
| photo | GAT | Pinned source profiles | passed |  |
| Citeseer | GCN | Pinned source profiles | passed |  |
| Citeseer | MLP | Pinned source profiles | passed |  |
| Citeseer | GraphSAGE | Pinned source profiles | passed |  |
| Citeseer | GAT | Pinned source profiles | passed |  |
| Pubmed | GCN | Pinned source profiles | passed |  |
| Pubmed | MLP | Pinned source profiles | passed |  |
| Pubmed | GraphSAGE | Pinned source profiles | passed |  |
| Pubmed | GAT | Pinned source profiles | passed |  |
| actor | GCN | Local fallback profiles | passed |  |
| actor | MLP | Local fallback profiles | passed |  |
| actor | GraphSAGE | Local fallback profiles | passed |  |
| actor | GAT | Local fallback profiles | passed |  |
| texas | GCN | Local fallback profiles | passed |  |
| texas | MLP | Local fallback profiles | passed |  |
| texas | GraphSAGE | Local fallback profiles | passed |  |
| texas | GAT | Local fallback profiles | passed |  |
| cornell | GCN | Local fallback profiles | passed |  |
| cornell | MLP | Local fallback profiles | passed |  |
| cornell | GraphSAGE | Local fallback profiles | passed |  |
| cornell | GAT | Local fallback profiles | passed |  |
| Amazon-ratings | GCN | Pinned source profiles | passed |  |
| Amazon-ratings | MLP | Pinned source profiles | passed |  |
| Amazon-ratings | GraphSAGE | Pinned source profiles | passed |  |
| Amazon-ratings | GAT | Pinned source profiles | passed |  |
| Minesweeper | GCN | Pinned source profiles | passed |  |
| Minesweeper | MLP | Pinned source profiles | passed |  |
| Minesweeper | GraphSAGE | Pinned source profiles | passed |  |
| Minesweeper | GAT | Pinned source profiles | passed |  |
| Tolokers | GCN | Local fallback profiles | passed |  |
| Tolokers | MLP | Local fallback profiles | passed |  |
| Tolokers | GraphSAGE | Local fallback profiles | passed |  |
| Tolokers | GAT | Local fallback profiles | passed |  |

Each available configuration folder contains a copied report plus PNG and PDF plots. Checkpoint weights remain under the run root. The audit checks their existence, source identity, all path metrics, and both repaired midpoint replays.
