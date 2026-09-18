# Replication protocol for Li et al. (2025)

This note separates facts in *Unveiling Mode Connectivity in Graph Neural Networks* from choices required for a runnable replication. The source inspected was arXiv:2502.12608v1, submitted 18 February 2025. The paper reports a KDD 2025 study, but the arXiv manuscript still contains an unfilled conference header and sparse implementation appendix. Sources: [arXiv record](https://arxiv.org/abs/2502.12608v1), [HTML paper](https://arxiv.org/html/2502.12608), [submitted source](https://export.arxiv.org/e-print/2502.12608v1).

## Core paper protocol

The task is transductive node classification on one graph. The main backbone is GCN. The architecture comparison replaces GCN with MLP, GraphSAGE, and GAT. Two modes, \(\theta_a\) and \(\theta_b\), come from independent training configurations. The study holds all hyperparameters but one fixed and varies initialization, data order, or model choice. It repeats experiments three times with different random seeds and reports the mean. The paper does not list the seed values or define which result constitutes one repeat. [Section 2.3](https://arxiv.org/html/2502.12608#S2.SS3), [Section 3.1](https://arxiv.org/html/2502.12608#S3.SS1).

For interpolation coefficient \(\alpha\in[0,1]\), the linear path is

\[
\phi_{\mathrm{lin}}(\alpha)=(1-\alpha)\theta_a+\alpha\theta_b.
\]

The quadratic Bézier path is

\[
\phi_{\mathrm{bez}}(\alpha)=(1-\alpha)^2\theta_a
+2\alpha(1-\alpha)\theta_c+\alpha^2\theta_b,
\]

where the paper calls the control point \(\theta\); this note uses \(\theta_c\) to distinguish it from the full model parameters. The endpoints remain \(\theta_a\) and \(\theta_b\), and \(\theta_c\) is learned. The manuscript does not state its initialization, optimizer, learning rate, objective estimator, number of updates, or \(\alpha\)-sampling rule. [Equations 7 and 8](https://arxiv.org/html/2502.12608#S2.SS3).

For a chosen evaluation split, the paper defines the loss barrier as

\[
B(\theta_a,\theta_b)=\max_{\alpha\in[0,1]}
\left[\mathcal L(\phi(\alpha))-
\left((1-\alpha)\mathcal L(\theta_a)+\alpha\mathcal L(\theta_b)\right)\right].
\]

This definition applies to either path once \(\phi\) is selected. A sampled implementation estimates the maximum on an \(\alpha\) grid. The paper reports train and test cross-entropy loss and accuracy along each path. Its equations write cross-entropy as a sum over labeled nodes, not a mean. [Equations 2 to 5](https://arxiv.org/html/2502.12608#S2.SS1), [Definition 2.2](https://arxiv.org/html/2502.12608#S2.SS3).

The paper also discusses a "training accuracy barrier" for generalization analysis, but it does not give a formula or sign convention for an accuracy barrier. Do not treat \(B\) above as an accuracy metric. [Section 4.1](https://arxiv.org/html/2502.12608#S4.SS1).

## Data and experiment scope

The paper says it uses 12 graphs, while Appendix A plots 13 named graphs:

| Domain | Datasets named in figures and Appendix C |
|---|---|
| Citation | Cora, CiteSeer, PubMed |
| Amazon | Amazon-Computer, Amazon-Photo |
| Coauthor | Coauthor-CS, Coauthor-Physics |
| WikiCS | WikiCS |
| Wikipedia | Squirrel, Chameleon |
| Heterophilous benchmark | Roman-Empire, Amazon-Ratings, Minesweeper |

The 12-versus-13 discrepancy is internal to the manuscript. Appendix C describes the families but gives no download source, preprocessing, mask construction, split ratios, or selected split IDs. [Appendix A](https://arxiv.org/html/2502.12608#A1), [Appendix C](https://arxiv.org/html/2502.12608#A3).

The full paper scope contains five blocks:

1. Linear and quadratic Bézier paths for GCN modes on the real graphs.
2. MLP, GCN, GraphSAGE, and GAT comparison.
3. CSBM sweeps over density, homophily, feature separability, and graph size.
4. Correlation between mode-connectivity quantities and generalization.
5. Cross-domain distance based on Wasserstein-1 distance between loss curves, followed by domain-adaptation experiments.

The implementation covers the first two blocks' baseline procedures on the four
selected thesis datasets. The remaining blocks require additional definitions,
datasets, or baselines. [Sections 3 and 4](https://arxiv.org/html/2502.12608#S3).

## Missing replication details

Appendix B only says the authors follow the architecture and hyperparameters of Luo et al., *Classic GNNs are Strong Baselines*, and lists four GPU models. It does not reproduce those settings. Neither the arXiv record, manuscript, nor submitted TeX source links an author code repository. No official author implementation was identified. [Appendix B](https://arxiv.org/html/2502.12608#A2), [arXiv record](https://arxiv.org/abs/2502.12608v1).

Update, 2026-09-18: the cited baseline paper does provide official `tunedGNN`
code. The [source-check note](architecture_sources.md) records the pinned source
and exact profiles. `--preset reference` now incorporates those architecture and
endpoint training settings for GCN, GraphSAGE, and GAT. The MLP substitutes linear
blocks into the GCN profile. These settings are recoverable from the cited
baseline, but have not been confirmed as the exact configuration used by the
mode-connectivity authors. The thesis data pipeline is retained.

The following choices are therefore unknown from the paper:

- exact GCN depth, hidden width, activation, dropout, normalization, biases, and adjacency normalization;
- optimizer, learning rate, weight decay, scheduler, epoch count, early stopping, and checkpoint selection;
- split ratios, mask versions, minibatching or full-batch training, and the meaning of "data order" for each dataset;
- seed values, number of endpoints per repeat, and endpoint pairing;
- Bézier control initialization and training procedure;
- interpolation grid size and whether barriers use train or test loss;
- whether dropout is active while fitting the control point;
- loss reduction. The displayed loss is a sum, which makes barriers depend on split size;
- treatment of non-parameter buffers and normalization statistics during interpolation.

There is also an architecture notation ambiguity. Equation 1 applies \(W^{(l)}\) in every propagation layer, then the text defines final logits as \(H^{(L)}W^{(L)}\), apparently reusing the last-layer symbol. The number of graph convolutions cannot be recovered from this notation alone. [Section 2.1](https://arxiv.org/html/2502.12608#S2.SS1).

## Legacy GCN defaults and shared path choices

These are implementation defaults, not claims about the authors' setup. Reference
presets replace the model and endpoint training rows with the settings in the
[architecture source note](architecture_sources.md). Reference BatchNorm models
use fresh, label-free full-graph statistics at each evaluated point and endpoint;
temporary buffers isolate curve training. This normalization policy is a local
choice, separate from REPAIR.

| Item | MVP choice |
|---|---|
| Data | Use the thesis loader, preprocessing, and existing train, validation, and test masks. Save mask hashes and counts. |
| Model | Two-layer GCN, hidden width 64, ReLU, dropout 0.5. Preserve thesis features and edges; PyG GCNConv adds self-loops and symmetric adjacency normalization. |
| Endpoint training | Full-batch Adam, learning rate 0.01, weight decay \(5\times10^{-4}\), at most 200 epochs. Select the checkpoint with minimum validation loss. Never select on test metrics. |
| Endpoint variation | Keep graph, masks, architecture, and hyperparameters fixed. Change only the model/training seed. Use seeds 0 and 1 for the first pair. |
| Endpoint repeats | Default: three independent pairs (0,1), (2,3), and (4,5), then report the mean and population standard deviation. The smoke check uses only the first pair. |
| Bézier control | Initialize every control tensor to \((\theta_a+\theta_b)/2\). Freeze both endpoints. Optimize only \(\theta_c\) with full-batch training cross-entropy. |
| Curve optimization | Sample one \(\alpha\sim U(0,1)\) per step. Use Adam with learning rate 0.01, no weight decay, for 200 steps. Keep dropout active for the training objective and disable it for evaluation. Save the curve seed. Increase `--curve-epochs` for a convergence study. |
| Evaluation | Evaluate both paths at 21 equally spaced points including 0 and 1. Record mean cross-entropy and accuracy on train, validation, and test masks. Increase `--points` to check grid resolution. |
| Barrier | Compute the sampled barrier separately for each split and path with the displayed formula. Record the maximizing grid point. Use mean loss consistently so results are comparable across masks. |
| Parameters and buffers | Interpolate all floating trainable parameters with matching names and shapes. Keep non-parameter buffers fixed. The default GCN should avoid running-statistic buffers. |

The curve objective is the Monte Carlo approximation

\[
\min_{\theta_c}\;\mathbb E_{\alpha\sim U(0,1)}
\left[\mathcal L_{\mathrm{train}}\left(\phi_{\mathrm{bez}}(\alpha)\right)\right].
\]

This is a standard concrete interpretation of "learnable" in Equation 8, but the paper does not specify it. Differentiable parameter substitution must preserve the gradient from the training loss to every control tensor. Endpoint weights must remain byte-for-byte unchanged during curve fitting.

## Minimum reported result

For each endpoint and each sampled point on both paths, save the seed, split identity, \(\alpha\), train/validation/test loss, and train/validation/test accuracy. Summaries should include endpoint metrics, maximum loss barrier per split, its \(\alpha\), and the minimum accuracy along each path. Save raw values before plotting.

A result can be described as a replication of the paper's core procedure only with the data-pipeline deviation stated explicitly: it reuses the thesis splits and preprocessing because the paper does not publish exact masks or preprocessing. It should be described as a workflow validation until three endpoint pairs and the paper's broader graph set are run.
