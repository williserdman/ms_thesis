from pathlib import Path

import matplotlib.pyplot as plt


def _extract(metrics: dict[float, dict[str, float]]):
    alphas = list(metrics.keys())
    train_losses = [metrics[a]["train_loss"] for a in alphas]
    val_losses = [metrics[a]["val_loss"] for a in alphas]
    test_losses = [metrics[a]["test_loss"] for a in alphas]
    train_accs = [metrics[a]["train_acc"] for a in alphas]
    val_accs = [metrics[a]["val_acc"] for a in alphas]
    test_accs = [metrics[a]["test_acc"] for a in alphas]
    return alphas, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs


def plot_single_path(metrics: dict[float, dict[str, float]], title: str, output_path: Path) -> None:
    alphas, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = _extract(metrics)

    fig, ax_loss = plt.subplots(figsize=(10, 6))
    ax_acc = ax_loss.twinx()

    ax_loss.plot(alphas, train_losses, label="Train Loss", color="tab:green")
    ax_loss.plot(alphas, val_losses, label="Val Loss", color="tab:orange")
    ax_loss.plot(alphas, test_losses, label="Test Loss", color="tab:red")

    ax_acc.plot(alphas, train_accs, label="Train Acc", color="tab:blue", linestyle="--")
    ax_acc.plot(alphas, val_accs, label="Val Acc", color="tab:purple", linestyle="--")
    ax_acc.plot(alphas, test_accs, label="Test Acc", color="tab:brown", linestyle="--")

    ax_loss.set_xlabel("Alpha")
    ax_loss.set_ylabel("Loss")
    ax_acc.set_ylabel("Accuracy")
    ax_loss.set_title(title)

    handles1, labels1 = ax_loss.get_legend_handles_labels()
    handles2, labels2 = ax_acc.get_legend_handles_labels()
    ax_loss.legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_linear_vs_bezier(
    linear_metrics: dict[float, dict[str, float]],
    bezier_metrics: dict[float, dict[str, float]],
    title: str,
    output_path: Path,
) -> None:
    l_alphas, l_train_losses, _, l_test_losses, l_train_accs, _, l_test_accs = _extract(linear_metrics)
    b_alphas, b_train_losses, _, b_test_losses, b_train_accs, _, b_test_accs = _extract(bezier_metrics)

    fig, ax_loss = plt.subplots(figsize=(10, 6))
    ax_acc = ax_loss.twinx()

    ax_loss.plot(l_alphas, l_train_losses, label="Linear Train Loss", color="tab:green")
    ax_loss.plot(l_alphas, l_test_losses, label="Linear Test Loss", color="tab:red")
    ax_acc.plot(l_alphas, l_train_accs, label="Linear Train Acc", color="tab:blue")
    ax_acc.plot(l_alphas, l_test_accs, label="Linear Test Acc", color="tab:orange")

    ax_loss.plot(b_alphas, b_train_losses, label="Bezier Train Loss", color="tab:green", linestyle="--")
    ax_loss.plot(b_alphas, b_test_losses, label="Bezier Test Loss", color="tab:red", linestyle="--")
    ax_acc.plot(b_alphas, b_train_accs, label="Bezier Train Acc", color="tab:blue", linestyle="--")
    ax_acc.plot(b_alphas, b_test_accs, label="Bezier Test Acc", color="tab:orange", linestyle="--")

    ax_loss.set_xlabel("Alpha")
    ax_loss.set_ylabel("Loss")
    ax_acc.set_ylabel("Accuracy")
    ax_loss.set_title(title)

    handles1, labels1 = ax_loss.get_legend_handles_labels()
    handles2, labels2 = ax_acc.get_legend_handles_labels()
    ax_loss.legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
