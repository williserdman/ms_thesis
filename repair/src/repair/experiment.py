"""Training and interpolation experiments derived from the upstream VGG notebook."""

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .alignment import align_models, hidden_layer_names
from .core import interpolate, repair
from .models import tiny_mlp, vgg11


@torch.no_grad()
def evaluate(model, data):
    """Return sample-averaged cross entropy and accuracy without changing model modes."""
    modes = {module: module.training for module in model.modules()}
    model.eval()
    device = next(model.parameters()).device
    total_loss, correct, count = 0.0, 0, 0
    try:
        for inputs, labels in data:
            logits = model(inputs.to(device))
            labels = labels.to(device)
            total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.numel()
    finally:
        for module, training in modes.items():
            module.training = training
    if not count:
        raise ValueError("Evaluation data is empty")
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": count}


def _train(model, data, epochs, *, cifar=False):
    device = next(model.parameters()).device
    if cifar:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9, weight_decay=5e-4)
        # Upstream schedule: five-epoch warmup followed by linear decay to zero.
        steps = epochs * len(data)
        warmup = min(5 * len(data), max(1, steps // 2)) if epochs <= 5 else 5 * len(data)

        def schedule(step):
            if step < warmup:
                return step / warmup
            return max(0.0, (steps - step) / max(1, steps - warmup))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = None
    epoch_seconds = []
    for epoch in range(epochs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        model.train()
        for inputs, labels in data:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds.append(time.perf_counter() - epoch_started)
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/{epochs}, {epoch_seconds[-1]:.2f}s", flush=True)
    return model.eval(), epoch_seconds


def _synthetic_loaders(args):
    generator = torch.Generator().manual_seed(args.seed)
    teacher = torch.randn(16, 4, generator=generator)

    def dataset(size):
        inputs = torch.randn(size, 16, generator=generator)
        labels = (inputs @ teacher).argmax(1)
        return TensorDataset(inputs, labels)

    train = dataset(args.samples)
    test = dataset(max(128, args.samples // 2))
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True),
        DataLoader(train, batch_size=args.batch_size),
        DataLoader(test, batch_size=args.batch_size),
    )


def _cifar_loaders(args):
    try:
        from torchvision import datasets, transforms as T
    except ImportError as error:
        raise RuntimeError("Install the CIFAR extra: python -m pip install -e '.[cifar]'") from error
    normalize = T.Normalize(
        [125.307 / 255, 122.961 / 255, 113.8575 / 255],
        [51.5865 / 255, 50.847 / 255, 51.255 / 255],
    )
    train_transform = T.Compose([
        T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize,
    ])
    test_transform = T.Compose([T.ToTensor(), normalize])
    train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    # Fixed training images let every endpoint and sequential pass see identical data.
    calibration = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=test_transform)
    test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        # Calibration revisits this loader many times; keep its workers alive.
        "persistent_workers": args.num_workers > 0,
    }
    return (
        DataLoader(train, shuffle=True, **loader_options),
        DataLoader(calibration, shuffle=False, **loader_options),
        DataLoader(test, shuffle=False, **loader_options),
    )


def _save_state(model, path):
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, path)


def _barriers(curve, endpoints):
    result = {}
    for variant in ("unaligned", "aligned", "repaired"):
        losses, errors = [], []
        for row in curve:
            alpha = row["alpha"]
            baseline_loss = (1 - alpha) * endpoints[0]["loss"] + alpha * endpoints[1]["loss"]
            baseline_accuracy = (1 - alpha) * endpoints[0]["accuracy"] + alpha * endpoints[1]["accuracy"]
            losses.append(row[variant]["loss"] - baseline_loss)
            errors.append(baseline_accuracy - row[variant]["accuracy"])
        result[variant] = {
            "loss_barrier": max(0.0, max(losses)),
            "error_barrier": max(0.0, max(errors)),
        }
    return result


def _parser():
    parser = argparse.ArgumentParser(description="Train, align, interpolate, and REPAIR two classifiers")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("demo", "cifar10"):
        child = commands.add_parser(command)
        child.add_argument("--epochs", type=int, default=20 if command == "demo" else 100)
        child.add_argument("--batch-size", type=int, default=64 if command == "demo" else 500)
        child.add_argument("--seed", type=int, default=0)
        child.add_argument("--device", default="auto")
        child.add_argument("--threads", type=int, default=2)
        child.add_argument("--alphas", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
        child.add_argument("--method", choices=["batchnorm", "sequential"], default="batchnorm")
        child.add_argument("--calibration-batches", type=int, default=None if command == "demo" else 10)
        child.add_argument("--output", type=Path, default=Path("runs") / command)
        if command == "demo":
            child.add_argument("--samples", type=int, default=512)
        else:
            child.add_argument("--data-dir", type=Path, default=Path("data"))
            child.add_argument("--width", type=float, default=1)
            child.add_argument("--num-workers", type=int, default=0)
            child.add_argument("--checkpoint-a", type=Path)
            child.add_argument("--checkpoint-b", type=Path)
    return parser


def main(argv=None):
    experiment_started = time.perf_counter()
    parser = _parser()
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size < 2 or args.threads <= 0:
        parser.error("epochs and threads must be positive; batch-size must be at least 2")
    if args.calibration_batches is not None and args.calibration_batches <= 0:
        parser.error("calibration-batches must be positive")
    if any(not math.isfinite(a) or not 0 <= a <= 1 for a in args.alphas):
        parser.error("alphas must be finite values in [0, 1]")
    if args.command == "demo" and (args.samples < 2 or args.samples % args.batch_size == 1):
        parser.error("samples must be at least 2 and must not leave a one-sample calibration batch")
    output = args.output
    artifact_names = ["endpoint_a.pt", "endpoint_b.pt", "aligned_b.pt", "repaired_midpoint.pt", "report.json"]
    if any((output / name).exists() for name in artifact_names):
        parser.error("Output contains experiment artifacts; choose a new --output directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cifar = args.command == "cifar10"
    train, calibration, test = _cifar_loaders(args) if cifar else _synthetic_loaders(args)
    factory = (lambda: vgg11(width=args.width)) if cifar else tiny_mlp
    output.mkdir(parents=True, exist_ok=True)
    models = []
    timings = {"training": {}}
    for index, key in enumerate(("a", "b")):
        torch.manual_seed(args.seed + index)
        model = factory().to(device)
        checkpoint = getattr(args, f"checkpoint_{key}", None)
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval()
        else:
            print(f"Training endpoint {key}", flush=True)
            model, epoch_seconds = _train(model, train, args.epochs, cifar=cifar)
            timings["training"][key] = {
                "total_seconds": sum(epoch_seconds), "epoch_seconds": epoch_seconds,
            }
        models.append(model)
        _save_state(model, output / f"endpoint_{key}.pt")
    model_a, model_b = models
    print("Aligning channels", flush=True)
    alignment_started = time.perf_counter()
    aligned = align_models(model_a, model_b, calibration, max_batches=args.calibration_batches)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["alignment_seconds"] = time.perf_counter() - alignment_started
    _save_state(aligned, output / "aligned_b.pt")
    endpoints = [evaluate(model, test) for model in models]
    layer_names = hidden_layer_names(model_a)
    curve = []
    curve_started = time.perf_counter()
    for alpha in sorted(set(args.alphas)):
        repaired = repair(
            model_a, aligned, calibration, alpha, layer_names=layer_names,
            max_batches=args.calibration_batches, method=args.method,
        )
        row = {
            "alpha": alpha,
            "unaligned": evaluate(interpolate(model_a, model_b, alpha), test),
            "aligned": evaluate(interpolate(model_a, aligned, alpha), test),
            "repaired": evaluate(repaired, test),
        }
        curve.append(row)
        print(json.dumps(row), flush=True)
        if alpha == 0.5:
            _save_state(repaired, output / "repaired_midpoint.pt")
    if 0.5 not in args.alphas:
        midpoint = repair(model_a, aligned, calibration, 0.5, layer_names=layer_names,
                          max_batches=args.calibration_batches, method=args.method)
        _save_state(midpoint, output / "repaired_midpoint.pt")
    timings["curve_seconds"] = time.perf_counter() - curve_started
    timings["total_seconds"] = time.perf_counter() - experiment_started
    report = {
        "schema_version": 1,
        "dataset": "cifar10" if cifar else "synthetic",
        "architecture": {"name": "vgg11", "width": args.width, "num_classes": 10} if cifar else {
            "name": "tiny_mlp", "input_dim": 16, "hidden_dims": [32, 32], "num_classes": 4,
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "timings": timings,
        "method": args.method,
        "layer_names": layer_names,
        "calibration_batches": args.calibration_batches,
        "endpoints": endpoints,
        "curve": curve,
        "barriers": _barriers(curve, endpoints),
        "note": "Barriers are maxima over sampled coefficients. One endpoint pair; no uncertainty estimate.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
