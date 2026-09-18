"""Command line entry point; run from the gcn_bezier_conn directory."""

from __future__ import annotations

import argparse

from .experiment import Config, resolved_config, run


def seed_pair(value):
    try:
        a, b = (int(s) for s in value.split(":"))
        if a == b or min(a, b) < 0:
            raise ValueError
        return a, b
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use two different nonnegative seeds, e.g. 0:1") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description="GNN mode connectivity baselines using the existing thesis loader.")
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="Train endpoints, fit Bézier controls, and save path metrics.")
    runner.add_argument("--datasets", nargs="+", default=["Cora"])
    runner.add_argument("--pairs", nargs="+", type=seed_pair, default=[(0, 1), (2, 3), (4, 5)], help="Independent endpoint seed pairs; default: 0:1 2:3 4:5")
    runner.add_argument("--preset", choices=("legacy", "reference"), default="legacy")
    runner.add_argument("--architecture", choices=("gcn", "mlp", "graphsage", "gat"), default="gcn")
    runner.add_argument("--hidden-channels", type=int)
    runner.add_argument("--depth", type=int)
    runner.add_argument("--dropout", type=float)
    runner.add_argument("--epochs", type=int)
    runner.add_argument("--curve-epochs", type=int, default=200)
    runner.add_argument("--lr", type=float)
    runner.add_argument("--weight-decay", type=float)
    runner.add_argument("--normalization", choices=("none", "batch", "layer"))
    runner.add_argument("--residual", action=argparse.BooleanOptionalAction, default=None)
    runner.add_argument("--pre-linear", action=argparse.BooleanOptionalAction, default=None)
    runner.add_argument("--heads", type=int)
    runner.add_argument("--selection", choices=("val_loss", "val_accuracy"))
    runner.add_argument("--curve-lr", type=float, default=0.01)
    runner.add_argument("--curve-samples", type=int, default=1, help="Uniform t samples averaged per curve optimizer step.")
    runner.add_argument("--points", type=int, default=21)
    runner.add_argument("--data-seed", type=int, default=0)
    runner.add_argument("--curve-seed", type=int, default=10000)
    runner.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N; CUDA must be available.")
    runner.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads.")
    runner.add_argument("--thesis-root", help="Parent thesis checkout; inferred from this source tree by default.")
    runner.add_argument("--smoke", action="store_true", help="Cap training at 5 endpoint/5 curve epochs, width 8, 5 points, first pair only.")
    runner.add_argument("--output", required=True, help="New or empty output directory.")
    runner.add_argument("--no-plots", action="store_true")
    plotter = commands.add_parser("plot", help="Plot saved report.json without retraining.")
    plotter.add_argument("report")
    repair_runner = commands.add_parser("repair", help="Compare alignment and REPAIR on an existing run's linear paths.")
    repair_runner.add_argument("--source", required=True, help="Original GCN connectivity report.json.")
    repair_runner.add_argument("--output", required=True, help="New or empty output directory.")
    repair_runner.add_argument("--thesis-root", help="Original thesis loader checkout; defaults to the source report's recorded location.")
    repair_runner.add_argument("--device", default="cpu")
    repair_runner.add_argument("--threads", type=int, default=1)
    repair_runner.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plot":
        from .plotting import plot_report
        for path in plot_report(args.report):
            print(path)
        return
    if args.command == "repair":
        if args.threads < 1:
            parser.error("--threads must be positive")
        from .repair_experiment import run_repair
        report = run_repair(args.source, args.output, thesis_root=args.thesis_root, device=args.device, threads=args.threads)
        print(f"Report: {report}")
        if not args.no_plots:
            from .plotting import plot_report
            for path in plot_report(report):
                print(path)
        return
    values = vars(args).copy()
    output = values.pop("output")
    no_plots = values.pop("no_plots")
    values.pop("command")
    profile_fields = ("hidden_channels", "depth", "dropout", "epochs", "lr", "weight_decay",
                      "normalization", "residual", "pre_linear", "heads", "selection")
    overrides = {name: values.pop(name) for name in profile_fields}
    values["overrides"] = {name: value for name, value in overrides.items() if value is not None}
    config = Config(**values)
    try:
        for dataset in config.datasets:
            resolved_config(config, dataset)
    except ValueError as exc:
        parser.error(str(exc))
    if config.data_seed < 0 or config.curve_seed < 0:
        parser.error("Seeds must be nonnegative")
    if len({tuple(sorted(pair)) for pair in config.pairs}) != len(config.pairs):
        parser.error("--pairs must not repeat the same endpoint pair")
    if len({name.lower() for name in config.datasets}) != len(config.datasets):
        parser.error("--datasets must not contain duplicates")
    report = run(config, output)
    print(f"Report: {report}")
    if not no_plots:
        from .plotting import plot_report
        for path in plot_report(report):
            print(path)


if __name__ == "__main__":
    main()
