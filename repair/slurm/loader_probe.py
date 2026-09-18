"""Measure repeated CIFAR-10 DataLoader startup with persistent workers off and on."""

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T


def make_dataset() -> datasets.CIFAR10:
    normalize = T.Normalize(
        [125.307 / 255, 122.961 / 255, 113.8575 / 255],
        [51.5865 / 255, 50.847 / 255, 51.255 / 255],
    )
    transform = T.Compose([T.ToTensor(), normalize])
    return datasets.CIFAR10("data", train=True, download=False, transform=transform)


def measure(persistent_workers: bool) -> dict:
    loader = DataLoader(
        make_dataset(),
        batch_size=500,
        shuffle=False,
        num_workers=4,
        persistent_workers=persistent_workers,
    )
    passes = []
    for pass_number in range(2):
        started = time.perf_counter()
        iterator = iter(loader)
        iterator_seconds = time.perf_counter() - started
        digest = hashlib.sha256()
        input_sum = 0.0
        label_sum = 0
        samples = 0
        first_batch_seconds = None
        for batch_number in range(10):
            batch_started = time.perf_counter()
            inputs, labels = next(iterator)
            if first_batch_seconds is None:
                first_batch_seconds = time.perf_counter() - batch_started
            digest.update(inputs.numpy().tobytes())
            digest.update(labels.numpy().tobytes())
            input_sum += inputs.double().sum().item()
            label_sum += labels.sum().item()
            samples += labels.numel()
        elapsed = time.perf_counter() - started
        row = {
            "pass": pass_number + 1,
            "batches": 10,
            "samples": samples,
            "sha256": digest.hexdigest(),
            "input_sum": input_sum,
            "label_sum": label_sum,
            "iterator_seconds": iterator_seconds,
            "first_batch_seconds": first_batch_seconds,
            "total_seconds": elapsed,
        }
        passes.append(row)
        print(json.dumps({"persistent_workers": persistent_workers, **row}), flush=True)
        del iterator
    del loader
    return {"persistent_workers": persistent_workers, "passes": passes}


def main() -> None:
    mp.set_start_method("forkserver", force=True)
    torch.set_num_threads(4)
    results = [measure(False), measure(True)]
    signatures = {
        (row["samples"], row["sha256"], row["label_sum"])
        for result in results
        for row in result["passes"]
    }
    if len(signatures) != 1:
        raise RuntimeError(f"Data mismatch across passes/configurations: {signatures}")
    output = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "multiprocessing_start_method": mp.get_start_method(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "data": "CIFAR-10 training split, deterministic evaluation transform",
        "results": results,
        "data_matches": True,
    }
    output_path = Path("runs") / f"loader-probe-{output['job_id'] or 'local'}.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
