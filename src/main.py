import torch
import pytorch_lightning as pl
from optuna_trainer import run_optimization, test_best_model
import datetime
import multiprocessing
import time
import json

# dispatches jobs to multiple GPUs or CPU depending on setup
SEED = 42
# to use tensor cores
torch.set_float32_matmul_precision(
    "high"
)  # "medium" also works to take advantage of tensor cores

ALL_DATASETS = [
    "Questions",
    "Roman-empire",
    "Amazon-ratings",
    "Tolokers",
    "computers",
    "photo",
    "texas",
    "cornell",
    "Cora",
    "Citeseer",
    "Pubmed",
    "squirrel",
    "chameleon",
    "actor",
    "Minesweeper",
]

ALL_DATASETS = ["squirrel", "chameleon", "Roman-empire"]


def train_job(network_name, gpu_id, results_list):
    pl.seed_everything(SEED, workers=True)

    study = run_optimization(network_name)
    results = test_best_model(study, network_name)

    results_list.append(
        (network_name, {"status": "ok", "test": results, "gpu": gpu_id})
    )

    return


def main():
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"training_results_{timestamp}.json"

    manager = multiprocessing.Manager()
    results = manager.list()

    if gpu_count <= 0:
        # fallback: run sequentially on CPU
        for d in ALL_DATASETS:
            train_job(d, "cpu", results)
    else:
        print(1)
        free_gpus = list(range(gpu_count))
        processes = {}

        for d in ALL_DATASETS:
            # wait until a GPU is available
            while not free_gpus:
                # poll running processes and reclaim finished GPUs
                for p in list(processes.keys()):
                    if not p.is_alive():
                        p.join()
                        free_gpus.append(processes.pop(p))
                if not free_gpus:
                    time.sleep(1)

            gpu = free_gpus.pop(0)
            p = multiprocessing.Process(target=train_job, args=(d, gpu, results))
            p.start()
            processes[p] = gpu

        # wait for all processes to finish
        for p in list(processes.keys()):
            p.join()
            processes.pop(p)

    # Print summary of results
    print("Training summary:")
    for item in list(results):
        dataset_name, info = item
        print(f"{dataset_name}: {info}")

    # write results to file
    output = []
    for item in list(results):
        dataset_name, info = item
        entry = {"dataset": dataset_name}
        entry.update(info)
        output.append(entry)
    with open(out_filename, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"Wrote results to {out_filename}")


if __name__ == "__main__":
    main()
