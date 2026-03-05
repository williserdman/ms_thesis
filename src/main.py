import torch
import pytorch_lightning as pl
from src.optuna_trainer import OptunaTrainer
from src.two_stage_trainer import TwoStageTrainer
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

# Set to True to use the two-stage cluster-GArnoldi pipeline,
# False to use the original single-stage DiffusedAttention pipeline.
TWO_STAGE = True
# "mincut" groups tightly-connected nodes; "maxcut" pushes towards bipartiteness.
CUT_TYPE = "mincut"


def train_job(network_name, gpu_id, results_list):
    pl.seed_everything(SEED, workers=True)

    if isinstance(gpu_id, int):
        ot = OptunaTrainer("gpu", gpu_id)
    else:
        ot = OptunaTrainer("cpu", "auto")

    study = ot.run_optimization(network_name)
    results = ot.test_best_model(study, network_name)

    results_list.append(
        (network_name, {"status": "ok", "test": results, "gpu": gpu_id})
    )

    return


def train_job_two_stage(network_name, gpu_id, results_list, cut_type="mincut"):
    """Two-stage cluster-GArnoldi pipeline."""
    pl.seed_everything(SEED, workers=True)

    if isinstance(gpu_id, int):
        tst = TwoStageTrainer("gpu", gpu_id)
    else:
        tst = TwoStageTrainer("cpu", "auto")

    study_s2, clusters, num_clusters = tst.run_optimization(
        network_name, cut_type=cut_type
    )
    results = tst.test_best_model(study_s2, network_name, clusters, num_clusters)

    results_list.append(
        (
            network_name,
            {
                "status": "ok",
                "test": results,
                "gpu": gpu_id,
                "cut_type": cut_type,
                "pipeline": "two_stage",
            },
        )
    )

    return


def main():
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_tag = "twostage" if TWO_STAGE else "singlestage"
    out_filename = f"training_results_{pipeline_tag}_{timestamp}.json"

    job_fn = (
        (lambda name, gpu, res: train_job_two_stage(name, gpu, res, CUT_TYPE))
        if TWO_STAGE
        else train_job
    )

    manager = multiprocessing.Manager()
    results = manager.list()

    if gpu_count <= 0:
        # fallback: run sequentially on CPU
        for d in ALL_DATASETS:
            job_fn(d, "cpu", results)
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
            p = multiprocessing.Process(target=job_fn, args=(d, gpu, results))
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
