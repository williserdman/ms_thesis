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

# Set to True to use the two-stage cluster-GArnoldi pipeline,
# False to use the original single-stage DiffusedAttention pipeline.
TWO_STAGE = True
# "mincut" groups tightly-connected nodes; "maxcut" pushes towards bipartiteness.
CUT_TYPE = "maxcut"
SPLIT_INDEX = 1
SPLIT_SEED = 42
OPTUNA_SEED = 42
N_TRIALS_S1 = 10
N_TRIALS_S2 = 15
N_TRIALS_SINGLE_STAGE = 20


def _detect_accelerator() -> tuple[str, int]:
    if torch.cuda.is_available():
        return "gpu", torch.cuda.device_count()

    if torch.backends.mps.is_available():
        return "mps", 1

    return "cpu", 0


def train_job(network_name, gpu_id, results_list):
    pl.seed_everything(SEED, workers=True)

    if gpu_id == "mps":
        ot = OptunaTrainer(
            "mps",
            "auto",
            split_index=SPLIT_INDEX,
            split_seed=SPLIT_SEED,
            optuna_seed=OPTUNA_SEED,
        )
    elif isinstance(gpu_id, int):
        ot = OptunaTrainer(
            "gpu",
            gpu_id,
            split_index=SPLIT_INDEX,
            split_seed=SPLIT_SEED,
            optuna_seed=OPTUNA_SEED,
        )
    else:
        ot = OptunaTrainer(
            "cpu",
            "auto",
            split_index=SPLIT_INDEX,
            split_seed=SPLIT_SEED,
            optuna_seed=OPTUNA_SEED,
        )

    study = ot.run_optimization(network_name, n_trials=N_TRIALS_SINGLE_STAGE)
    results = ot.test_best_model(study, network_name)

    results_list.append(
        (
            network_name,
            {
                "status": "ok",
                "test": results,
                "gpu": gpu_id,
                "split_index": SPLIT_INDEX,
                "split_seed": SPLIT_SEED,
                "optuna_seed": OPTUNA_SEED,
                "n_trials": N_TRIALS_SINGLE_STAGE,
            },
        )
    )

    return


def train_job_two_stage(network_name, gpu_id, results_list, cut_type="mincut"):
    """Two-stage cluster-GArnoldi pipeline."""
    pl.seed_everything(SEED, workers=True)

    if gpu_id == "mps":
        tst = TwoStageTrainer(
            "mps",
            "auto",
            split_index=SPLIT_INDEX,
            split_seed=SPLIT_SEED,
            optuna_seed=OPTUNA_SEED,
        )
    elif isinstance(gpu_id, int):
        tst = TwoStageTrainer(
            "gpu",
            gpu_id,
            split_index=SPLIT_INDEX,
            split_seed=SPLIT_SEED,
            optuna_seed=OPTUNA_SEED,
        )
    else:
        tst = TwoStageTrainer(
            "cpu",
            "auto",
            split_index=SPLIT_INDEX,
            split_seed=SPLIT_SEED,
            optuna_seed=OPTUNA_SEED,
        )

    study_s2, clusters, num_clusters = tst.run_optimization(
        network_name,
        cut_type=cut_type,
        n_trials_s1=N_TRIALS_S1,
        n_trials_s2=N_TRIALS_S2,
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
                "split_index": SPLIT_INDEX,
                "split_seed": SPLIT_SEED,
                "optuna_seed": OPTUNA_SEED,
                "n_trials_s1": N_TRIALS_S1,
                "n_trials_s2": N_TRIALS_S2,
            },
        )
    )

    return


def main():
    accelerator, worker_count = _detect_accelerator()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_tag = "twostage" if TWO_STAGE else "singlestage"
    out_filename = f"training_results_{pipeline_tag}_{timestamp}.json"

    # --- THE FIX ---
    # Define the target function and keyword arguments cleanly
    # instead of using an unpicklable lambda.
    if TWO_STAGE:
        target_fn = train_job_two_stage
        process_kwargs = {"cut_type": CUT_TYPE}
    else:
        target_fn = train_job
        process_kwargs = {}
    # ---------------

    manager = multiprocessing.Manager()
    results = manager.list()

    if accelerator == "cpu":
        # fallback: run sequentially on CPU
        for d in ALL_DATASETS:
            # Unpack kwargs here for the CPU fallback
            target_fn(d, "cpu", results, **process_kwargs)
    elif accelerator == "mps":
        # MPS supports one device; run sequentially to avoid contention.
        for d in ALL_DATASETS:
            target_fn(d, "mps", results, **process_kwargs)
    else:
        free_gpus = list(range(worker_count))
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

            # Pass the function directly to 'target' and use 'kwargs' for extra arguments
            p = multiprocessing.Process(
                target=target_fn, args=(d, gpu, results), kwargs=process_kwargs
            )
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
