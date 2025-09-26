import argparse
import collections
import logging
import re
import time
import subprocess

import tqdm
import wandb

# TODO(HE): check the below statement: should it check for running evals 
# of the train run id?
# Note: this script does not account for being run twice with the same
# run id simultaneously.

# This script depends heavily on the config.
# Also the training run configs.
EVAL_CONFIG = "transformer_v3_ens_v1"
EVALS_RUN_PATH = "research/evaluating_our_models"


def cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Offline evaluation for all checkpoints of a run."
        " Note that this runs the checkpoints from the last step to the first."
    )
    parser.add_argument("run_id", type=str)
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Weights and Biases API timeout parameter.",
    )
    parser.add_argument(
        "--ens-minibatch",
        type=int,
        default=2,
        help="Ensenmble minibatch size.",
    )
    parser.add_argument(
        "--recompute-existing",
        action="store_true",
        help=(
            "Whether to recompute existing evals for this run."
            " If True, we don't skip evaluating checkpoints which already have an eval."
        ),
    )
    return parser.parse_args()


def checkpoint_step_from_artifact_name(name: str) -> int | None:
    match = re.match("checkpoint-[a-z0-9]*-([0-9]*):.*", name)
    return None if match is None else int(match.groups()[0])


def checkpoint_step_from_eval_run(run: wandb.apis.public.Run) -> int:
    return run.config["evaluation_config"]["model"]["kwargs"]["checkpoint_step"]


def timestamp_from_eval_run(run: wandb.apis.public.Run) -> float:
    match = re.match(r"^[a-z0-9]*_step_[0-9]*_timestamp_([0-9]*\.[0-9]*).*", run.name)
    if match is None:
        raise ValueError("Can't find timestamp in run name.")
    return float(match.groups()[0])


# TODO(HE): refactor
def get_steps_to_eval(
    api: wandb.Api,
    run_to_eval: wandb.apis.public.Run,
    recompute_existing: bool,
) -> tuple[set[int]]:
    eval_run_filters = {
        "$and": [
            {"display_name": {"$regex": f"^{run_to_eval.id}.*"}},
            {"$or": [{"state": "finished"}, {"state": "running"}]},
        ]
    }
    evals_to_skip_gen = tqdm.tqdm(
        api.runs(EVALS_RUN_PATH, filters=eval_run_filters, per_page=50),
        desc="Finding existing evals.",
    )

    existing_evals = collections.defaultdict(list)
    for r in evals_to_skip_gen:
        existing_evals[checkpoint_step_from_eval_run(r)].append(r)

    steps_to_eval = set()
    skipped = set()
    for artifact in tqdm.tqdm(
        run_to_eval.logged_artifacts(),
        desc="Parsing run artifacts",
    ):
        if "checkpoint" in artifact.name:
            checkpoint_step = checkpoint_step_from_artifact_name(artifact.name)
            if checkpoint_step is None:
                print("Warning: checkpoint without matching step!")
                print(artifact.name)
            elif not recompute_existing and checkpoint_step in existing_evals:
                skipped.add(checkpoint_step)
            else:
                steps_to_eval.add(checkpoint_step)
    return steps_to_eval, skipped


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = cmd_args()
    start_time = time.time()

    run_path = f"research/training_setup/{args.run_id}"

    wandb.login(host="https://fundamental.wandb.io", key=None)
    api = wandb.Api(timeout=args.timeout)
    run = api.run(run_path)

    static_overwrites = {
        "run_path": run_path,
        "model.name": run.config["model"]["name"],
        "model.kwargs.ens_minibatch": args.ens_minibatch,
    }

    logging.info("Calculating checkpoint steps to evaluate.")
    steps_to_eval, skipped = get_steps_to_eval(api, run, args.recompute_existing)
    logging.info("Found existing evals for %s.", sorted(skipped))
    while steps_to_eval:
        checkpoint_step = max(steps_to_eval)

        overwrites = {
            **static_overwrites,
            "name": f"{run.id}_step_{checkpoint_step}_timestamp_{start_time}",
            "model.kwargs.checkpoint_step": checkpoint_step,
        }

        command = [
            "python",
            "-m",
            "evaluation.evaluate",
            f"--config_name={EVAL_CONFIG}",
            "--overwrites",
            *(f"{k}={v}" for k, v in overwrites.items()),
        ]
        logging.info("Evaluating step %s.", checkpoint_step)
        logging.info("Executing %s.", " ".join(command))
        subprocess.run(command, check=True)

        logging.info("Refreshing checkpoint steps to evaluate.")
        steps_to_eval, skipped = get_steps_to_eval(api, run, args.recompute_existing)
        logging.info("Found existing evals for %s.", sorted(skipped))
