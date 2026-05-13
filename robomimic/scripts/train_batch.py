"""
Hardcoded batch launcher for PatcherBot BC and diffusion configs.

Set RUN_MODE to "sequential" or "parallel", then run:

    py -3 robomimic/scripts/train_batch.py
"""

from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback

import robomimic
from robomimic.config import config_factory
from robomimic.scripts.train import train
import robomimic.utils.torch_utils as TorchUtils


MODEL_VERSIONS = {
    "Burglary": {
        "bc": [987],
        "df": [],
    },
    # Backwards-compatible shorthand is still accepted:
    # "Burglary": [981, 982],  # equivalent to {"bc": [981, 982]}
    # "PipetteFinding": {"bc": [980]},
    # "NeuronHunting": {"bc": [980]},
}
RUN_MODE = "parallel"  # "sequential" or "parallel"

# Keep empty to use the experiment names from the configs, e.g. v0_923.
# Set to something like "_rerun" if those output folders already exist.
RUN_NAME_SUFFIX = ""

# Parallel jobs should not wait at train.py's overwrite prompt.
REQUIRE_FRESH_OUTPUT_DIRS = True


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = SCRIPT_DIR.parent / "exps" / "templates"


@dataclass(frozen=True)
class TrainingJob:
    model: str
    model_type: str
    version: int


def build_jobs():
    jobs = []
    for model, model_jobs in MODEL_VERSIONS.items():
        if isinstance(model_jobs, dict):
            for model_type, versions in model_jobs.items():
                for version in versions:
                    jobs.append(TrainingJob(model=model, model_type=model_type, version=version))
        else:
            for version in model_jobs:
                jobs.append(TrainingJob(model=model, model_type="bc", version=version))
    return jobs


def job_label(job):
    return f"{job.model} {job.model_type} v0_{job.version}"


def config_dir(job):
    return CONFIG_ROOT / job.model / job.model_type


def load_config(job):
    config_path = config_dir(job) / f"{job.model_type}-PatcherBot_v0_{job.version}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config for {job_label(job)}: {config_path}")

    with config_path.open("r") as f:
        ext_cfg = json.load(f)

    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
        if RUN_NAME_SUFFIX:
            config.experiment.name = f"{config.experiment.name}{RUN_NAME_SUFFIX}"

    config.lock()
    return config


def experiment_dir(config):
    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    return Path(base_output_dir).resolve() / config.experiment.name


def restore_stdio(orig_stdout, orig_stderr):
    redirected = []
    for stream in (sys.stdout, sys.stderr):
        if stream not in (orig_stdout, orig_stderr) and stream not in redirected:
            redirected.append(stream)

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

    for stream in redirected:
        log_file = getattr(stream, "log_file", None)
        if log_file is not None and not log_file.closed:
            log_file.close()


def run_job(job):
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    error = None

    try:
        print(f"\n===== Training {job_label(job)} =====", flush=True)
        config = load_config(job)
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
        print(f"Using device for {job_label(job)}: {device}", flush=True)
        train(config, device=device, resume=False)
    except Exception:
        error = traceback.format_exc()
        print(error)
    finally:
        restore_stdio(orig_stdout, orig_stderr)

    if error is not None:
        print(f"===== {job_label(job)} failed =====\n{error}", file=sys.stderr, flush=True)
        return 1

    print(f"===== Finished {job_label(job)} =====", flush=True)
    return 0


def run_child(job):
    sys.exit(run_job(job))


def check_output_dirs(jobs):
    existing = []
    for job in jobs:
        config = load_config(job)
        out_dir = experiment_dir(config)
        if out_dir.exists():
            existing.append((job, out_dir))

    if not existing:
        return

    msg = [
        "These experiment output directories already exist, so train.py would prompt for overwrite:",
    ]
    msg.extend(f"  {job_label(job)}: {out_dir}" for job, out_dir in existing)
    msg.extend(
        [
            "",
            "Use a fresh RUN_NAME_SUFFIX in train_batch.py, delete or rename the old folders,",
            "or set REQUIRE_FRESH_OUTPUT_DIRS = False if you want train.py's interactive prompt.",
        ]
    )
    raise RuntimeError("\n".join(msg))


def run_sequential(jobs):
    for job in jobs:
        status = run_job(job)
        if status != 0:
            return status
    return 0


def run_parallel(jobs):
    ctx = mp.get_context("spawn")
    processes = []

    for job in jobs:
        process = ctx.Process(
            target=run_child,
            args=(job,),
            name=f"train-{job.model}-{job.model_type}-v0_{job.version}",
        )
        process.start()
        processes.append((job, process))

    status = 0
    for job, process in processes:
        process.join()
        if process.exitcode != 0:
            print(f"{job_label(job)} exited with code {process.exitcode}", file=sys.stderr)
            status = 1
    return status


def main():
    if RUN_MODE not in ("sequential", "parallel"):
        raise ValueError(f'RUN_MODE must be "sequential" or "parallel", got {RUN_MODE!r}')

    jobs = build_jobs()
    if not jobs:
        raise ValueError("MODEL_VERSIONS must contain at least one model/version pair")

    if REQUIRE_FRESH_OUTPUT_DIRS:
        check_output_dirs(jobs)

    if RUN_MODE == "sequential":
        return run_sequential(jobs)
    return run_parallel(jobs)


if __name__ == "__main__":
    sys.exit(main())
