"""
Hardcoded batch launcher for Gigasealing BC configs.

Set RUN_MODE to "sequential" or "parallel", then run:

    py -3 robomimic/scripts/train_batch.py
"""

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


VERSIONS = [929, 930, 931]
RUN_MODE = "parallel"  # "sequential" or "parallel"

# Keep empty to use the experiment names from the configs, e.g. v0_923.
# Set to something like "_rerun" if those output folders already exist.
RUN_NAME_SUFFIX = ""

# Parallel jobs should not wait at train.py's overwrite prompt.
REQUIRE_FRESH_OUTPUT_DIRS = True


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "exps" / "templates" / "Gigasealing" / "bc"


def load_config(version):
    config_path = CONFIG_DIR / f"bc-PatcherBot_v0_{version}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config for v0_{version}: {config_path}")

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


def run_version(version):
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    error = None

    try:
        print(f"\n===== Training v0_{version} =====", flush=True)
        config = load_config(version)
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
        print(f"Using device for v0_{version}: {device}", flush=True)
        train(config, device=device, resume=False)
    except Exception:
        error = traceback.format_exc()
        print(error)
    finally:
        restore_stdio(orig_stdout, orig_stderr)

    if error is not None:
        print(f"===== v0_{version} failed =====\n{error}", file=sys.stderr, flush=True)
        return 1

    print(f"===== Finished v0_{version} =====", flush=True)
    return 0


def run_child(version):
    sys.exit(run_version(version))


def check_output_dirs():
    existing = []
    for version in VERSIONS:
        config = load_config(version)
        out_dir = experiment_dir(config)
        if out_dir.exists():
            existing.append((version, out_dir))

    if not existing:
        return

    msg = [
        "These experiment output directories already exist, so train.py would prompt for overwrite:",
    ]
    msg.extend(f"  v0_{version}: {out_dir}" for version, out_dir in existing)
    msg.extend(
        [
            "",
            "Use a fresh RUN_NAME_SUFFIX in train_batch.py, delete or rename the old folders,",
            "or set REQUIRE_FRESH_OUTPUT_DIRS = False if you want train.py's interactive prompt.",
        ]
    )
    raise RuntimeError("\n".join(msg))


def run_sequential():
    for version in VERSIONS:
        status = run_version(version)
        if status != 0:
            return status
    return 0


def run_parallel():
    ctx = mp.get_context("spawn")
    processes = []

    for version in VERSIONS:
        process = ctx.Process(target=run_child, args=(version,), name=f"train-v0_{version}")
        process.start()
        processes.append((version, process))

    status = 0
    for version, process in processes:
        process.join()
        if process.exitcode != 0:
            print(f"v0_{version} exited with code {process.exitcode}", file=sys.stderr)
            status = 1
    return status


def main():
    if RUN_MODE not in ("sequential", "parallel"):
        raise ValueError(f'RUN_MODE must be "sequential" or "parallel", got {RUN_MODE!r}')

    if REQUIRE_FRESH_OUTPUT_DIRS:
        check_output_dirs()

    if RUN_MODE == "sequential":
        return run_sequential()
    return run_parallel()


if __name__ == "__main__":
    sys.exit(main())
