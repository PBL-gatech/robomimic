"""
Plot training and validation loss curves from robomimic log.txt files.

Run from the repo root:

    py -3 robomimic/scripts/loss_comparison.py
"""

import json
import math
from pathlib import Path
import re


VERSIONS = [922, 923, 924, 925, 926, 928, 929,930,931,932]  # versions to compare, e.g. [922, 923, 924, 9251]

# Leave empty to use the newest timestamp folder for each version.
# Example override: {922: "20260502213703"}
TIMESTAMPS = {}


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "bc_patcherBot" / "Gigasealing"
OUTPUT_DIR = RUN_ROOT / "results" / "loss_comparison_922_925"
EPOCH_RE = re.compile(r"^(Train|Validation) Epoch\s+(\d+)")


def find_log_path(version):
    run_dir = RUN_ROOT / f"v0_{version}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")

    timestamp = TIMESTAMPS.get(version)
    if timestamp:
        log_path = run_dir / timestamp / "logs" / "log.txt"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log for v0_{version}: {log_path}")
        return log_path

    candidates = [
        path / "logs" / "log.txt"
        for path in run_dir.iterdir()
        if path.is_dir() and (path / "logs" / "log.txt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No timestamp folders with logs found in {run_dir}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_epoch_blocks(log_path):
    records = {"Train": [], "Validation": []}
    lines = log_path.read_text(errors="replace").splitlines()
    i = 0

    while i < len(lines):
        match = EPOCH_RE.match(lines[i].strip())
        if match is None:
            i += 1
            continue

        phase = match.group(1)
        epoch = int(match.group(2))
        i += 1

        while i < len(lines) and lines[i].strip() != "{":
            i += 1
        if i >= len(lines):
            break

        block = []
        depth = 0
        while i < len(lines):
            line = lines[i]
            block.append(line)
            depth += line.count("{") - line.count("}")
            i += 1
            if depth == 0:
                break

        try:
            data = json.loads("\n".join(block))
        except json.JSONDecodeError:
            continue

        data["epoch"] = epoch
        records[phase].append(data)

    return records


def collect_loss_metrics(all_records):
    metrics = set()
    for records in all_records.values():
        for phase_records in records.values():
            for row in phase_records:
                metrics.update(key for key in row if key == "Loss" or key.endswith("_Loss"))

    def sort_key(metric):
        if metric == "Loss":
            return (0, metric)
        return (1, metric)

    return sorted(metrics, key=sort_key)


def series(records, phase, metric):
    rows = [row for row in records[phase] if metric in row]
    epochs = [row["epoch"] for row in rows]
    values = [row[metric] for row in rows]
    return epochs, values


def plot_phase_losses(all_records, metrics, phase, output_name):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for loss_comparison.py. Install it in the same "
            "environment you use for training."
        ) from exc

    cols = 2
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 4.8 + 1.2))
    try:
        axes = list(axes.flat)
    except AttributeError:
        axes = [axes]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, metric in zip(axes, metrics):
        for run_i, (label, records) in enumerate(all_records.items()):
            color = color_cycle[run_i % len(color_cycle)]
            epochs, values = series(records, phase, metric)
            if epochs:
                ax.plot(epochs, values, color=color, linewidth=1.8, label=label)

        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect(1)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"Gigasealing BC {phase} Loss Comparison", y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / output_name
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def plot_losses(all_records, log_paths):
    metrics = collect_loss_metrics(all_records)
    if not metrics:
        raise RuntimeError("No loss metrics were found in the provided logs.")

    train_plot_path = plot_phase_losses(all_records, metrics, "Train", "train_loss_comparison.png")
    valid_plot_path = plot_phase_losses(all_records, metrics, "Validation", "validation_loss_comparison.png")

    index_path = OUTPUT_DIR / "logs_used.txt"
    with index_path.open("w") as f:
        for label, log_path in log_paths.items():
            f.write(f"{label}: {log_path}\n")

    return train_plot_path, valid_plot_path, index_path


def main():
    all_records = {}
    log_paths = {}

    for version in VERSIONS:
        label = f"v0_{version}"
        log_path = find_log_path(version)
        log_paths[label] = log_path
        all_records[label] = parse_epoch_blocks(log_path)

    train_plot_path, valid_plot_path, index_path = plot_losses(all_records, log_paths)
    print(f"Wrote train plot: {train_plot_path}")
    print(f"Wrote validation plot: {valid_plot_path}")
    print(f"Wrote log index: {index_path}")


if __name__ == "__main__":
    main()
