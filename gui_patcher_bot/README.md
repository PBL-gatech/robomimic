# PatcherBot Evaluation GUI

PyQt5 GUI for evaluating PatcherBot checkpoints while delegating the actual rollout and plotting work to shared `robomimic` modules.

## What Changed

- `gui_patcher_bot` no longer carries its own full evaluator and plot implementation.
- Checkpoint evaluation is delegated to `robomimic.utils.patcherbot_eval`.
- Plot generation is delegated to `robomimic.utils.patcherbot_visualization`.
- The GUI now renders the generated PNG artifacts and loads metadata for the selected result.

## Usage

```bash
cd gui_patcher_bot
python main.py
```

## GUI Flow

1. Select a checkpoint directory containing `.pth` files.
2. Set the dataset path and output directories.
3. Click `Evaluate Checkpoints`.
4. Select a row in `Summary` to inspect plots and metadata.
5. Use `Open CSV` to visualize an existing CSV without running evaluation.

## Output

- CSVs are written per checkpoint into `CSV Output Dir`.
- Metadata JSON files are written per checkpoint into `Metadata Dir`.
- Plot PNGs are written into `<CSV Output Dir>/plots`.

## Dependencies

The GUI expects the local `robomimic` repository to be available, since the evaluation and plotting implementations now live under `robomimic/utils`.
