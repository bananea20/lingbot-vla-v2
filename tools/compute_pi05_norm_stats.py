#!/usr/bin/env python3
"""Stream native Cartesian parquet data through the pi0.5 physical transform.

Example for the native SO(3) fridge configuration::

    .venv/bin/python tools/compute_pi05_norm_stats.py \
        --chunk-size 50 --output assets/norm_stats/s1_fridge_pi05.json

Only numeric parquet columns are read; videos and model weights are not loaded.
Statistics describe Pi05IO's packed physical features. Normalization uses these
32D statistics BEFORE model_indices scatters the result into 55D model slots. A
nonlinear adapter (for example, 6D rotation to quaternion) must recompute stats
after that adapter; it cannot transform quantiles from this output directly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbotvla.data.vla_data.pi05_io import Pi05IO
from lingbotvla.utils.normalize import RunningStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config", type=Path,
        default=REPO_ROOT / "configs/robot_configs/s1_fridge_pi05.yaml",
    )
    parser.add_argument(
        "--train-list", type=Path,
        default=REPO_ROOT / "assets/training_data/s1_fridge_pi05.txt",
    )
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--anchor-stride", type=int, default=1,
        help="Use every Nth observation within each episode; 1 means full coverage.",
    )
    parser.add_argument(
        "--max-episodes", type=int,
        help="Optional deterministic prefix for a smoke check; recorded as partial coverage.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly allow replacing the selected output file.",
    )
    args = parser.parse_args()
    for name in ("chunk_size", "batch_size", "torch_threads", "anchor_stride", "max_episodes"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output exists: {args.output}; choose a new file or pass --overwrite")
    return args


def source_feature(config: dict, category: str) -> tuple[str, str]:
    entries = config[category]
    if len(entries) != 1 or len(entries[0]) != 1:
        raise ValueError(f"Expected one packed {category} feature")
    key, spec = next(iter(entries[0].items()))
    origin = spec.get("origin_keys")
    if not isinstance(origin, str):
        raise ValueError(f"{category}.origin_keys must name a complete native vector")
    return key, origin


def dataset_roots(train_list: Path, robot_name: str) -> list[Path]:
    roots = []
    for lineno, line in enumerate(train_list.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{train_list}:{lineno}: expected robot_name dataset_root")
        if parts[0] != robot_name:
            raise ValueError(f"{train_list}:{lineno}: expected robot {robot_name}, got {parts[0]}")
        root = Path(parts[1])
        if not root.is_absolute():
            root = REPO_ROOT / root
        if not root.is_dir():
            raise FileNotFoundError(root)
        roots.append(root.resolve())
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Dataset list must contain distinct, existing roots")
    return roots


def read_episode(path: Path, state_column: str, action_column: str) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=[state_column, action_column, "episode_index", "frame_index"])
    if len(table) == 0:
        raise ValueError(f"Empty episode: {path}")
    episodes = table["episode_index"].to_numpy()
    frames = table["frame_index"].to_numpy()
    if np.unique(episodes).size != 1 or not np.array_equal(frames, np.arange(len(table))):
        raise ValueError(f"Expected one complete, ordered episode per parquet: {path}")
    states = np.asarray(table[state_column].to_pylist(), dtype=np.float32)
    actions = np.asarray(table[action_column].to_pylist(), dtype=np.float32)
    if states.ndim != 2 or actions.shape != states.shape:
        raise ValueError(f"State/action shape mismatch in {path}: {states.shape}, {actions.shape}")
    return states, actions


def stats_dict(accumulator: RunningStats, count: int) -> dict:
    result = accumulator.get_statistics()
    output = {
        key: np.asarray(getattr(result, key)).tolist()
        for key in ("mean", "std", "min", "max", "q01", "q99", "q02", "q98")
    }
    output["count"] = count
    return output


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    config_text = args.robot_config.read_text()
    config = yaml.safe_load(config_text)
    io_config = config["pi05_io"]
    processor = Pi05IO(io_config)
    state_key, state_column = source_feature(config, "states")
    action_key, action_column = source_feature(config, "actions")
    roots = dataset_roots(args.train_list, args.robot_config.stem)
    files = [(root, path) for root in roots for path in sorted((root / "data").rglob("*.parquet"))]
    if not files:
        raise ValueError("No episode parquet files found")
    available_episodes = len(files)
    if args.max_episodes is not None:
        files = files[:args.max_episodes]

    state_stats, action_stats = RunningStats(), RunningStats()
    total_rows = total_anchors = total_actions = 0
    episode_manifest = []
    offsets = np.arange(args.chunk_size)
    with torch.inference_mode():
        for number, (root, path) in enumerate(files, 1):
            states, actions = read_episode(path, state_column, action_column)
            if states.shape[-1] != processor.raw_dim:
                raise ValueError(f"Expected {processor.raw_dim} native dimensions, got {states.shape[-1]}: {path}")
            anchors = np.arange(0, len(states), args.anchor_stride)
            for start in range(0, len(anchors), args.batch_size):
                batch_anchors = anchors[start:start + args.batch_size]
                # LeRobot repeats the final frame for offsets past the episode.
                # Like pi0.5's flattened statistics, keep these padded actions.
                future = np.minimum(batch_anchors[:, None] + offsets[None, :], len(actions) - 1)
                packed_state, packed_action = processor.preprocess(states[batch_anchors], actions[future])
                state_stats.update(packed_state.cpu().numpy().astype(np.float64))
                action_stats.update(packed_action.cpu().numpy().reshape(-1, processor.action_max_dim).astype(np.float64))
                total_anchors += len(batch_anchors)
                total_actions += len(batch_anchors) * args.chunk_size
            total_rows += len(states)
            episode_manifest.append({
                "root": str(root), "path": str(path.relative_to(root)),
                "rows": len(states), "anchors": len(anchors),
            })
            if number == 1 or number % 10 == 0 or number == len(files):
                print(f"Episodes {number}/{len(files)}; anchors {total_anchors:,}; action vectors {total_actions:,}", flush=True)

    metadata = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_layout": "pi05_packed_physical_before_model_adapter",
        "pi05_io": io_config,
        "robot_config": str(args.robot_config.resolve()),
        "robot_config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "train_list": str(args.train_list.resolve()),
        "native_state_column": state_column,
        "native_action_column": action_column,
        "chunk_size": args.chunk_size,
        "chunk_offsets": list(range(args.chunk_size)),
        "episode_boundary": "repeat_last_action_including_padding_in_statistics",
        "quantile_estimator": "RunningStats_dynamic_histogram_5000_bins",
        "normalization_scope": "pooled_over_all_chunk_positions",
        "batch_size": args.batch_size,
        "anchor_stride": args.anchor_stride,
        "max_episodes": args.max_episodes,
        "available_episodes": available_episodes,
        "processed_episodes": len(files),
        "processed_episode_rows": total_rows,
        "state_vectors": total_anchors,
        "action_vectors": total_actions,
        "full_coverage": args.anchor_stride == 1 and len(files) == available_episodes,
        "episodes": episode_manifest,
    }
    output = {"norm_stats": {
        state_key: stats_dict(state_stats, total_anchors),
        action_key: stats_dict(action_stats, total_actions),
    }, "metadata": metadata}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.output}; full_coverage={metadata['full_coverage']}", flush=True)


if __name__ == "__main__":
    main()
