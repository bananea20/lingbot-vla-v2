#!/usr/bin/env python
"""
34D -> 25D converter that discovers sessions instead of hardcoding them.

tools/convert_s1_dataset.py assumes the pick_and_move layout
(<task>/<camera_group>/<session>/) with a fixed session list. Newer datasets
nest differently, e.g. open_refrigerator_with_move is
(<task>/<session>/lerobot_so3_data_30hz/) with all 5 cameras in one dataset
rather than split into head_camera/stereo_camera trees.

This version finds every meta/info.json under the source root and converts
each in place-relative, so it works for either layout.

Layouts are documented in tools/convert_s1_dataset.py; the 34D->25D math is
identical and imported from there.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.align_quat_signs import align_dataset
from tools.convert_s1_dataset import convert_34d_to_25d


def find_dataset_roots(src_root):
    """Every directory containing meta/info.json, relative to src_root."""
    roots = sorted(
        p.parent.parent.relative_to(src_root)
        for p in src_root.rglob("meta/info.json")
    )
    if not roots:
        raise SystemExit(f"no meta/info.json found under {src_root}")
    return roots


def convert_episode(src_pq, dst_pq):
    df = pd.read_parquet(src_pq)

    state = torch.from_numpy(
        np.stack(df["cartesian_so3_dict.cartesian_pose_state"].values)
    )
    action = torch.from_numpy(
        np.stack(df["cartesian_so3_dict.cartesian_pose_command"].values)
    )

    # task_index/sub_task_index are shape [2]; base_dataset.py calls .item() on
    # them, which raises for multi-element tensors. Take the first element.
    def flatten(col):
        return np.array([v[0] if hasattr(v, "__len__") else v for v in df[col]])

    out = pd.DataFrame({
        "observation.state": list(convert_34d_to_25d(state).numpy()),
        "action": list(convert_34d_to_25d(action).numpy()),
        "timestamp": df["timestamp"],
        "frame_index": df["frame_index"],
        "episode_index": df["episode_index"],
        "index": df["index"],
        "task_index": flatten("task_index"),
        "sub_task_index": flatten("sub_task_index"),
    })
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst_pq, index=False)
    return len(df)


def rewrite_info(info_path):
    info = json.load(open(info_path))
    orig = info.get("features", {})
    new = {
        "observation.state": {"shape": [25], "dtype": "float64"},
        "action": {"shape": [25], "dtype": "float64"},
    }
    # Keep video features verbatim; resolution/codec metadata matters and the
    # videos themselves are copied unchanged.
    for k, v in orig.items():
        if v.get("dtype") == "video":
            new[k] = v
    new.update({
        "timestamp": {"shape": [1], "dtype": "float32"},
        "frame_index": {"shape": [1], "dtype": "int64"},
        "episode_index": {"shape": [1], "dtype": "int64"},
        "index": {"shape": [1], "dtype": "int64"},
        "task_index": {"shape": [1], "dtype": "int64"},
        "sub_task_index": {"shape": [1], "dtype": "int64"},
    })
    info["features"] = new
    json.dump(info, open(info_path, "w"), indent=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--skip-videos", action="store_true",
                    help="Symlink videos/ instead of copying (saves disk; the "
                         "source must stay mounted).")
    ap.add_argument("--quat-ref-out",
                    help="Where to write the quaternion sign reference. Given "
                         "this, a second pass aligns every episode's quaternions "
                         "to one hemisphere per arm, which callers (training and "
                         "deploy/s1_protocol_bridge) must then honour. Omit only "
                         "to reproduce pre-alignment datasets.")
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    roots = find_dataset_roots(src_root)
    print(f"found {len(roots)} dataset root(s) under {src_root}")

    total_eps = total_frames = 0
    for rel in roots:
        src, dst = src_root / rel, dst_root / rel
        print(f"\n=== {rel} ===")

        for sub in ("meta", "videos"):
            s, d = src / sub, dst / sub
            if not s.exists():
                continue
            if sub == "videos" and args.skip_videos:
                d.parent.mkdir(parents=True, exist_ok=True)
                if not d.exists():
                    d.symlink_to(s.resolve())
                print(f"  {sub}: symlinked")
            else:
                print(f"  {sub}: copying")
                shutil.copytree(s, d, dirs_exist_ok=True)

        pqs = sorted((src / "data").rglob("episode_*.parquet"))
        for pq in tqdm(pqs, desc=f"  {rel.parts[0][:40]}"):
            total_frames += convert_episode(pq, dst / pq.relative_to(src))
            total_eps += 1

        rewrite_info(dst / "meta/info.json")

    print(f"\nTotal: {total_eps} episodes, {total_frames} frames -> {dst_root}")

    if args.quat_ref_out:
        # Second pass, over the whole task at once: Rotation.from_matrix picks
        # between the two quaternion representations of a rotation (q and -q) on
        # numerical grounds, independently per frame and per call, so state and
        # action disagree on sign for a small fraction of frames. Physically
        # nothing moved, but action - state then jumps to L2 = 2.0, and one such
        # frame poisons its entire 50-step chunk. Aligning to one hemisphere per
        # arm removes it; the reference is a task-level statistic, so this cannot
        # be folded into the per-episode conversion above.
        print("\n=== aligning quaternion signs ===")
        align_dataset(dst_root, ref_out=args.quat_ref_out)


if __name__ == "__main__":
    main()
