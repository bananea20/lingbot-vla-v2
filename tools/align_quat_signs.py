"""Make quaternion sign conventions consistent across a 25D dataset.

A rotation has two quaternion representations, q and -q. scipy's
Rotation.from_matrix picks whichever puts the largest-magnitude component
positive, so when two components are nearly tied the sign flips between
consecutive frames -- and independently between state and action, which are
converted in separate calls. Physically nothing moved; numerically all four
components negate, and action - state explodes to L2 = 2.0 (the maximum
possible distance between unit vectors).

Fix: pick one hemisphere per task per arm and align every frame to it. The
reference is the mean orientation over the whole task (Markley's method --
the top eigenvector of sum q q^T, which is invariant to each q's sign, so it
can be computed before the signs are fixed). Flipping q -> -q is an identity
transform on the rotation, so no physical quantity changes.

'w >= 0' would be the textbook choice but fails here: it means "reference =
zero rotation", and these wrists sit 88 deg away from that on average, with
|w| reaching 0.00002 -- straddling the 180 deg boundary where that rule
itself becomes unstable.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ARMS = {"left": slice(7, 11), "right": slice(15, 19)}
COLS = ("observation.state", "action")


def unit(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def angle_deg(q1, q2):
    """Rotation angle between two quaternions, invariant to either's sign."""
    d = np.abs((unit(q1) * unit(q2)).sum(-1)).clip(max=1.0)
    return 2 * np.degrees(np.arccos(d))


def mean_orientation(Q):
    """Markley: top eigenvector of sum q q^T. Invariant to each q's sign."""
    _, vecs = np.linalg.eigh(Q.T @ Q)
    ref = vecs[:, -1]
    # The eigenvector's own sign is arbitrary; fix it so the run is repeatable.
    if ref[np.abs(ref).argmax()] < 0:
        ref = -ref
    return ref


def compute_refs(files):
    acc = {k: [] for k in ARMS}
    for f in files:
        df = pd.read_parquet(f, columns=list(COLS))
        for col in COLS:
            a = np.stack(df[col].values)
            for name, sl in ARMS.items():
                acc[name].append(unit(a[:, sl]))
    return {k: mean_orientation(np.concatenate(v)) for k, v in acc.items()}


def align(arr, refs):
    """Flip each frame's quaternions into the reference hemisphere, in place."""
    for name, sl in ARMS.items():
        q = unit(arr[:, sl])
        arr[:, sl] = np.where((q @ refs[name])[:, None] < 0, -q, q)


def align_dataset(root, ref_out=None, dry_run=False):
    """Align every episode under `root` to one hemisphere per arm.

    Verifies that no rotation angle changed before writing anything, and exits
    if that fails. Importable so converters can run this as a second pass.
    """
    args = argparse.Namespace(root=root, ref_out=ref_out, dry_run=dry_run)

    files = sorted(Path(args.root).rglob("episode_*.parquet"))
    if not files:
        sys.exit(f"no parquet under {args.root}")
    print(f"{len(files)} episodes under {args.root}", flush=True)

    refs = compute_refs(files)
    for name, r in refs.items():
        print(f"  REF[{name}] = {np.round(r, 6).tolist()}", flush=True)

    worst = {"ref_angle": 0.0, "adjacent": 0.0, "delta": 0.0}
    flips_before = flips_after = 0
    rewritten = 0

    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        before = {c: np.stack(df[c].values) for c in COLS}
        after = {c: before[c].copy() for c in COLS}
        for c in COLS:
            align(after[c], refs)

        for name, sl in ARMS.items():
            sb, ab = unit(before["observation.state"][:, sl]), unit(before["action"][:, sl])
            sa, aa = after["observation.state"][:, sl], after["action"][:, sl]
            flips_before += int(((sb[1:] * sb[:-1]).sum(1) < 0).sum())
            flips_after += int(((sa[1:] * sa[:-1]).sum(1) < 0).sum())
            # Every rotation angle must survive the flip untouched.
            ref = refs[name][None]
            worst["ref_angle"] = max(worst["ref_angle"],
                                     np.abs(angle_deg(sb, ref) - angle_deg(sa, ref)).max())
            worst["adjacent"] = max(worst["adjacent"],
                                    np.abs(angle_deg(sb[:-1], sb[1:]) - angle_deg(sa[:-1], sa[1:])).max())
            worst["delta"] = max(worst["delta"],
                                 np.abs(angle_deg(sb, ab) - angle_deg(sa, aa)).max())

        if not args.dry_run:
            for c in COLS:
                df[c] = list(after[c])
            df.to_parquet(f, index=False)
            rewritten += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    print(f"\nsign flips along time: {flips_before} -> {flips_after}")
    print("rotation angles unchanged (must all be 0):")
    for k, v in worst.items():
        print(f"  {k:10s} max change {v:.3e} deg")
    ok = max(worst.values()) < 1e-9 and flips_after == 0
    print(f"\n{'rewrote ' + str(rewritten) + ' files' if rewritten else 'dry run, nothing written'}")
    if not ok:
        sys.exit("VERIFICATION FAILED -- physical quantities changed or flips remain")

    if args.ref_out is not None:
        Path(args.ref_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.ref_out).write_text(json.dumps(
            {"quat_ref_xyzw": {k: v.tolist() for k, v in refs.items()},
             "note": "Align state/action quaternions into this hemisphere before use. "
                     "Inference must apply the same alignment to incoming state."},
            indent=2) + "\n")
        print(f"REF written to {args.ref_out}")

    return refs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref-out", default=None,
                    help="Where to write the REF constants. Omit for a "
                         "verification-only run.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    align_dataset(args.root, ref_out=args.ref_out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
