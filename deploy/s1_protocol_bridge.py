"""
Layout bridge between the astribot runtime protocol (34D) and this repo's
trained S1 model (25D).

Runtime layout (34D) — matches vla_server-chassis_move's
CHASSIS_WITHOUT_HEAD_LAYOUT and the raw dataset before conversion:
    [ 0: 9]  torso          xyz(3) + SO(3) 6D(6)
    [ 9:18]  left arm       xyz(3) + SO(3) 6D(6)
    [18:19]  left gripper
    [19:28]  right arm      xyz(3) + SO(3) 6D(6)
    [28:29]  right gripper
    [29:31]  head           pan, tilt
    [31:34]  chassis        x, y, theta

Model layout (25D) — what tools/convert_s1_dataset.py produced for training:
    [ 0: 4]  waist          x, z, pitch, yaw   (torso 9D compressed)
    [ 4: 7]  left  xyz
    [ 7:11]  left  quat     (xyzw)
    [11:12]  left  gripper
    [12:15]  right xyz
    [15:19]  right quat     (xyzw)
    [19:20]  right gripper
    [20:22]  head
    [22:25]  base

The torso 9->4 compression is lossless for this robot: torso_y and roll are
structurally zero (verified over 403 episodes / 336k frames, max residual 1e-6).
"""
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lingbotvla.data.vla_data.s1_transforms import (
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
    torso_9d_to_waist_4d,
    waist_4d_to_torso_9d,
)

logger = logging.getLogger(__name__)

RUNTIME_DIM = 34
MODEL_DIM = 25

# Runtime 34D slices
R_TORSO = slice(0, 9)
R_LEFT_ARM = slice(9, 18)
R_LEFT_GRIP = slice(18, 19)
R_RIGHT_ARM = slice(19, 28)
R_RIGHT_GRIP = slice(28, 29)
R_HEAD = slice(29, 31)
R_CHASSIS = slice(31, 34)


def _safe_quat(quat, name):
    """Replace degenerate quaternions with identity before handing to scipy.

    A zero quaternion is not a rotation. scipy either raises or yields a
    garbage orientation, and on a real robot a garbage wrist orientation is
    dangerous. Zero-norm rows can appear when an action key was missing and
    got zero-filled upstream, so clamp them here rather than downstream.
    """
    quat = np.asarray(quat, dtype=np.float64)
    norms = np.linalg.norm(quat, axis=-1)
    bad = norms < 1e-6
    if np.any(bad):
        n_bad = int(np.count_nonzero(bad))
        logger.warning(
            "%s: %d/%d degenerate quaternion(s) (norm<1e-6); substituting identity. "
            "This usually means an action key was missing and zero-filled.",
            name, n_bad, quat.shape[0] if quat.ndim > 1 else 1,
        )
        quat = quat.copy()
        quat[bad] = np.array([0.0, 0.0, 0.0, 1.0])
    return quat


def quat_xyzw_to_6d(quat, name="quat"):
    """(..., 4) xyzw quaternion -> (..., 6) rotation, first two matrix columns."""
    quat = _safe_quat(quat, name)
    flat = quat.reshape(-1, 4)
    mats = np.stack([Rotation.from_quat(q).as_matrix() for q in flat])
    mats_t = torch.from_numpy(mats)
    six = rotation_matrix_to_6d(mats_t).numpy()
    return six.reshape(*quat.shape[:-1], 6)


def rot6d_to_quat_xyzw(rot6d):
    """(..., 6) rotation -> (..., 4) xyzw quaternion."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    mats = rotation_6d_to_matrix(torch.from_numpy(rot6d.reshape(-1, 6))).numpy()
    quats = np.stack([Rotation.from_matrix(m).as_quat() for m in mats])
    return quats.reshape(*rot6d.shape[:-1], 4)


# Quaternion sign convention, set from the model's packaged quat_ref file.
#
# A rotation has two quaternion representations, q and -q, and
# Rotation.from_matrix picks between them on numerical grounds (largest
# component positive), so the choice flips arbitrarily as the arm moves.
# Training data for such models is aligned to a fixed hemisphere per arm; the
# same alignment must be applied to incoming state here, or the model sees the
# opposite sign on roughly half the frames and silently mispredicts.
#
# Left as None for models trained before this convention existed -- those were
# fit on unaligned data, so aligning their input would be the mismatch.
_QUAT_REF = None


def load_quat_ref(path):
    """Install the quaternion sign convention from a quat_ref json file.

    Pass None to disable alignment (for models trained on unaligned data).
    Returns the loaded reference dict, or None.
    """
    global _QUAT_REF
    if path is None:
        _QUAT_REF = None
        print("[bridge] quaternion sign alignment: DISABLED")
        return None
    ref = json.loads(Path(path).read_text())["quat_ref_xyzw"]
    _QUAT_REF = {k: np.asarray(v, dtype=np.float64) for k, v in ref.items()}
    for arm, v in _QUAT_REF.items():
        print(f"[bridge] quaternion sign alignment: {arm} REF={np.round(v, 5).tolist()}")
    return _QUAT_REF


def _align_quat(quat, arm):
    """Flip quat into the configured reference hemisphere. Identity if unset.

    q and -q denote the same rotation, so this never changes the pose -- it only
    picks a consistent representative.
    """
    if _QUAT_REF is None:
        return quat
    return -quat if float(np.dot(quat, _QUAT_REF[arm])) < 0.0 else quat


def state_34d_to_25d(state):
    """Runtime 34D state -> model 25D state."""
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape[0] != RUNTIME_DIM:
        raise ValueError(f"Expected {RUNTIME_DIM}D state, got {state.shape[0]}D")

    waist = torso_9d_to_waist_4d(torch.from_numpy(state[R_TORSO]).unsqueeze(0))
    waist = waist.squeeze(0).numpy()

    left = state[R_LEFT_ARM]
    right = state[R_RIGHT_ARM]

    return np.concatenate([
        waist,                                                  # [ 0: 4]
        left[:3],                                               # [ 4: 7]
        _align_quat(rot6d_to_quat_xyzw(left[3:9]), "left"),     # [ 7:11]
        state[R_LEFT_GRIP],                                     # [11:12]
        right[:3],                                              # [12:15]
        _align_quat(rot6d_to_quat_xyzw(right[3:9]), "right"),   # [15:19]
        state[R_RIGHT_GRIP],                                    # [19:20]
        state[R_HEAD],                                          # [20:22]
        state[R_CHASSIS],                                       # [22:25]
    ]).astype(np.float32)


# Model action keys -> (expected dim, whether zero-fill is safe when absent).
# Zero-filling a *translation* means "don't move", which is a safe default.
# Quaternions cannot be zero-filled ([0,0,0,0] is not a rotation), so
# _safe_quat substitutes identity and logs loudly.
_ACTION_SPEC = [
    ("action.waist.position", 4),
    ("action.end.position", 14),
    ("action.effector.position", 2),
    ("action.head.position", 2),
    ("action.base.position", 3),
]


def action_dict_to_25d(action_chunk, allow_zero_fill=True):
    """Policy output dict -> (T, 25) array in model layout.

    action_chunk maps the keys in _ACTION_SPEC to (T, dim) arrays. A missing
    key is zero-filled when allow_zero_fill, with a warning: absent means the
    robot config never mapped that joint group, and holding still is the safe
    interpretation. A *dimension mismatch* is never padded — that indicates the
    robot config and this bridge disagree, and silently padding would misalign
    every downstream slice.
    """
    lengths = {
        k: np.asarray(v).shape[0]
        for k, v in action_chunk.items()
        if isinstance(v, (np.ndarray, list))
    }
    if not lengths:
        raise ValueError("action_chunk contains no array values")
    horizon = max(lengths.values())

    parts = {}
    for key, dim in _ACTION_SPEC:
        if key in action_chunk:
            arr = np.asarray(action_chunk[key], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[-1] != dim:
                raise ValueError(
                    f"{key}: expected last dim {dim}, got {arr.shape[-1]}. "
                    "Robot config and s1_protocol_bridge disagree; refusing to pad."
                )
            parts[key] = arr
        elif allow_zero_fill:
            logger.warning("%s missing from action_chunk; zero-filling %dD", key, dim)
            parts[key] = np.zeros((horizon, dim), dtype=np.float64)
        else:
            raise KeyError(f"{key} missing from action_chunk")

    end = parts["action.end.position"]      # 14 = 2 x (xyz + quat)
    grip = parts["action.effector.position"]

    return np.concatenate([
        parts["action.waist.position"],     # [ 0: 4]
        end[:, 0:3],                        # [ 4: 7]  left xyz
        end[:, 3:7],                        # [ 7:11]  left quat
        grip[:, 0:1],                       # [11:12]
        end[:, 7:10],                       # [12:15]  right xyz
        end[:, 10:14],                      # [15:19]  right quat
        grip[:, 1:2],                       # [19:20]
        parts["action.head.position"],      # [20:22]
        parts["action.base.position"],       # [22:25]
    ], axis=-1)


def action_25d_to_34d(action_25d, state_34d, preserve_head_from_state=False):
    """Model (T, 25) actions -> runtime (T, 34) actions.

    preserve_head_from_state mirrors vla_server's CHASSIS_WITHOUT_HEAD_LAYOUT,
    which does not predict head and instead broadcasts the current head state.
    Our model *was* trained on head, so the default uses its prediction.
    """
    a = np.asarray(action_25d, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[-1] != MODEL_DIM:
        raise ValueError(f"Expected {MODEL_DIM}D actions, got {a.shape[-1]}D")

    torso = waist_4d_to_torso_9d(torch.from_numpy(a[:, 0:4])).numpy()
    left_6d = quat_xyzw_to_6d(a[:, 7:11], "left_quat")
    right_6d = quat_xyzw_to_6d(a[:, 15:19], "right_quat")

    head = a[:, 20:22]
    if preserve_head_from_state:
        state = np.asarray(state_34d, dtype=np.float64).reshape(-1)
        head = np.broadcast_to(state[R_HEAD], (a.shape[0], 2))

    return np.concatenate([
        torso,              # [ 0: 9]
        a[:, 4:7],          # [ 9:12]  left xyz
        left_6d,            # [12:18]
        a[:, 11:12],        # [18:19]  left gripper
        a[:, 12:15],        # [19:22]  right xyz
        right_6d,           # [22:28]
        a[:, 19:20],        # [28:29]  right gripper
        head,               # [29:31]
        a[:, 22:25],        # [31:34]  chassis
    ], axis=-1).astype(np.float32)
