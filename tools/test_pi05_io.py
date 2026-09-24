"""Numerical checks for the native pi0.5 action IO adapter.

Run from the repository root with ``python tools/test_pi05_io.py``.
Use --reference to compare against a checkout of the original BaseProcessor.py.
Only the reference class AST is loaded, so its unrelated training dependencies
are not needed. The production adapter never depends on this external file.
"""

import sys
import importlib.util
import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import types

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_MODULE_PATH = Path(__file__).resolve().parents[1] / "lingbotvla/data/vla_data/pi05_io.py"
_SPEC = importlib.util.spec_from_file_location("pi05_io_test_module", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
Pi05IO = _MODULE.Pi05IO


CONFIG = {
    "raw_dim": 34,
    "action_max_dim": 32,
    "action_valid_dim": {
        "torso": [0, 9],
        "left_arm": [9, 18],
        "left_gripper": [18, 19],
        "right_arm": [19, 28],
        "right_gripper": [28, 29],
        "chassis": [31, 34],
    },
    "action_mode": "delta_so3",
    "action_delta_mask": [9, 9, -1, 9, -1, 3],
    "state_axis": "base",
    "action_axis": "self",
}

DEFAULT_REFERENCE = Path("/kpfs-cognition/baifu/workspace/robot-learning-framework-aug/src/utils/BaseProcessor.py")
MODEL_INDICES = [30, 39, 31, 32, 33, 40, 41, 42, 43,
                 14, 15, 16, 17, 18, 19, 20, 51, 52, 28,
                 21, 22, 23, 24, 25, 26, 27, 53, 54, 29, 36, 37, 38]


def _load_reference(path):
    tree = ast.parse(path.read_text())
    names = {"_MaskSegment", "_MaskInfo", "StateActionTransform"}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseProcessor")
    select = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "create_select_pad_transform")
    select.decorator_list = []
    nodes.append(select)
    module = types.ModuleType("pi05_reference_for_io_test")
    sys.modules[module.__name__] = module
    module.__dict__.update({"torch": torch, "np": np, "dataclass": dataclass})
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    program = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(program, str(path), "exec"), module.__dict__)
    return module


def _random_native(batch=4, horizon=7):
    state = torch.randn(batch, 34)
    actions = torch.randn(batch, horizon, 34)
    for source in (state, actions):
        for start in (0, 9, 19):
            matrix, _ = torch.linalg.qr(torch.randn(*source.shape[:-1], 3, 3))
            matrix[..., :, 2] *= torch.linalg.det(matrix).unsqueeze(-1)
            source[..., start + 3:start + 9] = matrix[..., :2, :].reshape(*source.shape[:-1], 6)
    return state, actions


def _expect_error(call):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError("expected explicit validation error")


def check_model_slots():
    io = Pi05IO({**CONFIG, "model_indices": MODEL_INDICES})
    for shape in ((32,), (7, 32), (2, 7, 32)):
        values = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape) + 1
        original = values.clone()
        model = io.pack_model(values, 55)
        assert model.shape == (*shape[:-1], 55)
        for coordinate, model_index in enumerate(MODEL_INDICES):
            assert torch.equal(model[..., model_index], values[..., coordinate])
        mask = io.model_mask(55)
        assert mask.dtype == torch.bool and mask.shape == (55,)
        assert mask.sum().item() == 32 and not model[..., ~mask].any()
        assert torch.equal(io.unpack_model(model), values)
        assert torch.equal(values, original)
        assert not torch.equal(model[..., mask], values), "mask ordering must differ for this mapping"
    values = torch.arange(32, dtype=torch.float32)
    model = io.pack_model(values, 55)
    assert torch.equal(model[14:17], values[9:12])   # left xyz
    assert torch.equal(model[21:24], values[19:22])  # right xyz
    assert torch.equal(model[28:30], values[[18, 28]])  # both grippers
    assert torch.equal(model[36:39], values[29:32])  # chassis xyz / yaw
    assert torch.equal(model[[30, 31]], values[[0, 2]])  # torso x / z
    assert not model[:14].any()  # arm joint-angle slots remain unused
    assert not model[34:36].any()  # head slots remain unused
    default = Pi05IO(CONFIG)
    assert torch.equal(default.pack_model(values, 55)[:32], values)
    assert default.model_mask(55).sum() == 32
    padded = Pi05IO({**CONFIG, "action_max_dim": 55})
    assert padded.model_mask(55).sum() == 32

    _expect_error(lambda: Pi05IO({**CONFIG, "model_indices": MODEL_INDICES[:-1]}))
    _expect_error(lambda: Pi05IO({**CONFIG, "model_indices": MODEL_INDICES[:-1] + [30]}))
    _expect_error(lambda: Pi05IO({**CONFIG, "model_indices": [-1] + MODEL_INDICES[1:]}))
    _expect_error(lambda: Pi05IO({**CONFIG, "model_indices": [True] + MODEL_INDICES[1:]}))
    _expect_error(lambda: Pi05IO({**CONFIG, "model_indices": [30.0] + MODEL_INDICES[1:]}))
    _expect_error(lambda: io.pack_model(values, 54))
    _expect_error(lambda: io.model_mask(54))
    _expect_error(lambda: io.model_mask(55.0))
    _expect_error(lambda: io.pack_model(torch.zeros(31), 55))
    _expect_error(lambda: io.pack_model(torch.tensor(1.0), 55))
    _expect_error(lambda: io.unpack_model(torch.zeros(54)))
    _expect_error(lambda: io.unpack_model(torch.tensor(1.0)))
    bad_packed, bad_model = values.clone(), model.clone()
    bad_packed[0] = float("nan")
    bad_model[14] = float("nan")
    _expect_error(lambda: io.pack_model(bad_packed, 55))
    _expect_error(lambda: io.unpack_model(bad_model))


def check_validation_and_layouts():
    state, actions = _random_native()
    io = Pi05IO(CONFIG)
    old_state, old_actions = state.clone(), actions.clone()
    packed_state, packed_actions = io.preprocess(state, actions)
    restored = io.postprocess(state, packed_actions)
    assert torch.equal(state, old_state) and torch.equal(actions, old_actions)
    assert torch.allclose(io.select(restored), io.select(actions), atol=2e-6)
    assert torch.equal(restored[..., 29:31], state[:, None, 29:31].expand(-1, actions.shape[1], -1))
    assert torch.equal(packed_actions[..., 18], actions[..., 18])
    assert torch.equal(packed_actions[..., 28], actions[..., 28])
    assert torch.allclose(packed_actions[..., 29:32], actions[..., 31:34] - state[:, None, 31:34])
    for index in range(state.shape[0]):
        single_state, single_actions = io.preprocess(state[index], actions[index])
        torch.testing.assert_close(single_state, packed_state[index], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(single_actions, packed_actions[index], atol=2e-6, rtol=1e-5)

    # The raw observation is the restoration anchor even if the normalized
    # model state has been clipped, replaced, or mutated independently.
    packed_state.fill_(0)
    assert torch.allclose(io.postprocess(old_state, packed_actions), restored, atol=2e-6)
    padded = Pi05IO({**CONFIG, "action_max_dim": 55})
    pad_state, pad_actions = padded.preprocess(state, actions)
    assert pad_state.shape == (4, 55) and pad_actions.shape == (4, 7, 55)
    assert not pad_actions[..., 32:].any()
    assert torch.allclose(padded.postprocess(state, pad_actions), restored, atol=2e-6)
    assert torch.allclose(padded.postprocess(state, pad_actions[..., :32]), restored, atol=2e-6)

    full_ranges = dict(CONFIG["action_valid_dim"])
    full_ranges.pop("chassis")
    full_ranges.update({"head": [29, 31], "chassis": [31, 34]})
    full = Pi05IO({**CONFIG, "action_max_dim": 34, "action_valid_dim": full_ranges,
                   "action_delta_mask": [9, 9, -1, 9, -1, 2, 3]})
    _, full_actions = full.preprocess(state, actions)
    assert torch.allclose(full.postprocess(state, full_actions), actions, atol=2e-6)

    # Reference absolute mode only works for its default axes. Our adapter also
    # provides the inverse for explicit torso-frame absolute configurations.
    for state_axis in ("base", "torso", "abstorso"):
        for action_axis in ("self", "base", "torso"):
            absolute = Pi05IO({**CONFIG, "action_mode": "absolute", "state_axis": state_axis,
                              "action_axis": action_axis})
            _, absolute_actions = absolute.preprocess(state, actions)
            result = absolute.postprocess(state, absolute_actions)
            torch.testing.assert_close(io.select(result), io.select(actions), atol=2e-6, rtol=1e-5)

    _expect_error(lambda: Pi05IO({**CONFIG, "action_max_dim": 31}))
    _expect_error(lambda: Pi05IO({**CONFIG, "action_valid_dim": {"a": [0, 9], "b": [8, 17]}}))
    _expect_error(lambda: Pi05IO({**CONFIG, "action_delta_mask": [9, 9, -1, 9, 1, 2]}))
    _expect_error(lambda: io.select(torch.zeros(33)))
    _expect_error(lambda: io.preprocess(state, actions[:, :, :-1]))
    _expect_error(lambda: io.preprocess(state, actions[:2]))
    bad_state = state.clone()
    bad_state[0, 9] = float("nan")
    _expect_error(lambda: io.select(bad_state))
    bad_rotation = state.clone()
    bad_rotation[0, 12:18] = 0
    _expect_error(lambda: io.preprocess(bad_rotation, actions))


def check_reference_parity(path):
    reference = _load_reference(path)
    states, targets = _random_native(batch=3)
    max_error = 0.0
    for state_axis in ("base", "torso", "abstorso"):
        for action_axis in ("self", "base", "torso"):
            config = {**CONFIG, "state_axis": state_axis, "action_axis": action_axis}
            io = Pi05IO(config)
            transform = reference.StateActionTransform(
                "delta_so3", tuple(config["action_delta_mask"]),
                state_axis=state_axis, action_axis=action_axis,
                action_valid_dim=config["action_valid_dim"],
            )
            select = reference.create_select_pad_transform(["state", "actions"], config["action_valid_dim"], 32)
            for state, target in zip(states, targets):
                sample = select({"state": state.clone(), "actions": target.clone()})
                raw_packed_state = sample["state"].clone()
                torch.testing.assert_close(io.select(state), sample["state"], rtol=0, atol=0)
                torch.testing.assert_close(io.select(target), sample["actions"], rtol=0, atol=0)
                transform.apply_transforms(sample["state"], sample["actions"])
                actual_state, actual_actions = io.preprocess(state, target)
                torch.testing.assert_close(actual_state, sample["state"], atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(actual_actions, sample["actions"], atol=2e-6, rtol=1e-5)
                max_error = max(max_error, float((actual_actions - sample["actions"]).abs().max()))
                restored = transform.restore_transforms(raw_packed_state, sample["actions"])
                torch.testing.assert_close(io.select(io.postprocess(state, actual_actions)), restored, atol=3e-6, rtol=1e-5)

    # Joint deltas also match the original implementation on actual joint
    # segment widths. No rotation interpretation is imposed on joint values.
    joint_config = {
        "action_valid_dim": {"torso": [0, 4], "left_arm": [4, 11], "left_gripper": [11, 12],
                             "right_arm": [12, 19], "right_gripper": [19, 20], "head": [20, 22], "chassis": [22, 25]},
        "raw_dim": 25, "action_max_dim": 32, "action_mode": "delta_joints",
        "action_delta_mask": [4, 7, -1, 7, -1, 2, 3],
    }
    joint_io = Pi05IO(joint_config)
    joint_state, joint_actions = torch.randn(25), torch.randn(7, 25)
    joint_ref = reference.StateActionTransform("delta_joints", tuple(joint_config["action_delta_mask"]))
    expected_state, expected_actions = joint_io.select(joint_state), joint_io.select(joint_actions)
    joint_ref.apply_transforms(expected_state, expected_actions)
    actual_state, actual_actions = joint_io.preprocess(joint_state, joint_actions)
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
    torch.testing.assert_close(actual_actions, expected_actions, atol=0, rtol=0)
    torch.testing.assert_close(joint_io.postprocess(joint_state, actual_actions), joint_actions)
    print(f"reference parity passed (9 frame combinations, max action error {max_error:.3g})")


def _rotation(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    z = torch.zeros_like(yaw)
    o = torch.ones_like(yaw)
    rz = torch.stack((cy, -sy, z, sy, cy, z, z, z, o), -1).reshape(-1, 3, 3)
    ry = torch.stack((cp, z, sp, z, o, z, -sp, z, cp), -1).reshape(-1, 3, 3)
    return rz @ ry


def _pose(xyz, yaw, pitch):
    matrix = _rotation(yaw, pitch)
    return torch.cat((xyz, matrix[:, :2, :].reshape(-1, 6)), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    torch.manual_seed(7)
    io = Pi05IO(CONFIG)
    assert io.selected_dim == 32 and io.action_max_dim == 32
    state = torch.zeros(34)
    state[0:9] = _pose(torch.tensor([[0.2, 0.0, 1.1]]), torch.tensor([0.2]), torch.tensor([0.1]))[0]
    state[9:18] = _pose(torch.tensor([[0.5, -0.2, 1.0]]), torch.tensor([-0.4]), torch.tensor([0.3]))[0]
    state[18] = 0.0
    state[19:28] = _pose(torch.tensor([[0.5, 0.2, 1.0]]), torch.tensor([0.3]), torch.tensor([-0.2]))[0]
    state[28] = 0.0
    state[29:31] = torch.tensor([0.1, -0.2])
    state[31:34] = torch.tensor([1.0, -0.5, 0.4])

    target = state.unsqueeze(0).repeat(4, 1)
    target[:, 9] += torch.tensor([0.00, 0.05, 0.10, 0.15])
    target[:, 18] = 0.0
    target[:, 31:34] += torch.tensor([0.1, -0.2, 0.03])

    selected = io.select(state)
    assert selected.shape == (32,)
    assert torch.allclose(selected[29:32], state[31:34])
    pre_state, pre_action = io.preprocess(state, target)
    assert pre_state.shape == (32,) and pre_action.shape == (4, 32)
    assert torch.allclose(pre_action[:, 29:32], target[:, 31:34] - state[31:34])
    assert torch.allclose(pre_action[:, 18], target[:, 18])
    restored = io.postprocess(state, pre_action)
    assert restored.shape == (4, 34)
    assert torch.allclose(restored[:, :29], target[:, :29], atol=2e-5)
    assert torch.allclose(restored[:, 31:34], target[:, 31:34], atol=2e-5)
    assert torch.allclose(restored[:, 29:31], state[29:31].expand(4, 2))

    absolute = Pi05IO({**CONFIG, "action_mode": "absolute"})
    abs_state, abs_actions = absolute.preprocess(state, target)
    assert torch.allclose(abs_actions[:, 29:32], target[:, 31:34])
    assert torch.allclose(absolute.postprocess(state, abs_actions), restored, atol=2e-5)

    batch_state = torch.stack((state, state))
    batch_target = torch.stack((target, target))
    batch_state[1, 9:12] += 0.01
    batch_target[1, :, 9:12] += 0.01
    batch_pre_state, batch_pre_action = io.preprocess(batch_state, batch_target)
    batch_restored = io.postprocess(batch_state, batch_pre_action)
    assert batch_pre_state.shape == (2, 32) and batch_pre_action.shape == (2, 4, 32)
    assert batch_restored.shape == (2, 4, 34)
    assert torch.allclose(batch_restored[..., :29], batch_target[..., :29], atol=2e-5)
    check_validation_and_layouts()
    check_model_slots()
    if args.reference.is_file():
        check_reference_parity(args.reference)
    else:
        print(f"reference parity skipped: {args.reference} is unavailable")
    print("pi05_io numerical checks passed")


if __name__ == "__main__":
    main()
