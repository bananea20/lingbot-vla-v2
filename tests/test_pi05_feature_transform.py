"""Check native pi0.5 geometry across FeatureTransform's model channel mapping."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from lingbotvla.data.vla_data.utils import FeatureTransform


RAW_STATE = "cartesian_so3_dict.cartesian_pose_state"
RAW_ACTION = "cartesian_so3_dict.cartesian_pose_command"
STATE = "observation.state.robot.position"
ACTION = "action.robot.position"
MODEL_INDICES = [
    30, 39, 31, 32, 33, 40, 41, 42, 43,
    14, 15, 16, 17, 18, 19, 20, 51, 52,
    28,
    21, 22, 23, 24, 25, 26, 27, 53, 54,
    29,
    36, 37, 38,
]
CHUNK = 3


class TokenizerStub:
    def __call__(self, prompts, **kwargs):
        shape = (len(prompts), kwargs["max_length"])
        return {"input_ids": torch.ones(shape, dtype=torch.long),
                "attention_mask": torch.ones(shape, dtype=torch.bool)}


def make_sample():
    """Known native row rotations and positions, independent of Pi05IO helpers."""
    state = np.zeros(34, dtype=np.float32)
    rotations = [
        Rotation.from_euler("xyz", [0.15, -0.2, 0.35]).as_matrix(),
        Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        Rotation.from_euler("xyz", [-0.4, 0.1, -0.25]).as_matrix(),
    ]
    positions = ([0.1234567, 0.013, 0.8123456],
                 [1.2345678, -0.8234567, 0.7123456],
                 [-0.7345678, 0.9234567, 1.1123456])
    for start, rotation, position in zip((0, 9, 19), rotations, positions):
        state[start:start + 3] = position
        state[start + 3:start + 9] = rotation[:2].reshape(6)
    state[18], state[28] = 0.0, 0.625
    state[29:31] = [0.3333333, -0.4444444]
    state[31:34] = [12.345678, -5.234567, 1.345678]
    actions = np.repeat(state[None], CHUNK, axis=0)
    world_delta = np.array([0.03, -0.04, 0.02])
    for step in range(CHUNK):
        for start, rotation in zip((0, 9, 19), rotations):
            actions[step, start:start + 3] += world_delta * (step + 1)
            target = rotation @ Rotation.from_euler(
                "xyz", [0.02, -0.03, 0.04 * (step + 1)],
            ).as_matrix()
            actions[step, start + 3:start + 9] = target[:2].reshape(6)
        actions[step, 28] = 0.4 + 0.05 * step
        actions[step, 29:31] = [2.0, -2.0]  # Deliberately unselected commands.
        actions[step, 31:34] += [0.07 * (step + 1), -0.02, 0.03]
    return torch.from_numpy(state), torch.from_numpy(actions)


def reference_selected(raw_state, raw_actions):
    """pi0.5 self delta using matrices: R_s.T dp and R_s.T R_a."""
    state = raw_state.numpy().astype(np.float64)
    actions = raw_actions.numpy().astype(np.float64)
    selected_state = np.concatenate((state[:29], state[31:34]))
    selected_actions = np.concatenate((actions[:, :29], actions[:, 31:34]), axis=-1)

    def row_matrix(six):
        # Independent SO(3) reconstruction; scipy accepts the resulting full R.
        first = six[..., :3]
        first = first / np.linalg.norm(first, axis=-1, keepdims=True)
        second = six[..., 3:] - np.sum(six[..., 3:] * first, axis=-1, keepdims=True) * first
        second /= np.linalg.norm(second, axis=-1, keepdims=True)
        return np.stack((first, second, np.cross(first, second)), axis=-2)

    for start in (0, 9, 19):
        rs = row_matrix(state[start + 3:start + 9])
        ra = row_matrix(actions[:, start + 3:start + 9])
        selected_state[start + 3:start + 9] = rs[:2].reshape(6)
        selected_actions[:, start:start + 3] = (
            rs.T @ (actions[:, start:start + 3] - state[start:start + 3]).T
        ).T
        selected_actions[:, start + 3:start + 9] = (rs.T @ ra)[:, :2].reshape(CHUNK, 6)
    selected_actions[:, 29:32] -= selected_state[29:32]
    return torch.tensor(selected_state, dtype=torch.float32), torch.tensor(selected_actions, dtype=torch.float32)


def reference_normalize(value, stats):
    """Match BaseProcessor's float32 load sanitization and quantile clamp."""
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    high = np.where(high - low == 0.0, low + np.float32(1e-4), high)
    low, high = torch.from_numpy(low), torch.from_numpy(high)
    scale = torch.where(high - low == 0.0, 1e-6, high - low)
    return (((value.float() - low) / scale) * 2 - 1).clamp(-1.1, 1.1)


class Pi05FeatureTransformTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="pi05_feature_transform_")
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        self.stats = {}
        for key in (STATE, ACTION):
            low = np.full(32, -0.25) if key == STATE else -2.0 - np.arange(32) / 50
            high = np.full(32, 0.25) if key == STATE else 2.0 + np.arange(32) / 40
            # Constant left gripper tests the reference's load-time eps handling.
            low[18] = high[18] = 0.0
            self.stats[key] = {
                "mean": np.zeros(32).tolist(), "std": np.zeros(32).tolist(),
                "q01": low.tolist(), "q99": high.tolist(),
            }
        norm_path = folder / "norm.json"
        norm_path.write_text(json.dumps({"norm_stats": self.stats}))
        config = {
            "pi05_io": {
                "action_valid_dim": {"torso": [0, 9], "left_arm": [9, 18],
                    "left_gripper": [18, 19], "right_arm": [19, 28],
                    "right_gripper": [28, 29], "chassis": [31, 34]},
                "raw_dim": 34, "action_max_dim": 32,
                "action_mode": "delta_so3", "action_delta_mask": [9, 9, -1, 9, -1, 3],
                "state_axis": "base", "action_axis": "self", "model_indices": MODEL_INDICES,
            },
            "states": [{STATE: {"origin_keys": RAW_STATE}}],
            "actions": [{ACTION: {"origin_keys": RAW_ACTION, "subtract_state": False}}],
            "images": [], "norm_stats": str(norm_path),
        }
        robot_path = folder / "robot.yaml"
        robot_path.write_text(yaml.safe_dump(config, sort_keys=False))
        data_config = SimpleNamespace(
            joints=["{'robot.position': 32}"], cameras=[],
            norm_type=["{'robot.position': 'pi05_quantile'}"],
        )
        model_config = SimpleNamespace(max_action_dim=55, max_state_dim=55, tokenizer_max_length=8)
        processor = SimpleNamespace(tokenizer=TokenizerStub(), image_processor=SimpleNamespace(merge_size=2))
        self.transform = FeatureTransform(
            robot_path, data_config, model_config, processor,
            disabled_image_features=True, chunk_size=CHUNK,
        )
        self.state, self.actions = make_sample()

    def training_item(self):
        return {RAW_STATE: self.state.clone(), RAW_ACTION: self.actions.clone(),
                f"{RAW_ACTION}_is_pad": torch.zeros(CHUNK, dtype=torch.bool), "task": "open fridge"}

    def test_self_delta_normalization_and_noncontiguous_model_channels(self):
        output = self.transform.apply(self.training_item())
        selected_state, selected_actions = reference_selected(self.state, self.actions)
        # Left wrist is +90 degrees: self delta must swap/negate world xy.
        torch.testing.assert_close(selected_actions[0, 9:12], torch.tensor([-0.04, -0.03, 0.02]), atol=1e-6, rtol=0)
        expected_state = reference_normalize(selected_state, self.stats[STATE])
        expected_actions = reference_normalize(selected_actions, self.stats[ACTION])
        self.assertEqual(output["state"].shape, (55,))
        self.assertEqual(output["actions"].shape, (CHUNK, 55))
        torch.testing.assert_close(output["state"][MODEL_INDICES], expected_state, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(output["actions"][:, MODEL_INDICES], expected_actions, atol=1e-6, rtol=1e-6)
        # Scalar and Cartesian channels keep the requested pretrained positions.
        for packed, model in ((9, 14), (18, 28), (28, 29), (29, 36), (30, 37), (31, 38)):
            torch.testing.assert_close(output["actions"][:, model], expected_actions[:, packed])
        mask = torch.zeros(55, dtype=torch.bool)
        mask[MODEL_INDICES] = True
        for name in ("state_joint_mask", "action_joint_mask"):
            torch.testing.assert_close(output[name], mask)
            self.assertEqual(int(output[name].sum()), 32)
        torch.testing.assert_close(output["joint_mask"], mask.repeat(CHUNK, 1))
        self.assertEqual(int(torch.count_nonzero(output["state"][~mask])), 0)
        self.assertEqual(int(torch.count_nonzero(output["actions"][:, ~mask])), 0)
        torch.testing.assert_close(output["pi05_raw_state"], self.state, atol=0, rtol=0)

    def test_unapply_uses_raw_anchor_after_state_clipping_and_bfloat16(self):
        output = self.transform.apply(self.training_item())
        self.assertEqual(float(output["state"][14]), float(torch.tensor(1.1)))
        output["state"] = output["state"].to(torch.bfloat16)
        self.assertEqual(output["pi05_raw_state"].dtype, torch.float32)
        # Noisy unused model coordinates must not enter the reordered action.
        output["actions"][:, ~output["action_joint_mask"]] = 123.0
        restored = self.transform.unapply(output)
        expected = self.actions.clone()
        expected[:, 29:31] = self.state[29:31]
        self.assertEqual(restored[RAW_ACTION].shape, (CHUNK, 34))
        torch.testing.assert_close(restored[RAW_ACTION], expected, atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(restored[RAW_STATE], self.state, atol=0, rtol=0)
        torch.testing.assert_close(restored[RAW_ACTION][:, 29:31], self.state[29:31].repeat(CHUNK, 1), atol=0, rtol=0)

    def test_policy_eval_retains_anchor_and_uses_same_model_mask(self):
        output = self.transform.apply({RAW_STATE: self.state.clone(), "task": "open fridge"}, policy_eval=True)
        self.assertEqual(output["actions"].shape, (CHUNK, 55))
        self.assertEqual(int(output["action_joint_mask"].sum()), 32)
        _, selected_actions = reference_selected(self.state, self.actions)
        predicted = reference_normalize(selected_actions, self.stats[ACTION])
        output["actions"] = torch.zeros(CHUNK, 55)
        output["actions"][:, MODEL_INDICES] = predicted
        output["state"] = output["state"].to(torch.bfloat16)
        restored = self.transform.unapply(output)
        expected = self.actions.clone()
        expected[:, 29:31] = self.state[29:31]
        torch.testing.assert_close(restored[RAW_ACTION], expected, atol=2e-6, rtol=1e-6)

    def test_pi05_quantile_sanitizes_constant_gripper_and_clamps_like_reference(self):
        loaded = self.transform.normalizer.norm_stats[ACTION]
        self.assertEqual(loaded["q01"].dtype, np.float32)
        self.assertEqual(loaded["std"][18], np.float32(1e-4))
        self.assertEqual(loaded["q99"][18], np.float32(1e-4))
        values = torch.zeros(CHUNK, 32)
        values[:, 18] = torch.tensor([0.0, 1e-3, -1e-3])
        values[0, 0], values[1, 0] = 100.0, -100.0
        actual = self.transform.normalizer.normalize({ACTION: values})[ACTION]
        expected = reference_normalize(values, self.stats[ACTION])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual[:, 18], torch.tensor([-1.0, 1.1, -1.1]), atol=0, rtol=0)
        fixed = self.transform.normalizer.unnormalize({ACTION: actual[:1]})[ACTION]
        self.assertEqual(float(fixed[0, 18]), 0.0)


if __name__ == "__main__":
    unittest.main()
