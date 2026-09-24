"""Protocol routing checks; no model weights, torch, or GPU required."""
import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import deploy.s1_websocket_server as server


STATE_KEY = "cartesian_so3_dict.cartesian_pose_state"
ACTION_KEY = "cartesian_so3_dict.cartesian_pose_command"


def native_config():
    return {
        "pi05_io": {"action_max_dim": 32},
        "states": [{"observation.state.robot.position": {"origin_keys": STATE_KEY}}],
        "actions": [{"action.robot.position": {
            "origin_keys": ACTION_KEY, "subtract_state": False,
        }}],
        "images": [
            {"observation.images.camera_top": {"origin_keys": "images_dict.head.rgb"}},
            {"observation.images.camera_wrist_left": {"origin_keys": "images_dict.left.rgb"}},
            {"observation.images.camera_wrist_right": {"origin_keys": "images_dict.right.rgb"}},
        ],
    }


def observation():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {
        "state": np.arange(34, dtype=np.float64),
        "images": {"cam_high": image, "cam_left_wrist": image, "cam_right_wrist": image},
        "prompt": "open fridge",
    }


class S1WebsocketServerTest(unittest.TestCase):
    def test_native_route_preserves_raw_values_without_quaternion_bridge(self):
        config = native_config()
        expected = np.arange(68, dtype=np.float32).reshape(2, 34)
        policy = SimpleNamespace(
            robot_config=config,
            infer=mock.Mock(return_value={ACTION_KEY: expected}),
            reset=mock.Mock(),
        )
        obs = observation()
        with mock.patch.object(server, "_legacy_bridge", side_effect=AssertionError("bridge called")):
            protocol = server.LingbotAsyncProtocolServer(
                policy, "new_robot_not_in_a_name_list", model_path="unused", quat_ref="auto",
            )
            protocol._reset_policy()
            actions = protocol._infer(obs)
        np.testing.assert_array_equal(actions, expected)
        model_obs = policy.infer.call_args.args[0]
        self.assertNotIn("observation.state", model_obs)
        np.testing.assert_array_equal(model_obs[STATE_KEY], obs["state"])
        self.assertIsNot(model_obs[STATE_KEY], obs["state"])
        self.assertEqual(model_obs["task"], "open fridge")
        self.assertIn("images_dict.head.rgb", model_obs)

    def test_legacy_route_keeps_conversion_and_head_option(self):
        config = server._load_robot_config("s1_stationary_stereo")
        policy = SimpleNamespace(robot_config=config, infer=mock.Mock(return_value={"action": "chunk"}))
        bridge = mock.Mock()
        bridge.state_34d_to_25d.return_value = np.arange(25)
        bridge.action_dict_to_25d.return_value = np.ones((2, 25))
        expected = np.ones((2, 34), dtype=np.float32)
        bridge.action_25d_to_34d.return_value = expected
        with mock.patch.object(server, "_legacy_bridge", return_value=bridge):
            protocol = server.LingbotAsyncProtocolServer(
                policy, "s1_stationary_stereo", preserve_head_from_state=True,
            )
            actions = protocol._infer(observation())
        self.assertIs(actions, expected)
        model_obs = policy.infer.call_args.args[0]
        np.testing.assert_array_equal(model_obs["observation.state"], np.arange(25))
        self.assertIn("images_dict.head_stereo.rgb", model_obs)
        bridge.action_dict_to_25d.assert_called_once_with({"action": "chunk"}, allow_zero_fill=False)
        self.assertTrue(bridge.action_25d_to_34d.call_args.kwargs["preserve_head_from_state"])

    def test_native_rejects_wrong_state_and_action_shapes(self):
        policy = SimpleNamespace(robot_config=native_config(), infer=mock.Mock())
        protocol = server.LingbotAsyncProtocolServer(policy, "unused")
        for shape in ((32,), (1, 34), (2, 17)):
            with self.subTest(state_shape=shape):
                obs = observation()
                obs["state"] = np.zeros(shape)
                with self.assertRaisesRegex(ValueError, "state shape"):
                    protocol._infer(obs)
        policy.infer.assert_not_called()
        for shape in ((32, 32), (1, 2, 34), (34,), (0, 34)):
            with self.subTest(action_shape=shape):
                policy.infer.return_value = {ACTION_KEY: np.zeros(shape)}
                with self.assertRaisesRegex(ValueError, "action shape"):
                    protocol._infer(observation())
        policy.infer.return_value = {"action": np.zeros((2, 34))}
        with self.assertRaisesRegex(ValueError, "must return"):
            protocol._infer(observation())

    def test_reset_reloads_route_and_quaternion_reference_only_for_legacy(self):
        policy = SimpleNamespace(robot_config=native_config())
        configs = iter([
            server._load_robot_config("s1_stationary_stereo"),
            native_config(),
        ])

        def reset(_name):
            policy.robot_config = next(configs)

        policy.reset = mock.Mock(side_effect=reset)
        bridge = mock.Mock()
        with mock.patch.object(server, "_legacy_bridge", return_value=bridge):
            protocol = server.LingbotAsyncProtocolServer(policy, "switching_config", quat_ref="none")
            bridge.load_quat_ref.assert_not_called()
            protocol._reset_policy()
            self.assertFalse(protocol.route.native_so3)
            self.assertEqual(protocol.route.camera_map["head"], "images_dict.head_stereo.rgb")
            bridge.load_quat_ref.assert_called_once_with(None)
            bridge.reset_quat_continuity.assert_called_once()
            protocol._reset_policy()
            self.assertTrue(protocol.route.native_so3)
            self.assertEqual(protocol.route.state_key, STATE_KEY)
            bridge.load_quat_ref.assert_called_once()
            bridge.reset_quat_continuity.assert_called_once()

    def test_invalid_native_config_does_not_fall_back_to_bridge(self):
        for value in (None, {}, False):
            config = native_config()
            config["pi05_io"] = value
            with self.subTest(pi05_io=value), self.assertRaisesRegex(ValueError, "pi05_io"):
                server.S1ProtocolRoute.from_robot_config(config)
        config = copy.deepcopy(native_config())
        config["states"][0]["observation.state.robot.position"]["origin_keys"] = [{STATE_KEY: {}}]
        with self.assertRaisesRegex(ValueError, "origin_keys"):
            server.S1ProtocolRoute.from_robot_config(config)


if __name__ == "__main__":
    unittest.main()
