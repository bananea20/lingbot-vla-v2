#!/usr/bin/env python3
"""Serve LingBotVLA over the asynchronous astribot runtime protocol.

The client opens two connections. A ``sender`` pushes observations and a
``receiver`` receives action broadcasts. Only the newest pending observation
is retained while inference is running, matching the Pi0.5 server's behavior.

Robot configs with ``pi05_io`` use the native 34D SO3 runtime layout. Other
configs use the existing 34D/25D bridge in deploy/s1_protocol_bridge.py.

Usage:
  .venv/bin/python -m deploy.s1_websocket_server \
    --model_path output/s1_stationary_head/checkpoints/global_step_60000 \
    --robo_name s1_stationary_head --port 8000
"""
import argparse
import asyncio
import contextlib
import logging
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import websockets
import yaml

from deploy import msgpack_numpy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("s1_server")

# Model camera features -> runtime camera names. Raw dataset keys come from
# the robot config, including head versus stereo camera selection.
CAMERA_FEATURES = {
    "observation.images.camera_top": "head",
    "observation.images.camera_wrist_left": "left_wrist",
    "observation.images.camera_wrist_right": "right_wrist",
}


CAMERA_ALIASES = {
    "head": ("head", "cam_high"),
    "left_wrist": ("left_wrist", "cam_left_wrist"),
    "right_wrist": ("right_wrist", "cam_right_wrist"),
}


def _legacy_bridge():
    # Native SO3 policies do not need the quaternion bridge or its dependencies.
    from deploy import s1_protocol_bridge

    return s1_protocol_bridge


def _load_robot_config(robo_name):
    with (Path("configs/robot_configs") / f"{robo_name}.yaml").open() as stream:
        return yaml.safe_load(stream)


def _native_origin_key(robot_config, section):
    entries = robot_config.get(section)
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError(f"pi05_io requires exactly one {section} feature")
    entry = entries[0]
    if not isinstance(entry, Mapping) or len(entry) != 1:
        raise ValueError(f"Invalid pi05_io {section} feature: {entry!r}")
    feature = next(iter(entry.values()))
    origin_key = feature.get("origin_keys") if isinstance(feature, Mapping) else None
    if not isinstance(origin_key, str) or not origin_key:
        raise ValueError(f"pi05_io {section}.origin_keys must be one raw field name")
    return origin_key


@dataclass(frozen=True)
class S1ProtocolRoute:
    native_so3: bool
    state_key: str
    action_key: str | None
    camera_map: dict[str, str]

    @classmethod
    def from_robot_config(cls, robot_config):
        if not isinstance(robot_config, Mapping):
            raise ValueError("Robot config must be a mapping")
        native_so3 = "pi05_io" in robot_config
        if native_so3 and (
            not isinstance(robot_config["pi05_io"], Mapping)
            or not robot_config["pi05_io"]
        ):
            raise ValueError("pi05_io must be a nonempty configuration mapping")

        camera_map = {}
        for entry in robot_config.get("images", []):
            if not isinstance(entry, Mapping) or len(entry) != 1:
                raise ValueError(f"Invalid robot camera feature: {entry!r}")
            feature_name, feature = next(iter(entry.items()))
            if feature_name not in CAMERA_FEATURES:
                raise ValueError(f"Unsupported S1 camera feature: {feature_name}")
            client_key = CAMERA_FEATURES[feature_name]
            origin_key = feature.get("origin_keys") if isinstance(feature, Mapping) else None
            if not isinstance(origin_key, str) or not origin_key:
                raise ValueError(f"Camera {feature_name} needs one raw origin_keys field")
            if client_key in camera_map:
                raise ValueError(f"Duplicate runtime camera: {client_key}")
            camera_map[client_key] = origin_key
        if not camera_map:
            raise ValueError("Robot config has no S1 camera features")

        return cls(
            native_so3=native_so3,
            state_key=_native_origin_key(robot_config, "states") if native_so3 else "observation.state",
            action_key=_native_origin_key(robot_config, "actions") if native_so3 else None,
            camera_map=camera_map,
        )


def _configure_quat_reference(model_path, quat_ref):
    if quat_ref == "auto":
        mp = Path(model_path)
        found = [candidate for candidate in (
            *sorted((mp / "configs").glob("*quat_ref*.json")),
            *sorted((mp.parent / "configs").glob("*quat_ref*.json")),
            mp.parent / "quat_ref.json",
        ) if candidate.is_file()]
        quat_ref = str(found[0]) if found else None
        logger.info("quat_ref auto-detect: %s",
                    quat_ref or "not found -> alignment DISABLED (assuming a "
                                "pre-alignment model; pass --quat_ref if wrong)")
    elif quat_ref == "none":
        quat_ref = None
    _legacy_bridge().load_quat_ref(quat_ref)


def _get_camera(images, camera_name):
    aliases = CAMERA_ALIASES[camera_name]
    for key in aliases:
        if key in images:
            return images[key]
    raise KeyError(
        f"Missing camera '{camera_name}' (accepted: {list(aliases)}); "
        f"got {sorted(images)}"
    )


def _validate_image(image, camera_name):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"Camera '{camera_name}' must be HWC RGB (prefer uint8 [0,255]); "
            f"got shape {image.shape}. Set astribot_eval image_type='int'."
        )
    return image


def build_observation(obs, robo_name, route=None):
    """Client payload -> the dict LingbotVLAv2Server.infer expects."""
    if route is None:
        route = S1ProtocolRoute.from_robot_config(_load_robot_config(robo_name))
    state34 = np.asarray(obs["state"], dtype=np.float64)
    if route.native_so3:
        if state34.shape != (34,):
            raise ValueError(f"Expected native SO3 state shape (34,), got {state34.shape}")
        if not np.isfinite(state34).all():
            raise ValueError("Native SO3 state contains nonfinite values")
    else:
        state34 = state34.reshape(-1)
    if state34.shape[0] != 34:
        raise ValueError(f"Expected 34D state from client, got {state34.shape[0]}D")

    out = {
        route.state_key: state34.copy() if route.native_so3
        else _legacy_bridge().state_34d_to_25d(state34)
    }

    images = obs["images"]
    for client_key, lerobot_key in route.camera_map.items():
        out[lerobot_key] = _validate_image(
            _get_camera(images, client_key), client_key
        )

    out["task"] = obs.get("prompt", "")
    return out, state34


class LingbotAsyncProtocolServer:
    """Adapt a LingBot policy to the Pi0.5 sender/receiver wire protocol."""

    def __init__(self, policy, robo_name, preserve_head_from_state=False,
                 model_path=None, quat_ref=None):
        self.policy = policy
        self.robo_name = robo_name
        self.preserve_head_from_state = preserve_head_from_state
        self.model_path = model_path
        self.quat_ref = quat_ref
        self._refresh_route()
        self.receivers = set()
        self._latest_observation = None
        self._observation_ready = asyncio.Event()
        self._reset_pending = True
        self._trajectory_generation = 0
        self._packer = msgpack_numpy.Packer()

    def _refresh_route(self):
        robot_config = getattr(self.policy, "robot_config", None)
        if robot_config is None:
            robot_config = _load_robot_config(self.robo_name)
        self.route = S1ProtocolRoute.from_robot_config(robot_config)
        if not self.route.native_so3 and self.quat_ref is not None:
            _configure_quat_reference(self.model_path, self.quat_ref)

    def _reset_policy(self):
        self.policy.reset(self.robo_name)
        self._refresh_route()
        if not self.route.native_so3:
            _legacy_bridge().reset_quat_continuity()

    def request_reset(self):
        """Discard pending input and reset the policy before the next inference."""
        self._latest_observation = None
        self._reset_pending = True
        self._trajectory_generation += 1

    def submit_observation(self, observation):
        """Keep only the newest observation while the model is busy."""
        self._latest_observation = observation
        self._observation_ready.set()

    def _infer(self, observation):
        model_obs, state34 = build_observation(observation, self.robo_name, self.route)
        chunk = self.policy.infer(model_obs)
        if self.route.native_so3:
            if not isinstance(chunk, Mapping) or self.route.action_key not in chunk:
                raise ValueError(f"Native SO3 policy must return {self.route.action_key!r}")
            actions = np.asarray(chunk[self.route.action_key], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 34:
                raise ValueError(f"Expected native SO3 action shape (T, 34), got {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("Native SO3 actions contain nonfinite values")
            return actions
        bridge = _legacy_bridge()
        a25 = bridge.action_dict_to_25d(chunk, allow_zero_fill=False)
        return bridge.action_25d_to_34d(
            a25,
            state34,
            preserve_head_from_state=self.preserve_head_from_state,
        )

    async def _broadcast(self, response):
        packed = self._packer.pack(response)
        for receiver in list(self.receivers):
            try:
                await receiver.send(packed)
            except Exception:
                self.receivers.discard(receiver)

    async def inference_loop(self):
        while True:
            await self._observation_ready.wait()
            self._observation_ready.clear()
            observation = self._latest_observation
            self._latest_observation = None
            if observation is None:
                continue

            generation = self._trajectory_generation
            try:
                if self._reset_pending:
                    self._reset_pending = False
                    try:
                        await asyncio.to_thread(self._reset_policy)
                    except Exception:
                        self._reset_pending = True
                        raise
                actions = await asyncio.to_thread(self._infer, observation)
                if generation != self._trajectory_generation:
                    logger.info("discarding action from reset trajectory")
                    continue
                await self._broadcast(
                    {
                        "actions": actions,
                        "obs_timestamp": float(observation["obs_timestamp"]),
                    }
                )
            except Exception as error:
                logger.error(
                    "inference failed: %s\n%s", error, traceback.format_exc()
                )

    async def _handle_sender(self, ws):
        self.request_reset()
        logger.info("client registered as sender: %s", getattr(ws, "remote_address", "?"))
        async for raw in ws:
            try:
                message = msgpack_numpy.unpackb(raw)
                if isinstance(message, dict) and message.get("reset") == 1:
                    self.request_reset()
                    logger.info("policy reset requested by sender")
                    continue
                self.submit_observation(message)
            except Exception as error:
                logger.warning("invalid sender message: %s", error)

    async def _handle_receiver(self, ws):
        self.receivers.add(ws)
        logger.info("client registered as receiver: %s", getattr(ws, "remote_address", "?"))
        try:
            await ws.wait_closed()
        finally:
            self.receivers.discard(ws)

    async def handler(self, ws):
        peer = getattr(ws, "remote_address", "?")
        logger.info("client connected: %s", peer)
        try:
            initial_message = msgpack_numpy.unpackb(await ws.recv())
            role = initial_message.get("role") if isinstance(initial_message, dict) else None
            if role == "sender":
                await self._handle_sender(ws)
            elif role == "receiver":
                await self._handle_receiver(ws)
            else:
                logger.warning("unknown client role from %s: %r", peer, role)
        except websockets.exceptions.ConnectionClosed:
            logger.info("client disconnected: %s", peer)


def main():
    # Keep the heavyweight model import out of protocol-only tests.
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    p = argparse.ArgumentParser(description="S1 VLA websocket server")
    p.add_argument("--model_path", required=True)
    p.add_argument("--robo_name", required=True,
                   help="Robot config name under configs/robot_configs (for example s1_fridge_pi05)")
    p.add_argument("--norm_path", default=None)
    p.add_argument("--host", default="0.0.0.0")
    # vla_server defaults to 8000; astribot config/config.py uses 9011. Pass
    # explicitly to match whichever the client is actually configured for.
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--use_length", type=int, default=25)
    p.add_argument("--use_compile", action="store_true")
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument(
        "--preserve_head_from_state",
        action="store_true",
        help="Broadcast head from state instead of using the model's prediction "
             "(mirrors vla_server's CHASSIS_WITHOUT_HEAD_LAYOUT).",
    )
    p.add_argument(
        "--quat_ref", default="auto",
        help="quat_ref json fixing the quaternion sign convention. 'auto' "
             "(default) looks for one next to the checkpoint; 'none' disables "
             "alignment, which is correct only for models trained before this "
             "convention existed. A mismatch here is silent and halves "
             "effective accuracy, so the resolved choice is always logged.",
    )
    args = p.parse_args()

    # Validate the route before loading model weights. Native SO3 configs never
    # load quaternion references; all geometric transforms live in Pi05IO.
    S1ProtocolRoute.from_robot_config(_load_robot_config(args.robo_name))

    logger.info("loading %s (%s)", args.model_path, args.robo_name)
    policy = LingbotVLAv2Server(
        path_to_pi_model=args.model_path,
        robot_norm_path=args.norm_path,
        use_length=args.use_length,
        use_bf16=args.use_bf16,
        use_fp32=not args.use_bf16,
        chunk_ret=True,
        use_compile=args.use_compile,
    )
    policy.reset(args.robo_name)
    logger.info("model ready")

    protocol_server = LingbotAsyncProtocolServer(
        policy,
        args.robo_name,
        preserve_head_from_state=args.preserve_head_from_state,
        model_path=args.model_path,
        quat_ref=args.quat_ref,
    )

    async def serve():
        inference_task = asyncio.create_task(protocol_server.inference_loop())
        try:
            async with websockets.serve(
                protocol_server.handler,
                args.host,
                args.port,
                compression=None,
                max_size=100 * 1024 * 1024,
                max_queue=10,
            ):
                logger.info("listening on ws://%s:%d", args.host, args.port)
                await asyncio.Future()
        finally:
            inference_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await inference_task

    asyncio.run(serve())


if __name__ == "__main__":
    main()
