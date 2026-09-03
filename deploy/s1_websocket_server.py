#!/usr/bin/env python3
"""Serve LingBotVLA over the asynchronous astribot runtime protocol.

The client opens two connections. A ``sender`` pushes observations and a
``receiver`` receives action broadcasts. Only the newest pending observation
is retained while inference is running, matching the Pi0.5 server's behavior.

Internally the 34D runtime layout is bridged to the 25D layout the model was
trained on; see deploy/s1_protocol_bridge.py.

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
from pathlib import Path

import numpy as np
import websockets

from deploy import msgpack_numpy
from deploy.s1_protocol_bridge import (
    action_25d_to_34d,
    action_dict_to_25d,
    load_quat_ref,
    state_34d_to_25d,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("s1_server")

# Client camera names -> the raw LeRobot keys each robot config maps from.
CAMERA_MAP = {
    "s1_stationary_head": {
        "head": "images_dict.head.rgb",
        "left_wrist": "images_dict.left.rgb",
        "right_wrist": "images_dict.right.rgb",
    },
    "s1_stationary_stereo": {
        "head": "images_dict.head_stereo.rgb",
        "left_wrist": "images_dict.left.rgb",
        "right_wrist": "images_dict.right.rgb",
    },
    "s1_fridge_head_qrot": {
        "head": "images_dict.head.rgb",
        "left_wrist": "images_dict.left.rgb",
        "right_wrist": "images_dict.right.rgb",
    },
}


CAMERA_ALIASES = {
    "head": ("head", "cam_high"),
    "left_wrist": ("left_wrist", "cam_left_wrist"),
    "right_wrist": ("right_wrist", "cam_right_wrist"),
}


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


def build_observation(obs, robo_name):
    """Client payload -> the dict LingbotVLAv2Server.infer expects."""
    state34 = np.asarray(obs["state"], dtype=np.float64).reshape(-1)
    if state34.shape[0] != 34:
        raise ValueError(f"Expected 34D state from client, got {state34.shape[0]}D")

    out = {"observation.state": state_34d_to_25d(state34)}

    cam_map = CAMERA_MAP[robo_name]
    images = obs["images"]
    for client_key, lerobot_key in cam_map.items():
        out[lerobot_key] = _validate_image(
            _get_camera(images, client_key), client_key
        )

    out["task"] = obs.get("prompt", "")
    return out, state34


class LingbotAsyncProtocolServer:
    """Adapt a LingBot policy to the Pi0.5 sender/receiver wire protocol."""

    def __init__(self, policy, robo_name, preserve_head_from_state=False):
        self.policy = policy
        self.robo_name = robo_name
        self.preserve_head_from_state = preserve_head_from_state
        self.receivers = set()
        self._latest_observation = None
        self._observation_ready = asyncio.Event()
        self._reset_pending = True
        self._trajectory_generation = 0
        self._packer = msgpack_numpy.Packer()

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
        model_obs, state34 = build_observation(observation, self.robo_name)
        chunk = self.policy.infer(model_obs)
        a25 = action_dict_to_25d(chunk, allow_zero_fill=False)
        return action_25d_to_34d(
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
                    await asyncio.to_thread(self.policy.reset, self.robo_name)
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
    p.add_argument("--robo_name", required=True, choices=sorted(CAMERA_MAP))
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

    # A rotation is represented by both q and -q, and the 6D->quat conversion
    # picks between them on numerical grounds, so the sign flips arbitrarily as
    # the arm moves. Models trained on sign-aligned data need the same alignment
    # applied to incoming state here.
    quat_ref = args.quat_ref
    if quat_ref == "auto":
        mp = Path(args.model_path)
        found = [c for c in (
            *sorted((mp / "configs").glob("*quat_ref*.json")),
            *sorted((mp.parent / "configs").glob("*quat_ref*.json")),
            mp.parent / "quat_ref.json",
        ) if c.is_file()]
        quat_ref = str(found[0]) if found else None
        logger.info("quat_ref auto-detect: %s",
                    quat_ref or "not found -> alignment DISABLED (assuming a "
                                "pre-alignment model; pass --quat_ref if wrong)")
    elif quat_ref == "none":
        quat_ref = None
    load_quat_ref(quat_ref)

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
