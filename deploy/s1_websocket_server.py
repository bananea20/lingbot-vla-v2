#!/usr/bin/env python3
"""
WebSocket server exposing this repo's trained S1 checkpoint over the astribot
runtime protocol, so astribot_eval-frank connects without any client change.

Wire protocol (identical to vla_server-chassis_move):
  recv msgpack: {"images": {"head","left_wrist","right_wrist"}, "state": (34,),
                 "prompt": str}
  send msgpack: {"actions": (T, 34)}

Internally the 34D runtime layout is bridged to the 25D layout the model was
trained on; see deploy/s1_protocol_bridge.py.

Usage:
  .venv/bin/python -m deploy.s1_websocket_server \
    --model_path output/s1_stationary_head/checkpoints/global_step_60000 \
    --robo_name s1_stationary_head --port 8000
"""
import argparse
import asyncio
import logging
import traceback
from pathlib import Path

import numpy as np
import websockets

from deploy import msgpack_numpy
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
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
}


def build_observation(obs, robo_name):
    """Client payload -> the dict LingbotVLAv2Server.infer expects."""
    state34 = np.asarray(obs["state"], dtype=np.float64).reshape(-1)
    if state34.shape[0] != 34:
        raise ValueError(f"Expected 34D state from client, got {state34.shape[0]}D")

    out = {"observation.state": state_34d_to_25d(state34)}

    cam_map = CAMERA_MAP[robo_name]
    images = obs["images"]
    for client_key, lerobot_key in cam_map.items():
        if client_key not in images:
            raise KeyError(f"Missing camera '{client_key}'; got {sorted(images)}")
        out[lerobot_key] = np.asarray(images[client_key])

    out["task"] = obs.get("prompt", "")
    return out, state34


def main():
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

    packer = msgpack_numpy.Packer()

    async def handler(ws):
        peer = getattr(ws, "remote_address", "?")
        logger.info("client connected: %s", peer)
        try:
            async for raw in ws:
                try:
                    obs = msgpack_numpy.unpackb(raw)
                    model_obs, state34 = build_observation(obs, args.robo_name)
                    chunk = policy.infer(model_obs)
                    a25 = action_dict_to_25d(chunk)
                    a34 = action_25d_to_34d(
                        a25, state34,
                        preserve_head_from_state=args.preserve_head_from_state,
                    )
                    await ws.send(packer.pack({"actions": a34}))
                except Exception as e:
                    # Never drop the connection: the client runs a 250Hz control
                    # loop and reconnecting mid-episode is worse than one bad step.
                    logger.error("inference failed: %s\n%s", e, traceback.format_exc())
                    await ws.send(packer.pack({"error": str(e)}))
        except websockets.exceptions.ConnectionClosed:
            logger.info("client disconnected: %s", peer)

    async def serve():
        async with websockets.serve(handler, args.host, args.port, max_size=None):
            logger.info("listening on ws://%s:%d", args.host, args.port)
            await asyncio.Future()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
