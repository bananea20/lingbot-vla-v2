# S1 Stationary Training Run - 2026-08-13

## Overview

Two VLA models trained on S1 humanoid robot pick-and-move task with different camera configurations.

**Dataset**: `pick_and_move_from_table_to_wash_25d` (converted from 34D cartesian)  
**Base Model**: lingbot-vla-v2-6b  
**Training Duration**: ~25 hours per model  
**Hardware**: 8×H100 per training job

## Results

| Model | Camera Setup | Final Loss | VLA Loss | Steps | Checkpoints |
|-------|--------------|------------|----------|-------|-------------|
| **head** | 3-cam (top + dual wrist) | 0.0115 | 0.0078 | 60000 | 12 @ 5k intervals |
| **stereo** | 2-cam (dual wrist stereo) | 0.0112 | 0.0072 | 60000 | 12 @ 5k intervals |

### Convergence Metrics (Step 60000)

**Head Model**:
```
Loss: 0.0115
VLA_Loss: 0.0078
Depth_Loss: 0.2685
Future_Depth_Loss: 0.3653
FutureVideo_Loss: 0.0271
GradNorm: 0.0588
```

**Stereo Model**:
```
Loss: 0.0112
VLA_Loss: 0.0072
Depth_Loss: 0.3042
Future_Depth_Loss: 0.4029
FutureVideo_Loss: 0.0250
GradNorm: 0.0643
```

Both models converged to similar final losses, as expected for the same task from different viewpoints.

## Training Configuration

### Data Format

Converted 25D action space (FLU frame):
- `[0:4]` waist: x, z, pitch, yaw
- `[4:11]` left arm: xyz + quat + gripper
- `[11:19]` right arm: xyz + quat + gripper
- `[19:22]` head: pan, tilt
- `[22:25]` base: x, y, theta

See `configs/robot_configs/s1_stationary_head.yaml` for full joint mapping.

### Camera Configurations

**Head** (`configs/vla/s1_stationary/s1_stationary_head.yaml`):
- `camera_top`: overhead view
- `camera_wrist_left`: left wrist-mounted
- `camera_wrist_right`: right wrist-mounted

**Stereo** (`configs/vla/s1_stationary/s1_stationary_stereo.yaml`):
- `camera_wrist_left`: left wrist-mounted
- `camera_wrist_right`: right wrist-mounted
- Depth estimation from stereo pair

### Normalization

Statistics: `assets/norm_stats/s1_stationary.json`  
Method: meanstd for all joints  
Inference: automatic denormalization via `feature_transform.unapply()`

## Checkpoints

**Location**:
- Head: `output/s1_stationary_head/checkpoints/global_step_{10000..60000}/`
- Stereo: `output/s1_stationary_stereo/checkpoints/global_step_{10000..60000}/`

**Size**: 874 GB per model (12 checkpoints @ ~73 GB each)

**Format**: FSDP sharded checkpoints + HuggingFace conversion

## Inference

### Open-Loop Evaluation

```bash
python scripts/open_loop_eval.py \
  --model_path output/s1_stationary_head/checkpoints/global_step_60000 \
  --robo_name s1_stationary_head \
  --data_path <validation_dataset> \
  --policy qwen3vl \
  --traj_ids 0 1 2 3 4
```

Output: dictionary of denormalized actions in original units (meters, radians, normalized gripper [0,1])

### Real Robot Deployment

Use `experiment/robotwin/eval_policy_client_lingbotvla.py` (may require protocol adaptation).

## Known Issues

### Training Completion

Stereo training completed all 60000 steps but crashed during checkpoint save with NCCL timeout (rank 2 BROADCAST operation exceeded 600s watchdog). **All data and final checkpoint (global_step_60000) were successfully saved before the crash.** No data loss occurred.

### Wandb Integration

Initial wandb runs were empty due to:
1. `sync_tensorboard=True` with `SummaryWriter` created before `wandb.init()` (no hook)
2. No explicit `wandb.log()` calls in original code
3. Proxy configuration issues on remote nodes

Fixed in this commit (see `docs/fixes/wandb_integration_fix.md`).

## Training Logs

- Head: `logs/train_head_20260813_121712.log`
- Stereo: `logs/train_stereo_20260813_161046.log`

TensorBoard events:
- Head: `output/s1_stationary_head/runs/`
- Stereo: `output/s1_stationary_stereo/runs/`

## Next Steps

- [ ] Upload checkpoints to model registry (HuggingFace/KS3)
- [ ] Run quantitative open-loop evaluation on validation set
- [ ] Test real-robot deployment
- [ ] Compare head vs stereo performance on manipulation tasks
- [ ] Evaluate depth prediction quality (stereo model)
