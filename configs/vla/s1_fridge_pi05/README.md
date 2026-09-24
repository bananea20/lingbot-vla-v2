# Native SO(3) fridge training

This configuration reads the original 34D Cartesian state/command columns from
both `data/open_refrigerator_with_move` subsets. It uses pi0.5's pose selection,
SO(3) delta transform (`state_axis=base`, `action_axis=self`) and quantile
normalization. Each rotation is represented by the first two matrix rows.

The model keeps Lingbot's continuous state input, 50-step action horizon and 55D
projection sizes. The selected 32D physical features are normalized, then
scattered according to `pi05_io.model_indices`; predictions are gathered in the
reverse order before unnormalization and native 34D reconstruction. The two head
dimensions are copied from the observation when restoring commands.

## Channel mapping

All indices are zero-based. Noncontiguous rotation slots are intentional.

| Physical feature | Model indices |
| --- | --- |
| Torso xyz | `30, 39, 31` |
| Torso rotation 6D | `32, 33, 40, 41, 42, 43` |
| Left arm xyz | `14, 15, 16` |
| Left arm rotation 6D | `17, 18, 19, 20, 51, 52` |
| Left gripper | `28` |
| Right arm xyz | `21, 22, 23` |
| Right arm rotation 6D | `24, 25, 26, 27, 53, 54` |
| Right gripper | `29` |
| Chassis | `36, 37, 38` |

Arm XYZ, gripper and chassis channels retain their locations. Their numeric
distribution still depends on the new frame/delta transform and dataset
normalization. Rotation channels change representation; indices `[39:44]`
repurpose part of the pretrained hand channels, and `[51:55]` use the reserved
channels. Those weights are retained and finetuned, rather than claiming an
unchanged rotation representation. Joint `[0:14]`, head `[34:36]` and remaining
hand `[44:51]` channels are masked out of supervision.

## Compute statistics

Run from the repository root using the existing training environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python tools/compute_pi05_norm_stats.py \
  --chunk-size 50 --torch-threads 1 \
  --output assets/norm_stats/s1_fridge_pi05.json
```

The tool streams episode parquet files without decoding video or loading model
weights. It uses every observation and pools all 50 action positions, repeating
the final action at episode boundaries as in the reference data loader. The
output records the processing configuration, episode manifest and coverage.
Existing output files require an explicit `--overwrite`.

The generated shared file covers all 226 episodes and 448,026 observations from
the two subsets, with 22,401,300 action vectors across the 50-step windows.

These statistics use the packed physical order, not the final 55D slot order.
Change the horizon, selected features or physical transform only together with
recomputed statistics. A pure model slot scatter does not change the physical
values being normalized.

## Training entry points

After statistics and the data/inference adapters have been verified, the direct
training entry point is:

```bash
OMP_NUM_THREADS=1 .venv/bin/torchrun \
  --nnodes=1 --nproc-per-node=8 --master-port=62620 \
  tasks/vla/train_lingbotvla.py \
  configs/vla/s1_fridge_pi05/s1_fridge_pi05.yaml
```

The supplied batch sizes target eight GPUs. Supply the site's existing FFmpeg
library and logging environment when required, as for the other training
configurations. Checkpoints go to the new `s1_fridge_pi05` output directory;
existing fridge checkpoints and statistics are not reused as SO(3) outputs.
`run_train.sh` has no `fridge_pi05` variant; use this direct entry point.

## Compare mapped and sequential layouts

`s1_fridge_pi05.yaml` uses the channel table above.
`s1_fridge_pi05_sequential.yaml` puts the identical packed physical features in
model `[0:32]`, with `[32:55]` masked. Both configurations share the same native
datasets, pi0.5 transforms, normalization file, images, pretrained checkpoint,
continuous state input, horizon and training settings. Their dataset-list and
run identifiers differ to select the corresponding robot configuration and
keep outputs separate. No projection weights are permuted between runs: this
comparison measures how each layout adapts from the existing initialization.

Run one version on each of two separate nodes. The 2026-09-24 experiment uses
`10.2.14.189` for mapped and `10.2.35.154` for sequential; both were verified to
have eight H100 80GB GPUs. Both use seed 42 and a global batch of 64
(`8 microbatch × 8 GPUs × 1 accumulation`). W&B runs offline, with local
TensorBoard and stdout metrics also available.

The shared `.venv` still depends on a node-local Python interpreter at
`/root/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu`. On the new node,
that same interpreter was copied from the existing node, and the FFmpeg 4
runtime packages `libavdevice58` and `libavfilter7` were installed. CUDA,
FlashAttention and actual LeRobot video decoding passed before launch.

On the node assigned to the mapped version:

```bash
OMP_NUM_THREADS=1 WANDB_MODE=offline .venv/bin/torchrun \
  --nnodes=1 --nproc-per-node=8 --master-port=62620 \
  tasks/vla/train_lingbotvla.py \
  configs/vla/s1_fridge_pi05/s1_fridge_pi05.yaml \
  --train.gradient_accumulation_steps 1 --train.seed 42
```

On the other node assigned to the sequential version:

```bash
OMP_NUM_THREADS=1 WANDB_MODE=offline .venv/bin/torchrun \
  --nnodes=1 --nproc-per-node=8 --master-port=62620 \
  tasks/vla/train_lingbotvla.py \
  configs/vla/s1_fridge_pi05/s1_fridge_pi05_sequential.yaml \
  --train.gradient_accumulation_steps 1 --train.seed 42
```

The shared normalization makes physical action losses comparable in scale.
Compare the action loss and its trend at equal optimizer steps; the total loss
also contains visual auxiliary terms. Channel locations change the association
between physical dimensions and sampled flow noise, so identical seeds do not
make every physical noise sample equal. Training loss alone does not establish
which policy executes the fridge-opening motion better; use the same held-out
trajectories and execution checks for that conclusion.
