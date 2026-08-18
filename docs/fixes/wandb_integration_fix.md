# Wandb Integration Fix

## Problem

Training runs showed empty wandb dashboards despite `sync_tensorboard=True` being enabled. Metrics were successfully written to TensorBoard but never reached wandb servers.

## Root Causes

### 1. Constructor Ordering Bug (Primary)

**Issue**: `SummaryWriter` was instantiated **before** `wandb.init()` in `tasks/vla/train_lingbotvla.py:563-565`.

```python
# BEFORE (broken)
writer = AsyncTBWriter(log_dir=log_dir)  # line 563
if args.train.use_wandb:
    wandb.init(..., sync_tensorboard=True)  # line 565
```

**Why it fails**: `sync_tensorboard=True` works by patching `torch.utils.tensorboard.SummaryWriter` at init time. A writer already instantiated is never hooked, so `add_scalar()` calls only write tfevents files. **Zero history records are produced** on the wandb side—the uploader has nothing to send.

**Evidence**:
- Diagnostic agent reproduced both directions with real venv + server
- Live run `qnue251x` datastore scan: `{history: 0, output_raw: 18627, stats: 1830}` — console lines and GPU stats present, metrics absent
- Heartbeat current while metrics empty → transport up, payload missing

### 2. No Explicit Metric Logging

Original code had **zero `wandb.log()` calls**. All metrics went through `writer.add_scalar()` only. When `sync_tensorboard` failed due to issue #1, no fallback path existed.

### 3. Network Configuration (Secondary)

**Issue**: Both training nodes had proxy exports in `~/.bashrc` after the non-interactive guard:

```bash
# ~/.bashrc line 6
[ -z "$PS1" ] && return

# ... 109 lines later ...
export https_proxy=http://10.5.0.191:6666  # never executed for ssh jobs
```

SSH-launched training jobs are non-interactive, so the shell returned before reaching proxy configuration. All outbound HTTPS connections timed out.

**Note**: Network diagnostics revealed curl/requests with explicit proxy worked fine (200 response, 0.67s). The proxy itself was healthy; training processes simply never inherited the variables.

### 4. Conflicting Step Parameter

With `sync_tensorboard=True`, explicit `step=` in `wandb.log()` triggers a warning and is ignored:
```
wandb: WARNING Step cannot be set when using tensorboard syncing
```

## Fix

### Code Changes (tasks/vla/train_lingbotvla.py)

**1. Removed `sync_tensorboard` and reordered construction**

```python
# AFTER (fixed)
if args.train.use_wandb:
    # Deliberately NOT using sync_tensorboard: that works by patching
    # torch.utils.tensorboard at init time, so a SummaryWriter created
    # earlier is never hooked and no history records are produced. It
    # also conflicts with the explicit wandb.log(..., step=...) below.
    wandb.init(
        project=args.train.wandb_project,
        name=args.train.wandb_name,
        config={**vars(args.model), **vars(args.data), **vars(args.train)},
    )
# Constructed after wandb.init so ordering stays correct if TB syncing
# is ever re-enabled.
writer = AsyncTBWriter(log_dir=log_dir)
```

**2. Added explicit batched `wandb.log()` call**

Mirrored all TensorBoard scalars to wandb with a single batched call per step (lines 1069-1116):

```python
wandb_metrics = {
    "training/loss": total_loss,
    "training/vla_loss": total_vla_loss,
    "training/depth_loss": depth_loss,
    "training/future_depth_loss": future_depth_loss,
    "training/future_video_loss": future_video_loss,
    "training/grad_norm": grad_norm,
    "training/lr": current_lr,
    # ... (full list in code)
}

if args.train.use_wandb:
    try:
        wandb.log(
            {k: _tb_scalar(v) for k, v in wandb_metrics.items()},
            step=global_step,
        )
    except Exception as e:
        logger.warning(f"wandb.log failed at step {global_step}: {repr(e)}")
```

Wrapped in try/except so telemetry failures never crash training.

### Environment Changes

**1. Fixed proxy configuration in ~/.bashrc (both nodes)**

Moved proxy exports **above** the `[ -z "$PS1" ] && return` guard so non-interactive shells inherit them:

```bash
# ~/.bashrc now starts with:
PROXY_URL=http://10.5.0.191:6666
NO_PROXY_LIST="apt.ksyun.cn,10.0.0.0/8,127.0.0.1,localhost,pypi.ksyun.cn,198.18.0.0/15"
export http_proxy="$PROXY_URL"  https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"  HTTPS_PROXY="$PROXY_URL"
export no_proxy="$NO_PROXY_LIST"  NO_PROXY="$NO_PROXY_LIST"

# ... then the PS1 guard comes after
```

Both lowercase (Python `requests`) and uppercase (Go `net/http`, used by wandb-core 0.21+) are set.

**2. Added proxy + timeout to run_train.sh**

Double insurance in case `.bashrc` is later reverted:

```bash
PROXY="${PROXY:-http://10.5.0.191:6666}"
NOPROXY="apt.ksyun.cn,10.0.0.0/8,127.0.0.1,localhost,pypi.ksyun.cn,198.18.0.0/15"
export http_proxy="$PROXY"  https_proxy="$PROXY"
export HTTP_PROXY="$PROXY"  HTTPS_PROXY="$PROXY"
export no_proxy="$NOPROXY"  NO_PROXY="$NOPROXY"

# Both nodes saw occasional wandb init timeouts (default 90s)
export WANDB__SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=300
```

## Verification

Stereo training restarted with fixes applied:
- Run `82l550vk` created in <2 seconds (previously timed out after 90s)
- API confirmed metrics landing: `training/loss`, `training/vla_loss`, `training/grad_norm` all present
- Step 0: loss=1.2695, vla_loss=1.2344, grad_norm=2.2867 (matches log exactly)

## Alternative: Offline Sync for Completed Runs

For runs that finished before the fix (e.g., head training), TensorBoard events can be uploaded retroactively:

```bash
wandb sync --sync-tensorboard \
  --project lingbotvla-s1 --entity 562696940qq-astribot \
  output/s1_stationary_head/runs
```

This is read-only on the tfevents file and safe to run while training continues.

## Lessons

1. **Constructor ordering matters** when a library uses monkey-patching. Always instantiate the target (SummaryWriter) *after* calling the patcher (wandb.init).
2. **Explicit is better than implicit**. `sync_tensorboard` is fragile (timing-dependent, no errors, just silent data loss). Explicit `wandb.log()` fails loudly and is testable.
3. **Shell init guards break automation**. `[ -z "$PS1" ] && return` is common but deadly for ssh-launched jobs. Put essential env vars above it.
4. **Network issues present as silent failures** in telemetry. Heartbeat alive + metrics empty = look at payload generation, not transport.

## Files Changed

- `tasks/vla/train_lingbotvla.py`: removed `sync_tensorboard`, added explicit `wandb.log()`, reordered writer construction
- `run_train.sh`: added proxy exports and timeout overrides
- `~/.bashrc` (both nodes): moved proxy above non-interactive guard

## Related Issues

- wandb/wandb#1534: sync_tensorboard with existing writer
- pytorch/pytorch#95736: torch.distributed NCCL timeout (unrelated to wandb, affected stereo post-save)
