---
date: 2026-09-07
topic: ckpt清理与hf_ckpt推理包
tags: [存储清理, checkpoint, hf_ckpt, 推理部署, NAS]
status: in-progress
session_id: f4e479bf-cc7f-436e-b8ad-be0e6d70492b
related:
  - "[[2026-08-18-S1训练归档与wandb修复]]"
---

## TL;DR

- **干了什么**:清理 `baifu_cache/lingbot-vla-v2/output` 的中间 checkpoint,每个 run 只留末尾 step,回收约 2.4T
- **结论 / 产物**:
  - `s1_stationary_head_v2` / `s1_stationary_stereo_v2`:各 12 → 1 个 ckpt(留 `global_step_60000`),回收 ~1.6T
  - `s1_fridge_head_qrot`:训练已于 09-01 21:44 跑完到 step 60000,12 → 1 个 ckpt,回收 ~800G
  - `baifu_cache` 2.7T → 1.1T(09-02 实测);09-07 复查涨回 1.5T,因新 run `s1_fridge_head_signfix` 已产 12 个 ckpt / 874G
  - **单个 ckpt 恒为 73G / 283 文件**,构成:`optimizer` 26G + `model` 24G + `hf_ckpt` 24G + `extra_state` 72K
- **关键决策**:
  - 只留末尾 step,不留中间 —— 训练已收敛且无回滚需求
  - 推理包只取 `hf_ckpt`,不含 optimizer/model —— 后两者是 DCP 续训态
  - 删前必查训练进程 + tfevents 时间戳,不看目录名
- **踩了什么坑**:
  - `du -sh . 2>/dev/null` 在 NAS 上吞错并少报(103G vs 实际 1.1T)
  - 容器内无 `bc` / `pigz` / `pv`,压缩率实测脚本全线报错
  - 训练"看起来跑完"≠ 跑完,`fridge_head_qrot` 隔天又多出 4 个 ckpt
- **下次接手要看**:
  - 打包**未做完**,见「后续 / 未完」
  - `s1_fridge_head_signfix` 874G / 12 ckpt 尚未清理,是当前最大单项
  - NAS 已 94T/100T(剩 6.1T),且为共享盘

—— 以上五条决定 Claude 是否继续往下读 ——

## 上下文 / 起点

起点是一句「看看这里存储大小结构」。`/kpfs-cognition/baifu_cache` 占 2.7T,其中 `lingbot-vla-v2` 2.6T,
全部落在 `output/*/checkpoints`。其余目录合计不到 100G(`hf` 27G、`prompt_cls` 18G、
`humandata_viz` 6.7G、`humandata_camera1_h264` 6.5G、`ks3_methylene_fix` 6.4G)。

当时三个 run 存着多份中间 ckpt:`s1_stationary_stereo_v2` 874G/12、`s1_stationary_head_v2` 874G/12、
`s1_fridge_head_qrot` 583G/8(仍在训练)。另有 4 个 run 各留 1 个 ckpt。

## 方法与决策

### 为什么只留末尾 step,不做"隔一个留一个"

两个 `_v2` run 的 ckpt 从 step 5000 均匀存到 60000(每 5000 一存,间隔约 2h),
tfevents 最后写入 08-27 22:24 后不再变动,训练已收敛到目标 step。中间 step 的用途只有
对比曲线和回滚,而曲线已在 tfevents 里(225MB,完整保留),回滚需求不存在。

### 删前的三重校验(不看目录名判断状态)

1. `ps aux | grep -iE 'train|torchrun|deepspeed'` —— 确认无活跃写入
2. `runs/` 下 tfevents 的 mtime —— 确认训练确实收尾
3. `find <ckpt> -type f | wc -l` 逐个比对 —— 283 文件是完整 ckpt 的指纹,残缺的会偏离

删除脚本内置 abort:保留目标不存在、或检出训练进程,直接 `exit 1` 不删。

### `fridge_head_qrot` 为什么隔天才删

09-01 首次查看时它 583G/8 ckpt,且 `ps` 里能看到 `ssh 10.2.27.97 ... run_train.sh fridge_head_qrot`
的 launcher 还在,当时判断为"正在训练,等跑完再清"。09-02 复查:ckpt 已到 12 个、末尾 `global_step_60000`
(与两个 `_v2` run 终点一致)、tfevents 停在 09-01 21:40、launcher 进程消失 —— 确认跑完才动手。

### 推理包只取 hf_ckpt(用户提示后确立)

拆开 73G 看构成后确认分工:
- `model/` + `optimizer/` = 256 个 `.distcp` 分片,DCP 分布式续训状态,**推理不需要**
- `hf_ckpt/` = 标准 HF 格式,6 个 safetensors(约 24G)+ `config.json` / `tokenizer.json` /
  `chat_template.jinja` / `preprocessor_config.json` / `video_preprocessor_config.json` 等
- 配 `workspace/lingbot-vla-v2/deploy/` 那套(`s1_websocket_server.py`、`lingbot_vla_v2_policy.py`、
  `s1_protocol_bridge.py`)即可起推理服务

所以"只打包推理相关" = 只 tar `hf_ckpt`,24G 而非 73G。

### 关于"之前打包过"的核对结论

用户提到之前打包过推理相关内容。**在本节点搜索未找到产物**:`baifu_cache` (maxdepth 4) 与
`workspace` (maxdepth 3) 下所有 `*.tar*` / `*.zst` / `*.tgz` 只有 ffmpeg 源码包、tailscale
安装包、npm 离线包、两个几百 K 的 eval 包,无 hf_ckpt 量级归档。
`2026-08-18-S1训练归档与wandb修复.md` 里的"归档"是把超参/结果写进 `docs/training_runs/`,
不是打包权重。用户已表示此项无需深究,不必再考古 —— 但"推理只需 `hf_ckpt`"这个结论来自
用户提示,是本次的关键收获,见上一节。

## 产物

清理后现存 ckpt 全景(09-07 实测,`baifu_cache/lingbot-vla-v2/output/`):

| run | 大小 | ckpt 数 |
|---|---|---|
| `s1_fridge_head_signfix` | 874G | 12 ← **未清理,当前最大项** |
| `s1_fridge` | 73G | 1 |
| `s1_fridge_head` | 73G | 1 |
| `s1_fridge_head_qrot` | 73G | 1(本次清理) |
| `s1_stationary_head` | 73G | 1 |
| `s1_stationary_head_v2` | 73G | 1(本次清理) |
| `s1_stationary_stereo` | 73G | 1 |
| `s1_stationary_stereo_v2` | 73G | 1(本次清理) |
| `s1_fridge_qrot` | 空 | 0 |

保留的 ckpt 均为 `global_step_60000`,283 文件完整。
非 ckpt 产物(`runs/` tfevents 约 152–225M、`images/`、`model_assets/`)每 run 一两百 M,未动。

清理命令(带 abort 保护,可复用):

```bash
cd /kpfs-cognition/baifu_cache/lingbot-vla-v2/output/<run> || exit 1
KEEP=global_step_60000
[ -d "checkpoints/$KEEP" ] || { echo "ABORT: missing $KEEP"; exit 1; }
ps aux 2>/dev/null | grep -iE 'run_train|torchrun|deepspeed' | grep -v grep \
  && { echo "ABORT: training active"; exit 1; }
for c in $(ls -1 checkpoints | sort -t_ -k3 -n); do
  [ "$c" = "$KEEP" ] && { echo "KEEP $c"; continue; }
  rm -rf -- "checkpoints/$c" && echo "DEL  $c"
done
```

环境事实:`nproc` 192;有 `zstd` (`/opt/conda/bin/zstd`) 和 `tar`;**无** `bc` / `pigz` / `pv`。
`/tmp` 在 overlay 上,3.5T 总 2.4T 可用,适合放打包中间产物。

## 踩坑

**1. `du` 静默少报**
现象:清理后 `du -sh .` 报 103G,与 `lingbot-vla-v2` 自身的 1022G 矛盾。
原因:命令带了 `2>/dev/null`,NAS 上的遍历错误被吞掉,统计提前结束。
怎么绕:核对总量时不要屏蔽 stderr,并交叉验证(`du -sh <子目录>` 之和、`df -h`)。重跑得 1.1T。

**2. 容器缺基础工具,测量脚本空跑**
现象:压缩率实测脚本每行 `bc: command not found`,输出 `s L1: % of orig | MB/s` 全空。
原因:镜像里没有 `bc`,连带 `pigz` / `pv` 也没有。
怎么绕:算术改用 `awk`(`awk -v a=$cs -v b=$sz 'BEGIN{printf "%.1f%%", a*100/b}'`);
并行压缩用 `zstd -T0`(它自带多线程,192 核可用),不依赖 `pigz`。

**3. 把"正在训练"当成"已结束"的风险**
现象:09-01 看 `fridge_head_qrot` 是 8 个 ckpt,若当时按"留最新"清理,会把 step 40000 当终点,
而真正的终点是隔天产出的 60000。
原因:ckpt 数量是训练进度的快照,不是终态。
怎么绕:以 launcher 进程 + tfevents mtime 判定训练是否收尾,两者都静了再动手。

**4. 空间不一定归自己**
`/kpfs-cognition` 是共享 NAS(100T)。09-01 是 92T used / 8.4T avail,
09-07 已 94T used / 6.1T avail —— 本次腾出的 2.4T 并未体现为可用量增长,被其他写入吃掉了。
所以清理不能当作"给自己攒空间",要清就尽早清。

## 后续 / 未完

**打包 `hf_ckpt` 这件事没做完**,进度停在压缩率实测(被工具缺失打断)。接手要做:

1. **先测压缩率再决定压不压**。safetensors 是高熵原始张量,通用压缩预期收益很小
   (**本次未测出数字,勿引用任何比率**)。测法:取 1G 样本,`zstd -1` 和 `-3` 各跑一次,
   用 `awk` 算比率和吞吐。若压缩率 > 95%,直接 `tar` 不压(省 CPU 和时间)。
2. **打包范围**:`checkpoints/global_step_60000/hf_ckpt/` 整个目录(24G),
   不含 `model/` `optimizer/`。是否连带 `deploy/` 那几个 py 一起打成自包含推理包,需用户确认。
3. **落盘位置**:`/tmp` 有 2.4T 可用;若要长期存或跨节点,走 ks3(有 `ks3util` skill,
   bucket `ks3://stardust/ks3-regular/`)。
4. **哪些 run 要打**:本次只确认了 `s1_fridge_head_qrot` 的构成,共 7 个 run 各有 1 个
   `global_step_60000`,打哪几个未定。

**另一件待办**:`s1_fridge_head_signfix` 874G / 12 ckpt(09-07 新发现,本次未动)。
清理前照样走三重校验 —— 这个 run 的训练状态未确认,可能仍在跑。
