---
date: 2026-09-10
topic: delta参数化重构-quaternion_local与se2_local
tags: [lingbot-vla-v2, s1_fridge, delta-action, quaternion, se2, relative_type, norm_stats]
status: in-progress
session_id: d09aac20-faed-4af3-aa5c-acfff6836827
related:
  - "[[2026-09-07-ckpt清理与hf_ckpt推理包]]"
---

## TL;DR

- **干了什么**:诊断「静止时旋转震荡」→ 改数据(失败)→ 换参数化(qlocal 变体已训完 60000 步待上机)
- **结论 / 产物**:
  - `s1_fridge_head_qlocal` 训完:`baifu_cache/lingbot-vla-v2/output/s1_fridge_head_qlocal/`,12 ckpt,29h21m,末 VLA 0.0028
  - 新代码:`ee_pose_transform.py` 加 `relative_pose_se2`/`absolute_pose_se2`/`rot2d`/`wrap_to_pi`;`utils.py` 正反两处接线
  - 新配置链:`configs/robot_configs/s1_fridge_head_qlocal.yaml` + `configs/vla/s1_fridge_qlocal/` + `assets/training_data/s1_fridge_head_qlocal.txt` + `assets/norm_stats/s1_fridge_head_qlocal.json`
  - 已推 fork:`e5d3392`(符号修复) `42cb4f0`(pnm 对齐) `e07436c`(merge infer 分支)
  - signfix 包:`baifu_cache/s1_fridge_head_signfix_step60000_inference.tar.gz` 21.56G(已被本次结论否决)
- **关键决策**:
  - 换参数化而非改数据 —— `s_q⁻¹⊗a_q` 对符号天然免疫
  - 底盘自写 SE(2) —— yht 三个 config 全无 base,无先例
  - 新建平行链不覆盖 —— linear/qrot 权重全留着可比
  - 平移选 local 而非 world —— 对里程计原点免疫
- **踩了什么坑**:
  - `enable_resume` 静默续训旧权重,25h 白跑
  - `quat_ref.json` 不进 ckpt,推理侧静默禁用对齐
  - `pgrep -f` 匹配自身命令行,连续误判 4-5 次
  - 我自己三次错误结论(clamp/wrap 噪声/qw 退化)都因省了验证
- **下次接手要看**:
  - 上机验证 qlocal:底盘左移到位后左手能否前伸开冰箱(唯一真判据)
  - 打包必须带 SE(2) 代码,否则底盘静默走偏(见「后续」节)
  - pnm(806ep)数据仍是我改过的符号修复版,若 qlocal 有效需照样建 qlocal 链

—— 以上五条决定 Claude 是否继续往下读 ——

## 上下文 / 起点

接手时:fridge 有 `linear`(基线,脏数据)和 `qrot`(`quaternion_world` 变体)两版权重,都是 8/31 `c84c6d1` 一起加的配置。用户报告**最后两次上机都卡在同一个具体位置**:底盘左移到位、左手该前伸开冰箱的那一刻冻住。

## 方法与决策

### 第一轮:诊断为四元数符号 bug,改数据(事后被否决)

`convert_s1_dataset.py:61` 逐帧独立调 `scipy Rotation.from_matrix`,Shepperd 法按「最大分量取正」定符号,两分量幅值打平时 argmax 换人 → 整体反号。state/action 各转一次,四条流互不知情。

证据:反号帧中 state/action 最大分量身份不同 **520/520(100%)**;最大与次大分量间距 反号帧 0.00136 vs 正常帧 0.10138(差 75 倍);反号帧 `|w|` 中位 0.70643 ≈ 1/√2。

后果:`q_target ≈ -q_current` 时 delta = `-2q`,L2 = 2.0(单位向量最大距离)。一个 chunk 50 步,一帧污染整段:帧级 0.54% → 静止 chunk **16.6%**。模型在视觉相同的静止画面上看到目标时而 0 时而 ±2.0,只能折中 —— 这解释了「出不来干净的零」。

修法选**全局 REF 半球对齐**重写 parquet:
- 不选训练时加 `relative_type`:只能修 delta 目标,修不了 state 作为模型输入的时间跳变(fridge 2129 次)和跨轨迹不一致
- 不选 `w>=0` 全局规范化:等价于「REF=零旋转」,而手腕离零旋转平均 88°,`|w|` 最小 0.00002 贴在 180° 分界线上,会在穿零处造新跳变
- REF 用 Markley 法(`M=Σq·qᵀ` 最大特征向量),对 ±q 天然免疫,不必先解决符号就能算。每任务每臂一个,余量 75-89°

实测:符号翻转 fridge 3474→0、pnm 4038→0;三项物理不变量全 `0.000e+00 deg`;静止 chunk 污染 16.6%→0.0%;norm span fridge `L_qw` 2.8276→0.2581(11x)、pnm 58.8x;平移 6 维全 1.0x 未动。重训 25.5h,末 100 步 VLA 0.0027。

### 第二轮:用户实测推翻,查 yht 找到真差异

用户反馈:**「上机效果最好的就是所谓脏数据下的数值直接相减」**。这否决了整条改数据路线 —— 我那 25h 版反而卡住。

查 `/kpfs-cognition/yht/lingbot-vla-v2`:`scripts/convert_astribot_s1_to_v30.py:74-125` 的 `matrix_to_quat_xyzw` 是 Shepperd 裸移植,末尾只有 `quat / norm`,**符号完全不处理**,和原做法一样。真正差异是 `astribot_s1_cap_pen.yaml:39` 的 `relative_type: quaternion_local`(我们是 `linear`)。

关键认识:`linear` 的旋转是 `a_q - s_q`,而 `q` 和 `-q` 是同一旋转,相减差 `2*a_q`。**那 0.54% 异常帧是 `linear` 这个参数化的产物,不是数据的毛病。** 5000 组随机四元数实测:翻转 action 侧符号后 `quaternion_local` delta 变化 `0.000e+00`(0/5000 受影响),`linear` 变化 1.999(2514/5000)。

### 第三轮:新建 qlocal 平行链

选新建 `s1_fridge_head_qlocal` 整条链,数据一字不动,原有 linear/qrot 全留:

| feature | linear(基线) | qlocal(本次) |
|---|---|---|
| `action.end.position` | `linear` | `quaternion_local` |
| `action.base.position` | `linear` | `se2_local` |
| `action.waist/head.position` | `linear` | `linear`(关节角,不是位姿) |

**底盘要自写 SE(2)**:`[x, y, theta]` 共 3 维,theta 是唯一 yaw;`quaternion_local` 按 `pose_dim=7` 切块(xyz+quat)套不上去。加 `relative_pose_se2` 走 `pose_dim=3`:平移用 `rot2d(d_xy, -theta)` 转进底盘朝向,角度直接相减 + `wrap_to_pi`(SO(2) 是交换群,不需四元数那套复合)。

选 `local` 而非 `world` 的依据:里程计原点+朝向偏移下 `se2_local` delta 变化 `1.4e-06`(免疫),`se2_world` 变化 0.143。真机每次开机 odom 原点不同,这项决定泛化。

**★底盘完全没有先例**:yht 三个 config `grep -c base` 全是 0(cap_pen 是固定站位)。SE(2) 这套是我按类比推的,只有数学验证(往返 1e-7)没有实测背书 —— 若这版上机仍有问题,底盘是首要怀疑对象而非末端。

### delta 的 base 是 state[t] 单帧广播

链路:`base_dataset.py:255` state 取 `chunk+1=51` 帧 → `utils.py:335` action=`state[1:]`(50帧) → `:339` state=`state[0]` collapse 成 1 帧 → 广播。

所以 `rel_q[k] = canonicalize(s_q[t]⁻¹ ⊗ a_q[t+k])`,k=0..49 **共用同一锚点**,不是链式。相对旋转随 k 累积:k=0 中位 0.18°,k=49 中位 6.32°/p99 58°,全 chunk max 102.9° → 距 `canonicalize` 的 180° 翻转边界还有 77° 余量,这份数据里 `w>=0` 规范化从不实际翻转任何帧。

## 产物

```
# 训练产物
/kpfs-cognition/baifu_cache/lingbot-vla-v2/output/s1_fridge_head_qlocal/
  checkpoints/global_step_{5000..60000}/   12 个, 各 73G
    hf_ckpt/      24G  ← 推理只要这个
    model/        24G
    optimizer/    26G
    extra_state/  72K
  lingbotvla_cli.yaml   ← deploy 从 ckpt 的 parent.parent.parent 读
  model_assets/         ← tokenizer 等

# 代码(已在本仓库, 未 commit)
lingbotvla/data/vla_data/ee_pose_transform.py   +relative_pose_se2/absolute_pose_se2/rot2d/wrap_to_pi
lingbotvla/data/vla_data/utils.py               正向 ~397 / 反向 ~515 加 elif _is_se2_relative_type

# 配置链
configs/robot_configs/s1_fridge_head_qlocal.yaml
configs/vla/s1_fridge_qlocal/s1_fridge_head_qlocal.yaml
assets/training_data/s1_fridge_head_qlocal.txt
assets/norm_stats/s1_fridge_head_qlocal.json
run_train.sh                                     +fridge_qlocal|fridge_head_qlocal variant

# 已推 fork (bananea20/lingbot-vla-v2)
e5d3392  符号修复(数据+stats+代码)
42cb4f0  pnm 806ep 对齐
e07436c  merge 用户的 infer 分支(astribot 异步协议)

# 被本次结论否决的产物(保留待清)
output/s1_fridge_head_signfix/                   874G, 12 ckpt
baifu_cache/s1_fridge_head_signfix_step60000_inference.tar.gz  21.56G
data/*/parquet 的符号修复版 + 备份 parquet_backup_fridge_20260903 / pnm_bak_20260903
```

训练日志:`logs/train_fridge_head_qlocal_20260908_171928.log`,wandb run `21rrwz3n`

VLA_Loss:5000=0.0116 → 15000=0.0043 → 之后在 0.003-0.008 震荡,60000=0.0028。单步瞬时值噪声大,35000 后**无可靠「最优 step」信号**,选 60000 是沿用惯例(signfix/qrot 都是 60000 便于对比),不是 loss 挑的。

### 可复制命令

```bash
# 起训(必须确认日志首行是 Step 1/..., Epoch 1)
bash run_train.sh fridge_head_qlocal

# norm stats 重算(★train.sh 用系统 python 会报 No module named lingbotvla)
.venv/bin/torchrun --nnodes=1 --nproc-per-node 8 --master-port=62711 \
  scripts/compute_norm_stats.py ./configs/vla/norm_compute/post_data.yaml \
  --data.robot_name s1_fridge_head_qlocal \
  --data.train_path assets/training_data/s1_fridge_head_qlocal.txt \
  --data.norm_path assets/norm_stats/s1_fridge_head_qlocal.json
# 8 卡约 1 分钟

# 覆盖 output_dir 绕开 resume(run_train.sh 末尾 "$@" 透传)
bash run_train.sh fridge_head --train.output_dir <新路径>
```

## 踩坑

**`enable_resume` 静默续训旧权重,数据修复等于没做**
起训后日志首行是 `Step 60001/7000, Epoch 9`,跑一步就退。原因:config `enable_resume: true`,而 output_dir 下已有 8/24 训的 `global_step_60000`,`max_steps` 也正好 60000 → 立刻达标退出,加载的是脏数据旧权重。绕法:命令行覆盖 `--train.output_dir` 到新目录。**判据:起训后必须确认日志是 `Step 1/...`, Epoch 1**,大于 1 就是在续训。

**`Step N/7000` 的分母是 epoch 内步数**
fridge 每 epoch 7000 步,`max_steps=60000`,约 8.6 epoch。判据:Epoch 2 起于 Step 7001、Epoch 3 起于 14001 → 分母 per-epoch,分子全局。所以 `Step 60001/7000` 的含义是「全局步 60001 已超 max_steps」。

**`quat_ref.json` 不会自动打进 ckpt,推理侧静默禁用对齐**
训练完的 ckpt 里没有它,bridge `--quat_ref auto` 三个探测位置全落空 → `load_quat_ref(None)` → 对齐静默关闭 → 模型在约一半帧上看到符号相反的 state,25h 重训收益作废且**无任何报错**。绕法:训完手动 `cp assets/quat_ref/X.json <ckpt>/configs/quat_ref.json`。根因:训练脚本不知道这个旁路常量的存在。(qlocal 不再需要此文件 —— 换参数化的直接收益)

**`pgrep -f` 匹配到自己的命令行**
打包早已结束但 `pgrep -f "gzip -1"` 仍返回真,连报三次「压缩中」。本会话踩了 4-5 次。原因:`bash -c "... pgrep -f X ..."` 里就含字符串 X。绕法:`ps -eo args --no-headers | grep "[g]zip -1"` 方括号技巧,或判断产物(文件字节数是否还在增长 / 写 `exit=` 标记文件)。`until ! pgrep -f X` 同理永不退出。

**后台 watcher 忘收会堆积**
`until/while ... sleep` 起的后台等待,任务完成后循环还活着,不停 spawn 新 sleep,累积到 17 个。绕法:用 Monitor 带 timeout,或任务结束即 TaskStop。

**tar 打包符号链接要 `-h`**
staging 用符号链接指向 24G 权重避免复制,不加 `-h` 包里只有几 KB 的链接文件。另:`hf_ckpt` 不含 norm_stats,必须单独拷进 `configs/`,否则部署起不来。gzip -1 对 safetensors 几无收益(24G → 21.56G),耗时约 9 分钟。

**NAS 读带宽是训练瓶颈,我的排查命令是诱因**
中途降速 1.5s/step → 3.7s/step,GPU 满频低温但利用率仅 ~45%。`tar tzf` 读 22G gzip、`du -sh` 扫 729G 目录,和 dataloader 抢同一条网络。**训练期间只读日志文件**。

**我连续三次错误结论,都因为省了验证**
1. 说 base theta 会被「放大 26000 倍到 ±4800」:错。`bounds_99` 带 `clamp(-1.5,1.5)`,真实后果是饱和成常数
2. 说 3.8e-5 是 `wrap_to_pi` 浮点噪声:错。噪声只有 1.16e-07,小 300 倍。改 round 式数学上更正确但没修掉该问题。真因是这任务底盘只平移不转,99.2% 帧 `action.theta==state.theta`,q01/q99 落在噪声量级 → 分位数退化。linear 和 se2_local 两版 theta clamp 后逐位相同(都 +1.5 常数),不影响对比
3. 说 qw 退化成常数、建议只预测 qxyz:错。当时只算了 k=0 一帧(转角 0.18°,qw 自然≈1)。真实 50 步 chunk 上 qw std 0.3514、归一化 RMS 0.94,和平移(0.32)/qxyz(0.34) 同量级

**判据:凡是「某维退化 / 某量被放大」的结论,必须在真实 chunk 上按训练管线的取帧方式算,不能只看单帧、不能省掉 clamp 和 q01 偏移。**

**节点表会过期**
`10.0.3.5` SSH 进去就是本机(`kaic-96081f8d-...` / `10.2.14.189`)。`10.0.3.237` 一分钟内从可连变成 sshd 挂掉(pod 重建),容器 IP 也从 `10.2.24.226` 变成 `10.2.27.97`。`.agent-brain` 里 19 个候选除本机外 18 个 22 端口全不通。→ 要跨机先重新发现,别信旧表。

## 后续 / 未完

**1. 上机验证 qlocal(唯一真判据)**
看底盘左移到位后左手能否前伸开冰箱。loss 不可跨版本比较(norm stats 量程不同)。

**2. 打包必须带 SE(2) 代码 —— 这次和以往不同**
`relative_pose_se2`/`absolute_pose_se2`/`_is_se2_relative_type` 只在本仓库,`astribot_eval-frank` 下**没有** `ee_pose_transform.py`,靠 `pip install lingbotvla`。旧版会走 `else` 分支退化成 `linear` 重构 —— **不报错,静默走偏**,双臂没事(`quaternion_local` 旧代码本就支持,框架默认值就是它),但底盘 x/y 少了 `quat_rotate` 那步会错。

包结构照 signfix 先例(24 项),两处调整:去掉 `quat_ref.json`、加上 `lingbotvla/data/vla_data/` 那两个改动文件。是否同时打 40000 + 60000 两版对冲过拟合,待定。

**3. pnm 数据仍是符号修复版**
806 ep 已被我改过(`42cb4f0`),对应四个模型都建立在改过的数据上。若 qlocal 有效,pnm 也该照样建 qlocal 链而非继续用改过的数据 + `linear`。

**4. 待清理(等 qlocal 验证结论)**
`output/s1_fridge_head_signfix/` 874G + 对应 21.56G 包。数据可从 `parquet_backup_fridge_20260903` / `pnm_bak_20260903` 还原成原始版(`quaternion_local` 对符号免疫,用改过的还是原始的结果逐位相同,但还原更干净且能弃用 `quat_ref.json`)。
