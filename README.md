# 多视角三维重建工程｜技术岗考核

**时长两天。允许使用任何 AI 工具。**

---

## 考核目标

在给定的三维重建工程上，完成环境配置、基线建立、输出诊断、参数实验与场景迁移，
并用可复现的证据说明每一步的判断依据。

## 关键资源

- 工程：当前工作区，入口为 `scripts/run_reconstruction.py`
- 权重：`checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`（已预置，不要联网下载）
- 数据：`data/scene_a/`（2 图）、`data/scene_b/`（7 图）、`data/scene_c/`（3 图）
- 场景配置：`configs/scenes/scene_a.yaml`、`scene_b.yaml`、`scene_c.yaml`
- 公共参数：`configs/base.yaml`

## 考核环境

- 机器：Tesla T4 16GB，CUDA 可用
- Python：3.11（系统默认 `python3` 即 3.11.1）
- 依赖：见 `requirements.txt`，可用 venv 或 conda 管理
- 入口：`python scripts/run_reconstruction.py --config configs/scenes/<场景>.yaml`

---

## 考核任务详情

### 1. 环境配置

目标：让入口命令在 `scene_a` 上成功执行。

```bash
python scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --dry-run
```

验证：命令正常返回；去掉 `--dry-run` 再执行一次，产出 `artifacts/scene.ply`。

交付 `ENVIRONMENT.md`：环境配置记录（如遇到问题请写出解决步骤）

### 2. 建立基线

把 `scene_a` 完整跑一次，读取本次运行的输出目录，在 `BASELINE.md` 中填写：

| 项目 | 来源 |
|---|---|
| run ID | 运行目录名 / `run_manifest.json` |
| 点数、相机数 | `artifacts/artifacts.json` 或 `standard_result.json` |
| 端到端耗时 | `run_manifest.json` 的 phases |
| 峰值显存 | `reconstruction.json` |
| 本次实际生效的配置 | `resolved_config.yaml` |
| 每个配置字段的来源 | `config_sources.json` |
| 配对边的条数 | `pair_graph.json` |

并回答：

- `point_sources.npz` 中 `source_observation_index` 是什么？是否为像素坐标？
- `cameras.json` 中 `scale_status` 是什么值？这一取值意味着什么？
- `standard_result.json` 中 `availability.point_confidence` 是什么状态？为什么？

### 3. 诊断 scene_b

跑通 `scene_b`，对其输出结果做诊断。

1. 在 `DIAGNOSIS.md` 中列出至少 3 个可能原因，每条写成「如果是 H，那么应该观察到 E」
2. 为每条假设设计一个能区分它的实验，并实际执行
3. 执行前在 `PREDICTIONS.md` 写下预期
4. 执行后对账，写明「与预期一致」或「不一致，原因是……」
5. 给出结论：根因在哪个文件哪一行、如何修复、用什么证明已修复
6. 回归：修复后 `scene_a` 是否仍与基线一致

### 4. 策略与成本

`configs/base.yaml` 中有两个参数：`backend.subsample` 与 `backend.matching_conf_thr`。

1. 分别测量它们的影响，**一次只改一个**
2. 每次运行记录：耗时、峰值显存、点数、可从产物观察到的质量迹象
3. 说明你用什么判断「变好了」，以及这一判断能证明什么、不能证明什么
4. 完整记录每一轮的实际配置

交付 `EXPERIMENTS.md`。

### 5. 迁移与交付

用最终方案跑 `scene_c`（3 图），**不要为它单独调参**。

交付 `SUBMISSION.md`：

- 从干净环境开始的完整命令序列
- 你改动了什么、为什么
- 方案在什么条件下会失效（写具体条件）

---

## 交付成果

在分支 `submission/<你的名字>` 上提交：

```
ENVIRONMENT.md    环境记录
BASELINE.md       scene_a 的基线观测记录
PREDICTIONS.md    每次关键实验前的预测（可边做边追加）
DIAGNOSIS.md      scene_b 的诊断过程与结论
EXPERIMENTS.md    subsample / matching_conf_thr 的对照表
SUBMISSION.md     复现步骤、改动说明、方案边界
```

另附：改动过的代码（或「未改代码」的结论）、`scene_c` 的产物。

结论须与对应的 run 相符；没有对应 run 的结论不计分。

## 规则

**允许**

- 使用任何 AI 对话工具
- 查阅公开文档
- 重建虚拟环境、修改 requirements、修改代码、重启环境

**不允许**

- 修改 `data/` 下的图片或 manifest 中声明的 sha256
- 修改 `checkpoints/` 中的权重
- 删除 `scene_b` 的图片使其「跑通」
- 在报告中写入未实际运行得到的数字

