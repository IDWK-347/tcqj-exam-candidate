# EXPERIMENTS.md — `subsample` / `matching_conf_thr` 单变量对照

> 规则：
> 1. **一次只改一个参数**，用 `--set` 覆盖，不修改 YAML；
> 2. 每个数字都必须来自一次真实 run 的产物，并记录 run_id；
> 3. 损失值只在**同一目标函数**内比较；粗对齐与细化的 loss 不可横向比较；
> 4. 点数 ≠ 质量：观测行允许跨图重复，不是唯一物理点；
> 5. 没有对应 run 的结论不写。

---

## 0. 指标语义（写结论前必读）

| 指标 | 产物位置 | 含义与边界 |
|---|---|---|
| 端到端耗时 | `run_manifest.json` → `phases[].started_at_utc/ended_at_utc` | 含配置、资源校验（含两遍权重 SHA256）、图像加载、配对、模型加载、重建、导出 |
| 重建耗时 | `reconstruction.json` → `reconstruction_seconds_excluding_model_load` | 只覆盖 `sparse_global_alignment` + 设备同步；**不含**模型加载、预检哈希、CPU 提取、导出 |
| 峰值显存 | `reconstruction.json` → `peak_allocated_gib` | 模型加载后 reset 的 PyTorch allocator 峰值，**含驻留模型**，不等于整卡 VRAM |
| 点数 | `artifacts/artifacts.json` → `point_count`（`standard_result.json` 可交叉验证） | 观测行数，跨图可重复，不是唯一物理点 |
| 相机数 | 同上 → `camera_count` | 应等于场景图数 |
| 最终损失 | `reconstruction.json` → `final_losses` | 恰好 2 条（粗/细），只在同一阶段内可比较 |
| 配对规模 | `pair_graph.json` → `logical_edge_count` / `directed_item_count` | 有向对 = 2 × 无向边；不等于实际前向次数 |
| 几何质量 | `standard_result.json` → `geometry_quality_accepted` | **恒为 `false`**，不能作为质量通过的证据 |
| 尺度 | `cameras.json` → `scale_status` | **恒为 `unverified`** |

查看某个 run：

```bash
python3 tools/inspect_run.py outputs/<scene>/<run_id>
```

---

## 1. 实验设计

**基线**：scene_a（2 图、`complete`、1 条无向边 / 2 条有向对），
基线配置为 `subsample=8`、`matching_conf_thr=0.0`。

| ID | 唯一变量 | 命令 | 与基线的关系 |
|---|---|---|---|
| E0 | 无（基线） | `python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml` | 基线 |
| E1 | `backend.subsample` 8 → 16 | `python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --set backend.subsample=16` | 采样更稀 |
| E2 | `backend.subsample` 8 → 4 | `python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --set backend.subsample=4` | 采样更密 |
| E3 | `backend.matching_conf_thr` 0.0 → 5.0 | `python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --set backend.matching_conf_thr=5.0` | 低置信对改走 DUSt3R 分支 |
| E4 | `backend.matching_conf_thr` 0.0 → 1.5 | `python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --set backend.matching_conf_thr=1.5` | 轻度过滤 |

> E1/E2 与 E3/E4 各自独立；**不要在一次命令里同时改 subsample 和 thr**。
> 执行顺序建议：E0 → E1 → E2 → E3 → E4（先建立基线，再动一个维度）。

**场景选择说明**：参数实验放在 scene_a 上做，因为它是稳定、已通过的配置；
scene_b 的异常运行只用于诊断，不混入本表。

---

## 2. 对照表（已完成）

| ID | run_id | 完整生效配置 | 唯一变量 | 重建耗时 (s) | 端到端耗时 (s) | allocator 峰值 (GiB) | 点数 | 相机数 | final_losses | 可观察迹象 | 结论/解读 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | `20261126T042248Z_3d7558c62c88` | [`resolved_config.yaml`](outputs/scene_a/20261126T042248Z_3d7558c62c88/resolved_config.yaml) | 基线 | 4.467 | 18.52 | 3.168 | 14,734 | 2 | `[0.00931588001549244, 0.3066957890987396]` | 10 阶段 succeeded；1 边/2 有向 | 基线 |
| E1 | `20261126T045415Z_bc7519ec6124` | [`resolved_config.yaml`](outputs/scene_a/20261126T045415Z_bc7519ec6124/resolved_config.yaml) | `subsample=16` | 3.298 | 16.94 | 3.168 | 3,814 | 2 | `[0.009082280099391937, 0.31482994556427]` | 点数 0.259×；10 阶段 succeeded | 采样变稀，成本下降 |
| E2 | `20261126T045607Z_b7d39c4a2ae0` | [`resolved_config.yaml`](outputs/scene_a/20261126T045607Z_b7d39c4a2ae0/resolved_config.yaml) | `subsample=4` | 4.934 | 18.74 | 3.175 | 53,878 | 2 | `[0.008976246230304241, 0.2938293218612671]` | 点数 3.66×；10 阶段 succeeded | 采样变密，成本上升 |
| E3 | `20261126T045752Z_a8af27bf9b5f` | [`resolved_config.yaml`](outputs/scene_a/20261126T045752Z_a8af27bf9b5f/resolved_config.yaml) | `matching_conf_thr=5.0` | 3.684 | 17.45 | 3.168 | 14,734 | 2 | `[0.00931588001549244, 0.3066957890987396]` | 与基线逐位相同；阈值未触发 | 该数据上几乎不生效 |
| E4 | `20261126T045956Z_817a38ed404d` | [`resolved_config.yaml`](outputs/scene_a/20261126T045956Z_817a38ed404d/resolved_config.yaml) | `matching_conf_thr=1.5` | 3.639 | 17.30 | 3.168 | 14,734 | 2 | `[0.00931588001549244, 0.3066957890987396]` | 与基线逐位相同；阈值未触发 | 该数据上几乎不生效 |

**耗时口径**：
- **重建耗时** = `reconstruction.json` 的 `reconstruction_seconds_excluding_model_load`，只覆盖 `sparse_global_alignment` + 设备同步；
- **端到端耗时** = 各自 `run_manifest.json` 的 `ended_at_utc − started_at_utc`（run 级墙钟），单位 **秒**，四舍五入保留 **2 位小数**；
- 两者不可混用：端到端包含配置、资源校验（含两遍 2.5 GiB 权重 SHA256）、图像加载、配对、模型加载、重建、导出等。

**链接与取值核对**：上表 5 个 `resolved_config.yaml` 均已逐个打开核对——链接目标存在、
run_id 与目录一致；`subsample` 依次为 **8 / 16 / 4 / 8 / 8**，
`matching_conf_thr` 依次为 **0.0 / 0.0 / 0.0 / 5.0 / 1.5**，与“唯一变量”列一致。
其余 16 个字段（`schema_version`、`preprocess.long_edge/square_ok`、`pairing.strategy`、
`backend.name/weights_manifest/shared_intrinsics/coarse_iterations/refine_iterations/coarse_lr/refine_lr`、
`runtime.device/precision`、`scene.id/manifest`、`output.root`）在 5 个 run 中相同；
每轮的完整配置以链接文件为准。

**质量迹象**：除点数/损失外，另用对极一致性对同一批 run 做了只读复算，见 §4.2；
`matching_conf_thr` 在 scene_b 上真正生效时的表现见 §4.3。

**“一次只改一个”的产物证据**：5 个 run 的 `config_sources.json` 中，
E1/E2 各只有 `backend.subsample` 一项来源为 `<CLI>`，E3/E4 各只有
`backend.matching_conf_thr` 一项，基线与回归 run 无 CLI 覆盖。

---

## 3. 预测（执行前已写入 `PREDICTIONS.md` P3）

| ID | 预测要点 | 可证伪点 |
|---|---|---|
| E1 | anchor 网格降为 1/4；点数大致下降但非严格成比例；耗时/显存下降 | 若点数、耗时几乎不变 → 说明 `subsample` 未按预期传播 |
| E2 | anchor 网格约 4 倍；点数、耗时、显存上升，scene_a 上应仍可完成 | 若 OOM → 与 scene_b 的 H1 互证 |
| E3 | 低置信对从 3D 匹配损失转为 DUSt3R 回归损失，可能改变轨迹、耗时、显存 | 若所有对 `confs.max()` 远大于 5.0 → 参数几乎不生效，产物应几乎不变 |
| E4 | 介于 E3 与基线之间；若数据置信度整体很高，可能几乎无变化 | 同上 |

---

## 4. 结果分析（已完成）

> 本节只写能由第 2 节表格支持的内容；不写几何质量结论。

### 4.1 可观察的质量迹象（只描述，不下质量结论）

**`subsample` 的影响（scene_a，complete 1 边 / 2 有向不变，2 相机不变）**：

| subsample | 点数 | 每图观测数 | 重建耗时 (s) | allocator 峰值 (GiB) |
|---:|---:|---:|---:|---:|
| 16（E1） | 3,814 | 1,907 | 3.298 | 3.168 |
| 8（E0 基线） | 14,734 | 7,367 | 4.467 | 3.168 |
| 4（E2） | 53,878 | 26,939 | 4.934 | 3.175 |

- 点数随采样变密单调上升：16→8 约 3.86×，8→4 约 3.66×；
- 重建耗时上升，但远小于点数倍数（3.298→4.934 s），说明该规模下固定开销占主导；
- allocator 峰值几乎不变（3.168→3.175 GiB）。

**`matching_conf_thr` 的影响（scene_a）**：

| thr | 点数 | final_losses | 与基线比较 |
|---:|---:|---|---|
| 0.0（E0） | 14,734 | `[0.00931588001549244, 0.3066957890987396]` | 基线 |
| 1.5（E4） | 14,734 | 同上，逐位相同 | 无变化 |
| 5.0（E3） | 14,734 | 同上，逐位相同 | 无变化 |

- 两个阈值都没有改变点数或损失，说明 scene_a 这一对的 `confs.max()` 高于 5.0；
- 在 scene_b 上该参数会改变优化目标（`20261126T044458Z_d7fb4feff3c9`），
  但那是不同数据/不同对，不能把两边的损失横向比较。

**“变好了”的判断标准**：在配对拓扑、相机数、输出字段一致的前提下，
用点数与耗时的变化衡量成本；**不**用损失或点数声称几何质量变好。
上述实验只能证明参数按预期传播，以及成本如何变化。

### 4.2 几何一致性迹象（对已完成 run 的事后只读复算）

> 本节**不是新的重建实验**：5 个 run 都是 §2 表里已经跑过的，这里只是用
> `tools/measure_epipolar.py` 对它们的产物做事后只读复算，没有新增 run、没有改配置。
> 指标定义与坐标口径见 `DIAGNOSIS.md` §2.4（固定 SIFT/RANSAC 内点、像素
> `F = K₂⁻ᵀ E K₁⁻¹`、平方 Sampson 距离开方）。scene_a 只有一对图（01–02），
> 因此只有一个样本。

复算命令：

```bash
python3 tools/measure_epipolar.py --scene scene_a \
  --run outputs/scene_a/<run_id> --pairs 01-02
```

| ID | run_id | 变量 | matches | F 内点 | Sampson 中位 (px) | p90 (px) | control 中位 (px) |
|---|---|---|---:|---:|---:|---:|---:|
| E0 | `20261126T042248Z_3d7558c62c88` | 基线 `subsample=8` | 345 | 317 | **0.046** | 0.134 | 0.052 |
| E1 | `20261126T045415Z_bc7519ec6124` | `subsample=16` | 345 | 317 | **0.061** | 0.132 | 0.052 |
| E2 | `20261126T045607Z_b7d39c4a2ae0` | `subsample=4` | 345 | 317 | **0.049** | 0.136 | 0.052 |
| E3 | `20261126T045752Z_a8af27bf9b5f` | `matching_conf_thr=5.0` | 345 | 317 | **0.046** | 0.134 | 0.052 |
| E4 | `20261126T045956Z_817a38ed404d` | `matching_conf_thr=1.5` | 345 | 317 | **0.046** | 0.134 | 0.052 |

- 五个 run 的中位误差都在 0.05–0.06 px 量级，与该对自身的 control 中位（0.052 px）相当；
- `subsample` 16 / 8 / 4 对应 0.061 / 0.046 / 0.049 px，变化 0.003–0.015 px，
  **没有随采样变密而单调改善**；该差异与坐标口径的近似偏差同阶
  （`DIAGNOSIS.md` §2.4：12 组复算的最大变化为 0.015 px），
  因此不能据此判断哪种采样“质量更好”；
- E3/E4 的产物与基线逐位相同，指标也逐位相同，与 §4.1 一致；
- **局限**：scene_a 只有一对图、只有一个样本；该指标只衡量**位姿—图像自洽性**，
  不衡量绝对几何精度（无真值），也不能外推到 scene_b/scene_c 的拓扑。

### 4.3 `matching_conf_thr` 在 scene_b 上的实际影响（跨场景补充）

scene_a 上两个阈值都没有触发（§4.1），因此只看本表会得出“该参数无效”的印象。
在 scene_b 上它确实切换了损失分支（`DIAGNOSIS.md` §3 H3）。下面两个 run 是同一场景、
同一份数据、只改 `matching_conf_thr` 的单变量对照，**不与 §2 的 scene_a 表横向比较**：

| 项目 | B0 `20261126T042744Z_ffdbbf471afb`（thr=0.0） | `20261126T044458Z_d7fb4feff3c9`（thr=5.0） |
|---|---|---|
| 点数 | 354,140 | 354,140（不变） |
| 重建耗时 | 24.702 s | 25.406 s |
| allocator 峰值 | 3.259 GiB | 3.259 GiB |
| final_losses（目标函数已变，不可横向比较） | 0.022824 / 0.885533 | 0.000431 / 0.000252 |
| 对极中位 01–02 | 0.254 px | 1.780 px |
| 对极中位 04–05 | 0.257 px | 2.250 px |
| 对极中位 01–07 | 2.166 px | 8.642 px |

- 点数和显存不变，损失与对极指标都明显变化 → 参数确实生效；
- 三个受测对的对极误差全面变差；这是**该数据上**的观察，不是“阈值越大越差”的普遍结论；
- 损失骤降是大量对转入 DUSt3R 分支、目标函数改变的结果，**损失小不等于质量好**。

### 4.4 不能由本组实验证明的

- 几何精度、尺度、与真值的一致性；
- “损失更小”不等于“质量更好”；
- `status: succeeded` 不等于几何质量通过（`geometry_quality_accepted` 恒为 `false`）。

---

## 5. 失败或无效实验记录

| run_id | 尝试的变量 | 失败阶段/原因 | 处置 |
|---|---|---|---|
| `20261126T045752Z_a8af27bf9b5f` | `matching_conf_thr=5.0` | 运行成功，但参数未触发（产物与基线逐位相同） | 记录为“该数据上不生效”，不作为质量结论 |
| `20261126T045956Z_817a38ed404d` | `matching_conf_thr=1.5` | 同上 | 同上 |
| `20261126T041729Z_4b2acf11e162`、`20261126T041937Z_581439180005`、`20261126T042110Z_7904ef813fba` | 无（环境问题） | `backend_reconstruct` / `CUBLAS_STATUS_NOT_INITIALIZED`（`LD_LIBRARY_PATH` 混用 CUDA 12.2） | 清空 `LD_LIBRARY_PATH` 后重跑成功；见 `ENVIRONMENT.md` §4.5 |
