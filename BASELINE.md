# BASELINE.md — scene_a 基线观测

> 本文件所有数字均来自 run `20261126T042248Z_3d7558c62c88` 的真实产物。
> 未采用 `outputs/` 中来自其他工程的历史 run（见文末）。

---

## 0. 基线的定义

- 场景：`configs/scenes/scene_a.yaml`（2 图、`pairing.strategy: complete`）
- 参数：base 配置（`subsample=8`、`coarse_iterations=100`、`refine_iterations=100`、
  `coarse_lr=0.07`、`refine_lr=0.01`、`matching_conf_thr=0.0`、`shared_intrinsics=false`）
- 运行命令（必须清空 `LD_LIBRARY_PATH`，原因见 `ENVIRONMENT.md` §4.5）：

```bash
cd /workspace
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml
```

- 产物只读复核（不重跑重建）：

```bash
python3 tools/inspect_run.py outputs/scene_a/20261126T042248Z_3d7558c62c88
```

---

## 1. 运行标识

| 项目 | 值 | 来源 |
|---|---|---|
| run_id | `20261126T042248Z_3d7558c62c88` | 输出目录名 / `run_manifest.json` → `run_id` |
| 开始时间（UTC） | `2026-11-26T04:22:48.475657+00:00` | `run_manifest.json` → `started_at_utc` |
| 结束时间（UTC） | `2026-11-26T04:23:06.971832+00:00` | `run_manifest.json` → `ended_at_utc` |
| 最终状态 | `succeeded` | `run_status.json` / `run_manifest.json` |
| 进程退出码 | 未记录（产物中无退出码字段；完成与否以 `run_status.status` 为准） | —— |
| 权重 SHA256 | `e28f91b4…568eb2` | `preflight.json` / `backend_fingerprint.json` |

---

## 2. 阶段耗时

| 阶段 | 开始（UTC） | 结束（UTC） | 耗时 (s) |
|---|---|---|---|
| configuration | 04:22:48.475657 | 04:22:48.517966 | 0.04 |
| asset_validation | 04:22:48.530363 | 04:22:50.633833 | 2.10 |
| image_loading | 04:22:50.646462 | 04:22:52.098094 | 1.45 |
| pairing | 04:22:52.452971 | 04:22:52.488379 | 0.04 |
| backend_prepare | 04:22:52.500815 | 04:23:02.135632 | 9.63 |
| backend_reconstruct | 04:23:02.148300 | 04:23:06.636692 | 4.49 |
| backend_close | 04:23:06.649645 | 04:23:06.802149 | 0.15 |
| normalize | 04:23:06.814727 | 04:23:06.834505 | 0.02 |
| export | 04:23:06.847069 | 04:23:06.931608 | 0.08 |
| report | 04:23:06.944142 | 04:23:06.971832 | 0.03 |
| **阶段合计** | — | — | **18.04**（= 按上表原始起止时间戳逐项求差后求和，取两位小数） |

墙钟总时长（`ended_at_utc - started_at_utc`）：**18.52 s**；两者差值来自阶段之间的
证据落盘与切换开销。

> `asset_validation` 与 `backend_prepare` 各包含一次 2.5 GiB 权重的 SHA256 计算（纯 IO），
> 与推理无关。`backend_prepare` 的 9.63 s 还包含模型加载与设备搬运。
> 阶段之间的缺口是 manifest 落盘/切换开销。

---

## 3. 关键产物值

| 指标 | 值 | 来源 |
|---|---|---|
| 相机数 `camera_count` | **2** | `artifacts/artifacts.json` / `standard_result.json` |
| 点数 `point_count` | **14,734** | 同上 |
| 每图观测数 | `01.png` 7,367；`02.png` 7,367 | `reconstruction.json` → `per_image_observation_counts` |
| 配对数（无向） | **1** | `pair_graph.json` → `logical_edge_count` |
| 有向对 | **2** | 同上 → `directed_item_count` |
| 配对图连通分量 | `[['01.png', '02.png']]` | 同上 |
| 图哈希 | `67284398e1288bf4c8bddef4f489a3ad16c12c434850ff3d38c10c5681c2227f` | 同上 |
| 最终损失（2 条） | `0.00931588001549244`（粗）、`0.3066957890987396`（细） | `reconstruction.json` → `final_losses` |
| 重建耗时 (s) | **4.467** | `reconstruction.json` → `reconstruction_seconds_excluding_model_load` |
| allocator 峰值 (GiB) | **3.168** | `reconstruction.json` → `peak_allocated_gib` |
| 本次实际生效的配置 | 18 个字段；关键值 `subsample=8`、`matching_conf_thr=0.0`、`coarse/refine_iterations=100`。完整字段见 §3.1 | [`resolved_config.yaml`](outputs/scene_a/20261126T042248Z_3d7558c62c88/resolved_config.yaml) |
| 每个配置字段的来源 | 18/18 字段可追溯：`configs/base.yaml` 12 项、`configs/t4.yaml` 2 项、`configs/scenes/scene_a.yaml` 4 项；**无 CLI 覆盖** | [`config_sources.json`](outputs/scene_a/20261126T042248Z_3d7558c62c88/config_sources.json) |
| 处理后图像尺寸 | `(384, 512)` × 2 | `standard_result.json` → `processed_hw` |
| 参考相机 | `01.png` | `cameras.json` → `reference_image_id` |
| 产物 | `artifacts/scene.ply` 398,032 B；`cameras.json` 3,887 B；`point_sources.npz` 62,019 B | `artifacts/artifacts.json` |
| 导出读回 | `readback_checks` 五项均为 `true`：`point_count_and_coordinates`、`rgb_row_correspondence`、`coordinate_range`、`camera_ids_matrices_and_conventions`、`source_rows_masks_and_optional_fields` | `artifacts/artifacts.json` |

**量级 sanity check**：与历史外来 run（2 图、16,328 点、alignment 5.32 s、峰值 3.17 GiB）
同一量级，但本文件的数字全部来自本次 run，未采用历史 run 的任何值。

**回归验证（2026-11-26T04:50Z）**：修改 `configs/scenes/scene_b.yaml` 的
`subsample` 后，用同一命令重跑 scene_a，run `20261126T045015Z_65797318eef8`
与本基线逐位一致：2 相机 / 14,734 点 / `final_losses` 相同 / 峰值 3.168 GiB。
重建耗时 3.715 s（基线 4.467 s）属于计时波动。

### 3.1 实际生效配置与字段来源（逐字段核对）

字段全集取自本 run 的
[`resolved_config.yaml`](outputs/scene_a/20261126T042248Z_3d7558c62c88/resolved_config.yaml)，
来源逐项取自
[`config_sources.json`](outputs/scene_a/20261126T042248Z_3d7558c62c88/config_sources.json)
的 `fields.<字段路径>`。“最终生效来源”是 `load_order`
（`configs/base.yaml` → `configs/t4.yaml` → `configs/scenes/scene_a.yaml`）
合并后最后生效的那一项。

| 字段路径 | 生效值 | 最终生效来源 |
|---|---|---|
| `schema_version` | `1` | `configs/scenes/scene_a.yaml`（base/t4/scene_a 都声明了 1，最后声明者生效） |
| `preprocess.long_edge` | `512` | `configs/base.yaml` |
| `preprocess.square_ok` | `false` | `configs/base.yaml` |
| `pairing.strategy` | `complete` | `configs/base.yaml` |
| `backend.name` | `mast3r_sparse_ga` | `configs/base.yaml` |
| `backend.weights_manifest` | `/workspace/assets/weights.json` | `configs/base.yaml`（原值 `../assets/weights.json`，相对 base 解析后为绝对路径） |
| `backend.shared_intrinsics` | `false` | `configs/base.yaml` |
| `backend.subsample` | `8` | `configs/base.yaml` |
| `backend.coarse_iterations` | `100` | `configs/base.yaml` |
| `backend.refine_iterations` | `100` | `configs/base.yaml` |
| `backend.coarse_lr` | `0.07` | `configs/base.yaml` |
| `backend.refine_lr` | `0.01` | `configs/base.yaml` |
| `backend.matching_conf_thr` | `0.0` | `configs/base.yaml` |
| `runtime.device` | `cuda` | `configs/t4.yaml` |
| `runtime.precision` | `fp32` | `configs/t4.yaml` |
| `scene.id` | `scene_a` | `configs/scenes/scene_a.yaml` |
| `scene.manifest` | `/workspace/data/scene_a/manifest.json` | `configs/scenes/scene_a.yaml`（原值 `../../data/scene_a/manifest.json`，相对声明文件解析） |
| `output.root` | `/workspace/outputs` | `configs/scenes/scene_a.yaml`（原值 `../../outputs`，相对声明文件解析） |

**字段集合核对**：`resolved_config.yaml` 扁平化后为 **18** 个字段，
`config_sources.json` 的 `fields` 也是 **18** 个，两者差集为空——
没有“生效但无来源记录”的字段，也没有“有来源但未生效”的字段。
**无法追溯的字段：无（18/18 均可追溯）**。

**为什么没有 `pairing.window`**：该字段只允许在 `pairing.strategy: sequential_window`
时出现（`config.py` 的白名单约束）；本 run 用 `complete`，因此它本来就不在字段全集内，
不是遗漏。

---

## 4. 三个必答题（用本次 run 的产物复核）

### 4.1 `source_observation_index` 是什么？

- 代码依据：`results.py:55` ——
  `zero-based row in each original sparse view; not a pixel coordinate`。
- 本次 run 的实测：
  - `artifacts/point_sources.npz` 中有
    `source_observation_index (14734,) int64`、
    `source_image_index (14734,) int64`、
    `source_pixels_xy (14734, 2) float64`、
    `view_mask_000 (7367,) bool`、`view_mask_001 (7367,) bool`；
  - `source_observation_index` 的取值范围是 **0–7366**（每张原稀疏视图内部的行号），
    不是 512×384 的像素坐标；
  - 像素坐标在 `source_pixels_xy`：14734 行 × 2 列，x 最大 511、y 最大 369，
    均落在处理后 `512×384` 范围内。
  - 前 5 行示例：`index = [0,1,2,3,4]`，`xy = [[0,1],[1,1],[4,1],[8,1],[10,1]]`。

**结论**：`source_observation_index` 是“原始稀疏视图中的零基行号”；像素坐标必须读
`source_pixels_xy`。两者不是一回事。

### 4.2 `scale_status` 是什么？为什么？

- 代码依据：`exporter.py:200`、`results.py:45` 写死 `"unverified"`；
  `sparse_io.py:135` 写明 `scale: unresolved, metric_units_verified: False`。
- 本次 run 的实测：`cameras.json` 与 `standard_result.json` 中均为
  `scale_status: "unverified"`。
- 含义：重建处于上游优化器的任意尺度世界，**没有度量尺度**，不能当作真实尺寸使用。

### 4.3 `availability.point_confidence` 为什么不可用？

- 代码依据：后端不读取上游私有缓存中的置信度（`sparse_io.py:123`；
  `mast3r.py` 的 capabilities 中 `point_confidence_supported: False`）。
- 本次 run 的实测：

```json
"point_confidence": {
  "status": "unavailable",
  "reason": "No validated per-observation reconstruction confidence was supplied"
}
```

- 因此 `filtering.confidence_strategy` 为 `"disabled"`、`confidence_threshold` 为 `null`，
  且 `retained_count = input_count = 14734`（没有因置信度过滤掉点）。

---

## 5. 本次 run **不能**证明的事

- 几何质量：`standard_result.json` 的 `geometry_quality_accepted` 为 `false`；
- 尺度：`scale_status` 为 `unverified`；
- 与真值的一致性：本场景没有标定几何真值；
- “10 个阶段全部 succeeded”只证明阶段与清理流程完成，不证明几何质量。

---

## 6. 运行条件（重要）

本次基线必须带 `LD_LIBRARY_PATH=` 前缀运行。默认 shell 的 `LD_LIBRARY_PATH` 含有
`/usr/local/cuda-12.2/lib64`，会让 pip 的 cuBLAS 12.8 加载 12.2 的 `libcublasLt`，
导致匹配阶段 `A @ B.T` 报 `CUBLAS_STATUS_NOT_INITIALIZED`。详细证据与修复见
`ENVIRONMENT.md` §4.5。

---

## 7. 未采用的历史材料

`outputs/scene_a/20261010T205345Z_230967b3a998/` 来自 `/workspace/tcqj-reconlab-golden`：

- 输入图片 sha256 与 `data/scene_a/` 不同；
- `resolved_config.yaml` 指向旧工程路径；
- 只能作为“量级参考”，**没有**填进本文件。

`docs/BASELINE.md` 是历史 `scene_01` 七图记录，也不是本次模板。
