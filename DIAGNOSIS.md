# DIAGNOSIS.md — scene_b 输出诊断

> 规则：
> 1. 结论必须指回一次**真实 run**（run_id + 产物）；
> 2. 静态检查结论单独标注，不复用为 run 结论；
> 3. 每条假设写成「**如果是 H，那么应观察到 E**」，并为每条设计**只改一个变量**的区分实验；
> 4. 执行前在 `PREDICTIONS.md` 写预测，执行后对账；
> 5. 未执行的实验写“未执行”，不填任何推测数字；
> 6. 不删除、不替换 `data/scene_b/` 的任何图片来“跑通”。

---

## 0. 诊断流程

1. **B0**：原样运行 `configs/scenes/scene_b.yaml`，取得完整输出；
2. 先判断它是“崩溃”还是“跑通但输出有问题”；
3. 对输出做一致性检查（每图点数、索引、掩码、位姿、对极误差）；
4. 按检查结果提出假设 → 写预测 → 执行只改一个变量的区分实验；
5. 根因确认后修复，回归 scene_a，再验证 scene_b。

> 重要：README 的原文是“**跑通 scene_b，对其输出结果做诊断**”，
> 不是“假设 scene_b 一定崩溃”。早期按崩溃设计的假设（OOM、空匹配 `max()`、NaN、
> 身份/顺序不符）见 `PREDICTIONS.md` P2；它们在 B0 中均未触发，保留为历史预测。

---

## 1. B0 真实输出（已完成）

| 项目 | 值 | 来源 |
|---|---|---|
| B0 run_id | `20261126T042744Z_ffdbbf471afb` | 输出目录 / `run_manifest.json` |
| 状态 | `succeeded`，`error: null` | `run_status.json` |
| 阶段 | 10 个阶段全部 `succeeded` | `run_manifest.json` |
| 总耗时 | 阶段合计约 39.7 s | `run_manifest.json` |
| 相机数 | **7** | `artifacts/artifacts.json` / `standard_result.json` |
| 点数 | **354,140**（观测行，不是唯一物理点） | 同上 |
| 每图观测数 | 01: 33,272；02: 65,992；03: 59,716；04: 54,294；05: 54,069；06: 56,784；07: 30,013 | `reconstruction.json` |
| 无向边 / 有向对 | **6 / 12**（explicit 链） | `pair_graph.json` |
| 连通分量 | 7 个节点，单一连通分量 | 同上 |
| 最终损失 | `0.022823568433523178`（粗）、`0.8855334520339966`（细） | `reconstruction.json` |
| 重建耗时 | 24.702 s | 同上 |
| allocator 峰值 | 3.259 GiB | 同上 |
| `scale_status` | `unverified` | `cameras.json` / `standard_result.json` |
| `point_confidence` | `unavailable` | `standard_result.json` |
| `source_pixels_xy` | `available` | 同上 |
| `geometry_quality_accepted` | `false` | 同上 |

关键判断：**B0 不是崩溃，也不是“0 点/缺相机”的假成功**。因此诊断对象是输出质量/一致性。

### 1.1 证据分类：流程 vs 几何质量

| 证据 | 能证明 | 不能证明 |
|---|---|---|
| 10 个阶段全部 `succeeded`、`error: null` | 声明的阶段按序执行、资源清理完成 | 几何质量 |
| `camera_count=7`、每图 30,013–65,992 个观测 | 所有视图都参与了重建，没有空视图 | 这些点是否落在正确的三维位置 |
| `source_observation_index` 为 0…n−1、掩码与 `source_image_index` 一致 | 观测行与源视图的对应关系正确 | 几何精度 |
| `removed_nonfinite=0`、两条 `final_loss` 有限 | 没有数值发散 | 几何精度 |
| `T_wc @ T_cw ≈ I`、主点 `(256,192)`、`processed_hw=(384,512)` | 导出的位姿/内参自洽且单位正确 | 位姿是否与真实相机一致 |
| `artifacts/artifacts.json` 的 `readback_checks` 五项均为 `true` | 导出文件可读回、内容自洽 | 几何质量 |
| **对极一致性（§2.4）** | 重建位姿与图像对极几何一致 | 绝对几何精度、尺度 |
| `scale_status=unverified`、`geometry_quality_accepted=false` | —— | 明确表示没有尺度与质量验收 |

结论：能证明“流程跑通”的证据很多；能评价“几何质量”的只有对极一致性，
而它只衡量**位姿与图像的自洽性**，不是与真值的偏差。

---

## 2. 输出一致性检查（已执行）

### 2.1 数据与预处理（静态，B0 之前已做）

- `data/` 12 张图 sha256 与 manifest 一致（12/12 OK）；
- 7 张图全部 `(384, 512)`，顺序 01…07，无 warning（`image_preprocessing.json` 的
  `records` 逐条可查）；
- 配对图：7 节点 / 6 条 explicit 边 / 12 有向对 / 单连通分量；
- vendor 固定哈希一致（`backend_fingerprint.json`）；
- SIFT 两两匹配（ratio 0.75、BFMatcher k=2）：21 个无序对的 good matches 为
  **648–805**，无异常低值。复现命令（`matches` 列即 good matches 数；
  对极指标不是本节主张，见 §2.4）：

```bash
python3 tools/measure_epipolar.py --scene scene_b \
  --run outputs/scene_b/20261126T042744Z_ffdbbf471afb \
  --pairs 01-02,01-03,01-04,01-05,01-06,01-07,02-03,02-04,02-05,02-06,02-07,03-04,03-05,03-06,03-07,04-05,04-06,04-07,05-06,05-07,06-07
```

### 2.2 观测索引、掩码与像素范围（用 B0 产物）

`artifacts/point_sources.npz`：

| 图 | 观测数 | `source_observation_index` 范围 | 唯一值数 | 掩码与索引一致 |
|---|---:|---|---:|---|
| 01.png | 33,272 | 0–33,271 | 33,272 | 是 |
| 02.png | 65,992 | 0–65,991 | 65,992 | 是 |
| 03.png | 59,716 | 0–59,715 | 59,716 | 是 |
| 04.png | 54,294 | 0–54,293 | 54,294 | 是 |
| 05.png | 54,069 | 0–54,068 | 54,069 | 是 |
| 06.png | 56,784 | 0–56,783 | 56,784 | 是 |
| 07.png | 30,013 | 0–30,012 | 30,013 | 是 |

- `source_pixels_xy` 全部为整数值，范围 x∈[0,511]、y∈[0,383]，落在处理后图像内；
- `filtering.removed_nonfinite = 0`，`retained_count = input_count = 354,140`；
- 未发现 NaN/Inf、缺图、0 点视图或索引越界。

### 2.3 相机位姿与焦距（用 B0 产物）

| 指标 | 值 |
|---|---|
| 相机中心两两最大距离 | 0.218（场景点云 z 约 1.8，尺度任意） |
| 相对 01 的旋转角 | 0°–5.07° |
| 焦距范围 | 566.73–611.70，相对跨度 **7.62%** |
| 位姿/内参有限性 | 全部有限 |
| `T_world_from_camera @ T_camera_from_world` | `≈ I`（最大绝对误差 **2.22e-16**，双精度舍入量级） |
| 内参主点 | `(256, 192)`，对应处理后 512×384 |
| `processed_hw` | 7 张图全部 `(384, 512)` |

单看这些数值不能判定对错（没有几何真值），需要与图像的对极几何交叉验证。

### 2.4 对极一致性（SIFT + 固定 RANSAC 内点 + 重建位姿）

**指标定义**（可复算脚本：`tools/measure_epipolar.py`）：

- 每对图像只做一次 SIFT（ratio 0.75），并用固定种子 `RANSAC(1.5 px)` 得到
  **固定内点集**；所有 run 共用这组对应点，避免匹配集不同导致不可比；
- 位姿方向由 `cameras.json` 实际核对：
  `T_world_from_camera @ T_camera_from_world ≈ I`（B0 最大绝对误差 2.22e-16），
  主点为处理后图像中心；
- 相对位姿：`X_cam2 = R₂ᵀ R₁ X_cam1 + R₂ᵀ(t₁ − t₂)`，即
  `R = R₂ᵀR₁`、`t = R₂ᵀ(t₁−t₂)`；
- 像素基础矩阵：`F = K₂⁻ᵀ [t]× R K₁⁻¹`；
- 统计量：**平方 Sampson 距离开方后的值**，像素单位；
- `control_median_px` 是直接用该对内点估出的 `F` 的 Sampson 中位，作为该数据对的噪声下限参考。

**坐标口径（当前脚本的近似，必须写明）**：脚本在**原始 1448×1086 图像**上提取 SIFT
（`measure_epipolar.py:69-74`），再按**纯比例** `(512/w, 384/h)` 映射到处理后坐标
（`:75-76`）。而工程预处理实际用的是含半像素平移的仿射变换
（`image_preprocessing.json` → `A_oriented_to_processed`，B0 七张图均为
`sx = sy = 0.353591`、`tx = ty = −0.323204`）。默认模式**省略了这个亚像素平移**，
因此“单位为处理后 512×384 像素”只到亚像素精度。加 `--preprocess-affine`
（读取每图的 `A_oriented_to_processed`）可复算补上平移后的值：

```bash
python3 tools/measure_epipolar.py --scene scene_b \
  --run outputs/scene_b/20261126T042744Z_ffdbbf471afb \
  --pairs 01-02,04-05,01-07 --preprocess-affine
```

实测：B0 的 01–07 由 2.166 变为 **2.164 px**；在 4 个 scene_b run × 3 个图像对共 12 组里，
最大变化为 **0.015 px**（thr=5.0 的 01–07 由 8.642 变为 8.657），其余均 ≤0.003 px，
**不改变任何比较与结论**。下文表格与结论一律引用默认口径（纯比例映射）的可复现值。
另需注意：SIFT 跑在原始分辨率而非预处理图上。复核记录中还有一组“换到 512×384 图上重算”
的数值，但原始配方未知、本轮无法逐位复现，也没有保留可复现产物，因此**本报告不引用该组数值**，
只写当前脚本可复现的值。

复算命令：

```bash
python3 tools/measure_epipolar.py --scene scene_b \
  --run outputs/scene_b/20261126T042744Z_ffdbbf471afb \
  --pairs 01-02,04-05,01-07
```

| 视图对 | SIFT matches | F 内点 | 中位 (px) | p90 (px) | control 中位 (px) |
|---|---:|---:|---:|---:|---:|
| 01–02 | 779 | 665 | **0.254** | 0.646 | 0.288 |
| 04–05 | 790 | 592 | **0.257** | 1.687 | 0.179 |
| 01–07 | 669 | 491 | **2.166** | 4.188 | 0.420 |

**三个历史数值的来源（逐项说明，只把可复现的那个用于结论）**：

| 数值 | 来源与状态 | 复现方式 | 是否用于本文结论 |
|---|---|---|---|
| **2.166 px** | **当前指标**（像素 `F = K₂⁻ᵀ E K₁⁻¹`、正确位姿方向、固定 SIFT/RANSAC 内点，669 matches / 491 inliers） | `python3 tools/measure_epipolar.py --scene scene_b --run outputs/scene_b/20261126T042744Z_ffdbbf471afb --pairs 01-07` | **是**，唯一引用的 B0 01–07 数值 |
| **36.970 px** | **旧算法结果**（已被取代，不代表任何几何结论） | 加 `--legacy` 精确复现：`… --pairs 01-07 --legacy` → `36.970` | 否；仅用于说明历史数值的成因 |
| **另一组复核数值** | **原始配方未知**、本轮不可复现的记录 | 无法逐位复现（当前脚本给 2.166） | 否；**不引用具体数值**，不进入任何结论 |

**36.970 px 的成因（可精确分解）**：两项错误叠加——①相对位姿方向用了
camera2→camera1 的中心差、却配 camera1→camera2 的 R；②把 `E` 直接当 `F` 用在像素坐标上，
未做 `K₂⁻ᵀ E K₁⁻¹`。只引入其中一项错误时也会得到明显偏大的误差；两项同时错即
`--legacy` 复现的 **36.970 px**（该值可精确复现，其余分解值本报告不引用）。

**解读与边界**：B0 的三对为 01–02 **0.254**、04–05 **0.257**、01–07 **2.166 px**。
**链端图像对（01–07）的误差明显高于相邻对**：约为 01–02 的 8.5×、04–05 的 8.4×；
相对**该对自身**的 control 中位（0.420 px）约为 5.2×，而相邻对的比值分别为
0.254/0.288 ≈ 0.88×、0.257/0.179 ≈ 1.44×。这只能说明“**该受测图像对的位姿—图像自洽性弱于相邻对**”，
**不能**据此判定绝对几何精度（无真值），也**不能**确认原因是链式漂移
（配对拓扑只是与这一变化同时出现的变量，未做因果隔离）。

### 2.5 已排除的输出异常

| 异常 | 依据 |
|---|---|
| 崩溃 / OOM | B0 10 阶段 succeeded，峰值 3.259 GiB |
| NaN/Inf 点或损失 | `removed_nonfinite=0`；两条 loss 有限 |
| 某图 0 点 / 缺相机 | 7 相机、每图 30k–66k 点 |
| 索引/掩码错位 | 2.2 全部一致 |
| 数据损坏/缺图 | 静态 sha256、尺寸、SIFT 检查均通过 |

---

## 3. 假设与区分实验

> **编号说明**：本节 H1–H5 指“输出质量/成本”假设，与 `PREDICTIONS.md` P2 的
> 崩溃假设编号（H1 OOM、H2 空匹配、H3 NaN、H4 身份/顺序、H5 `fetch_img`、H6 断言）
> **不是同一套编号**，不要混读；两套编号的对应关系见 §3.2 的说明。

| ID | 假设 | 依据 | 如果是 H，那么应观察到 E | 区分实验（只改一个变量） | 判据 | 状态 |
|---|---|---|---|---|---|---|
| H1 | 显式链只约束相邻视图，非相邻视图缺少直接约束，导致链式漂移 | `configs/scene_b.yaml` 选择 `explicit`；manifest 只有 01-02-…-07 链式 6 边；B0 的 01–07 正确指标为 2.166 px | `complete` 运行后 01–07 的 Sampson 中位误差明显低于 B0；位姿/焦距变化；边数 6→21 | `--set pairing.strategy=complete`（run `20261126T043428Z_87ba5af92253`） | 01–07 误差显著下降 → 支持；不降/升高 → 不支持 | **已执行：01–07 2.166→0.517 px（4.2× 改善），H1 在几何一致性上成立；但成本 1,221,896 点 / 76.396 s（3.45×/3.09×），未作为最终修复** |
| H2 | `subsample=4` 使 anchor 过密，点数/耗时被放大 | scene_b 覆写 `subsample: 4`；点数 354,140 ≈ 14,734 × 6 边 × 4 倍采样 | `subsample=8` 后点数约降为 1/4，耗时下降 | `--set backend.subsample=8`（run `20261126T044206Z_4b4ac7dfd49d`） | 点数/耗时显著下降 → 成本侧支持 | **已执行：354,140→117,830 点、24.70→14.53 s；正确指标 01–07 2.166→2.026、04–05 0.257→0.247，无退化 → 成本侧支持** |
| H3 | `matching_conf_thr=0.0` 让低置信匹配进入 3D 损失，影响位姿 | base 设置 `matching_conf_thr: 0.0`；`sparse_ga.py:343` 用 `x.max() > thr` | 阈值提高后部分对转入 DUSt3R 分支，损失/位姿改变；点数应不变 | `--set backend.matching_conf_thr=5.0`（run `20261126T044458Z_d7fb4feff3c9`） | 产物明显变化 → 支持；几乎不变 → 不支持 | **已执行：点数不变，但正确指标全面变差（01–07 8.642、01–02 1.780、04–05 2.250 px）；损失骤降是目标函数切换，不能作为修复** |
| H4 | 优化数值发散导致位姿漂移 | `mast3r.py:329-331` 要求 2 条有限 loss | 日志出现 NaN/Inf 或 loss 条数不足 | 读 B0 `reconstruction_optimizer.log` | B0 两条 loss 有限 → **已排除** | 已排除（见 §3.2） |
| H5 | 输入身份/顺序在 N>2 时错位 | `sparse_pixels.py:49/53/72`、`sparse_io.py:51-52` | `BackendIOError` 或索引/掩码错位 | 2.2 的索引/掩码检查 | 全部一致 → **已排除** | 已排除（见 §3.2） |

> H1–H3 都是“输出质量/成本”假设，不是崩溃假设。H1 与 H2/H3 的区别在于：
> H1 改变**配对拓扑**，H2 改变**采样密度**，H3 改变**匹配损失分支**。

### 3.1 实验执行规则

- 每次命令只加一个 `--set`；`config_sources.json` 会记录 `source: "<CLI>"`；
- 执行前在 `PREDICTIONS.md` 写预测；执行后填对账表；
- 不修改 `data/`、manifest 的 sha256、`checkpoints/`、`vendor/`；
- 每次运行统一加 `LD_LIBRARY_PATH=` 前缀（原因见 `ENVIRONMENT.md` §4.5）。

三个区分实验实际使用的命令（当时 `configs/scenes/scene_b.yaml` 仍为 `subsample: 4`）：

```bash
# H1：只改配对拓扑
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py \
  --config configs/scenes/scene_b.yaml --set pairing.strategy=complete

# H2：只改采样密度（相对修复前配置）
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py \
  --config configs/scenes/scene_b.yaml --set backend.subsample=8

# H3：只改匹配阈值
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py \
  --config configs/scenes/scene_b.yaml --set backend.matching_conf_thr=5.0
```

> 交付配置已把 `scene_b.yaml` 的 `subsample` 改为 8。要在**当前**配置下逐值复现
> H1/H3（它们以 4 为基线），需再加 `--set backend.subsample=4` 固定该基线；
> 相对 B0 仍然只改一个变量（拓扑或阈值）。三个 run 的 `config_sources.json`
> 均只记录一个 `<CLI>` 字段；修复 run `20261126T044924Z_ff9b3efbfe8e` 无 CLI 覆盖。

### 3.2 H4/H5 的逐条核查（只读复算）

> 编号说明：本节按 `DIAGNOSIS.md` §3 的编号。H4（数值发散）对应
> `PREDICTIONS.md` P2 的 H3；H5（身份/顺序）对应 `PREDICTIONS.md` P2 的 H4。
> `PREDICTIONS.md` P2 的 H5（`fetch_img=None`）是另一条崩溃假设，B0 构造函数正常，
> 也未触发，但不在本节两条之内。

#### H4 — 优化数值发散导致位姿漂移

| 项目 | 内容 |
|---|---|
| 假设成立时应观察到 | `reconstruction_optimizer.log` 出现 `nan`/`inf`，或 `>> final loss` 少于 2 条；后端报 `Optimizer loss check failed` |
| 用什么检查区分 | 直接解析 B0 的 `reconstruction_optimizer.log`，统计 `>> final loss` 条数与有限性 |
| 实际执行了什么 | 对 B0（`20261126T042744Z_ffdbbf471afb`）的日志做只读解析 |
| 观察到什么 | 恰好 **2** 条 `>> final loss`：`0.022823568433523178`、`0.8855334520339966`；无 `nan`/`inf`；`run_status.status = succeeded` |
| 证据在哪 | `outputs/scene_b/20261126T042744Z_ffdbbf471afb/reconstruction_optimizer.log`；同目录 `run_status.json` |
| 结论范围 | **B0 本次未观察到数值发散**；这只否定该 run 的该假设，不证明其他配置/随机种子绝不会发散 |

是否需要新增重建实验：**不需要**。该假设的判据就是日志，B0 日志已足以区分。

#### H5 — 输入身份/顺序在 N>2 时错位

| 项目 | 内容 |
|---|---|
| 假设成立时应观察到 | `BackendIOError`（`Captured image paths/order differ...`、`Anchor keys/insertion order differ...`、`Sparse anchor rows differ...`），或索引/掩码错位 |
| 用什么检查区分 | B0 `point_sources.npz` 逐视图检查：`source_observation_index` 是否为 0…n−1、`view_mask_*` 是否与 `source_image_index` 一致 |
| 实际执行了什么 | 对 B0 的 `point_sources.npz` 与 `reconstruction.json.per_image_observation_counts` 做只读逐视图核对 |
| 观察到什么 | 7 个视图全部：唯一值数=观测数、范围 0…n−1、严格递增、`view_mask_i` 全 `true`、对应切片的 `source_image_index` 全等于视图号（数值见下表） |
| 证据在哪 | `outputs/scene_b/20261126T042744Z_ffdbbf471afb/artifacts/point_sources.npz`；同目录 `reconstruction.json` |
| 结论范围 | **B0 本次未观察到身份/顺序错位**；这只否定该 run 的该假设，不证明任何 N>2 配置都不会发生 |

| 视图 | 观测数 | `source_observation_index` 范围 | 唯一值数 | 严格 0…n−1 | `view_mask_i` 全 true | 切片 `source_image_index` 全等于 i |
|---|---:|---|---:|---|---|---|
| 01 | 33,272 | 0–33,271 | 33,272 | 是 | 是 | 是 |
| 02 | 65,992 | 0–65,991 | 65,992 | 是 | 是 | 是 |
| 03 | 59,716 | 0–59,715 | 59,716 | 是 | 是 | 是 |
| 04 | 54,294 | 0–54,293 | 54,294 | 是 | 是 | 是 |
| 05 | 54,069 | 0–54,068 | 54,069 | 是 | 是 | 是 |
| 06 | 56,784 | 0–56,783 | 56,784 | 是 | 是 | 是 |
| 07 | 30,013 | 0–30,012 | 30,013 | 是 | 是 | 是 |

是否需要新增重建实验：**不需要**。B0 本身就是 N=7，且已有逐视图索引/掩码产物，足以做该检查。

---

## 4. 根因与修复（已完成）

| 项目 | 内容 |
|---|---|
| 确认的根因 | 成本侧：`configs/scenes/scene_b.yaml:11` 把 `backend.subsample` 覆写为 **4**（`configs/base.yaml` 为 **8**），anchor 采样过密，输出观测数与重建耗时被放大。**不是**数据损坏、崩溃、NaN、身份/顺序错误。几何上唯一可观察到的现象是链端 01–07 的自洽性弱于相邻对（2.166 vs 0.254/0.257 px）；该现象是否由链式拓扑引起，本场景无真值、也未做因果隔离，**未确认**。 |
| 对应 run_id | B0 `20261126T042744Z_ffdbbf471afb`（354,140 点 / 24.702 s）；CLI 对照 `20261126T044206Z_4b4ac7dfd49d`（subsample=8，117,830 点 / 14.529 s） |
| 文件与行号 | `configs/scenes/scene_b.yaml:11` |
| 修复方式 | `subsample: 4` → `subsample: 8`（与 base 一致）。未改 `data/`、manifest 的 sha256、`checkpoints/` 权重或 `vendor/`。 |
| 证明已修复的 run | `20261126T044924Z_ff9b3efbfe8e`：117,830 点、14.614 s、7 相机、6 边/12 有向、10 阶段 succeeded |

**未采用的方案与原因**：

- **H1（explicit → complete）不作为最终修复**：run `20261126T043428Z_87ba5af92253`
  在受测的 01–07 上把该指标从 2.166 降到 **0.517 px**（约 4.2×）。这只描述**一个图像对上的变化**；
  它不能确认变化的原因是“补齐了链式漂移”，也没有证明整体几何精度提高（无真值）。
  成本上增至 1,221,896 点 / 76.396 s（点数 3.45×、耗时 3.09×），
  而 B0 本身是可正常完成的运行，因此按成本优先保留 explicit；
- **H3（matching_conf_thr 5.0）不采用**：run `20261126T044458Z_d7fb4feff3c9` 中点数不变，
  但正确指标全面变差（01–07 8.642、01–02 1.780、04–05 2.250 px）；
  损失从 `0.022824/0.885533` 变为 `0.000431/0.000252` 是大量对转入 DUSt3R 分支、
  目标函数已变的结果，**损失小不等于质量好**。

**这个修复能证明什么**：

- 输出观测数从 354,140 降到 117,830（0.333×），重建耗时 24.702→14.614 s；
- 配对拓扑、7 相机、每图都有点、索引/掩码一致性保持不变；
- scene_a 回归与基线一致。

**不能证明什么**：几何精度、尺度、与真值的一致性；也不能证明“点更少所以质量更好”。
正确指标下 B0 的三对误差为 0.254 / 0.257 / 2.166 px，均在亚像素到 2 px 量级，
只能说明**位姿与图像对极几何自洽**，不能说明绝对几何精度。
其中 01–07 高于相邻对约 8 倍，这一差异在 subsample=8 后仍然存在（2.026 px，
相对其 control 0.420 px 为 4.8×），说明它**不是** `subsample=4` 采样过密造成的。

---

## 5. 回归与验证（已完成）

| 检查 | 判据 | 状态 |
|---|---|---|
| scene_a 回归 | run `20261126T045015Z_65797318eef8`：10 阶段 succeeded；2 相机 / 14,734 点；`final_losses=[0.00931588001549244, 0.3066957890987396]`（与基线逐位相同）；峰值 3.168 GiB | ✅ 一致（重建耗时 3.715 s vs 基线 4.467 s，属计时波动） |
| scene_b 修复后重跑 | run `20261126T044924Z_ff9b3efbfe8e`：10 阶段 succeeded；7 相机；6 边/12 有向；117,830 点；14.614 s | ✅ |
| 三个必答题 | `source_observation_index` 语义、`scale_status=unverified`、`point_confidence=unavailable`；与 `BASELINE.md` §4 一致 | ✅ |
| `status: succeeded` ≠ 几何质量 | 明确写 `geometry_quality_accepted=false` | ✅ |

---

## 6. 结论

**scene_b 的 B0 输出不是崩溃，也不是坏数据**：7 相机、每图 30,013–65,992 个观测、
`source_observation_index` 均为 0…n−1、掩码与索引一致、无 NaN、10 阶段 succeeded。

**几何自洽性的可复现描述**（固定 SIFT+RANSAC 内点、像素 `F = K₂⁻ᵀ E K₁⁻¹`）：
01–02 / 04–05 / 01–07 分别为 0.254 / 0.257 / **2.166 px**。
**链端图像对（01–07）的误差明显高于相邻对**（约 8.5×）。现有数据足以描述这一差异，
但缺少几何真值，**无法据此判定绝对精度**，也**无法确认链式漂移就是原因**——
`complete` 实验只显示该对指标降到 0.517 px，没有做因果隔离。

**历史数值的边界**：`36.970 px` 是旧算法（方向取反 + `E` 当 `F`）的产物，
`--legacy` 可精确复现，不代表几何结论；复核记录中的其他数值原始配方未知、无法逐位复现，
**本报告不引用**。全文只引用可复现的当前指标。

**可操作的根因是成本偏离**：`configs/scenes/scene_b.yaml:11` 将 `subsample` 覆写为 4，
而 base 为 8，使同一份 7 图数据的观测数从 117,830 放大到 354,140（约 3 倍），
重建耗时从 14.614 s 增加到 24.702 s。改回 8 后：

- run `20261126T044924Z_ff9b3efbfe8e`：117,830 点 / 14.614 s / 7 相机，10 阶段 succeeded；
- scene_a 回归 run `20261126T045015Z_65797318eef8`：与基线逐位一致；
- 正确指标 01–07 2.166→2.026 px、04–05 0.257→0.247 px，无退化。

**H1（complete）** 在受测的 01–07 上把该指标从 2.166 降到 0.517 px，但成本 3.45×；
该变化只有一个图像对、且没有真值对照，因此不作为最终修复，也不宣称“链式漂移”已被证实。

**修复的边界**：以上只证明成本与流程恢复正常，不证明几何质量、尺度或真值一致性；
本场景没有标定几何真值，`scale_status` 仍为 `unverified`，
`geometry_quality_accepted` 仍为 `false`。
