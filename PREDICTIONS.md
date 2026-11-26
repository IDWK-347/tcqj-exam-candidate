# PREDICTIONS.md — 实验前预测与执行后对账

> 规则：**每条预测都在对应实验执行之前写下**，执行后在同一小节补"实际观察"与"对账结论"。
> 未执行的实验一律标注"未执行"，不提前填写结果。
> 预测可以写错；写错的预测原样保留，另记"不一致，原因是……"。

---

## P0. 入口预检（`--dry-run`，scene_a）

**状态：已执行（2026-11-26T03:56Z）**

预测：
1. `configuration` 与 `asset_validation` 两个阶段均为 `succeeded`；
2. 由于 `entry.py:132-137` 会对完整权重计算 SHA256，本次 dry-run 会比普通预检多出数十秒的纯 IO；
3. 输出目录下会生成 `resolved_config.yaml`、`config_sources.json`、`preflight.json`、`run_manifest.json`、`run_status.json`；
4. `run_manifest.json` 的 `phases` 只有 `configuration`、`asset_validation` 两项（`reconstruction_pipeline.py:68-69`）；
5. 不加载模型、不解码图像。

**可能证伪的观察**：出现 `Checkpoint SHA256 mismatch`（权重不完整）、
`Asset path escapes its permitted directory`（路径越界）、或 `CUDA was requested, but CUDA is unavailable`。

### 实际观察

- run_id：`20261126T035611Z_077779e0fdea`
- `run_status.json`：`status: "succeeded"`，`error: null`
- `run_manifest.json`：`mode: "reconstruction"`，`provenance.dry_run: true`
- 阶段：
  - `configuration` succeeded，0.04 s
  - `asset_validation` succeeded，**2.12 s**
- 产物齐全：`resolved_config.yaml`、`config_sources.json`、`preflight.json`、
  `run_manifest.json`、`run_status.json`
- `preflight`：`images=2`、`directed_pairs=2`、`weight_sha=e28f91b4…`
- `run_status.json` 中 `geometry_quality_accepted: false`

### 对账

| 预测 | 实际 | 结论 |
|---|---|---|
| 两阶段 succeeded | 均 succeeded | 一致 |
| 权重哈希带来“数十秒”纯 IO | `asset_validation` 仅 2.12 s（约 1.3 GB/s） | **不一致**：预测偏保守；权重已在页缓存中，且本机 SHA256 很快 |
| 5 个证据文件齐全 | 齐全 | 一致 |
| `phases` 只有两项 | 只有两项 | 一致 |
| 不加载模型/不解码图像 | `mode=reconstruction`、`dry_run=true`，无后端阶段 | 一致 |

dry-run 通过只证明**资源与参数校验**通过，不证明推理或几何质量。

---

## P1. scene_a 基线（完整运行）

**状态：已执行（成功 run：`20261126T042248Z_3d7558c62c88`）**

> 前置说明：本预测写于运行之前。前三次执行均因环境问题
> （`LD_LIBRARY_PATH` 混用 CUDA 12.2，`CUBLAS_STATUS_NOT_INITIALIZED`）在
> `backend_reconstruct` 失败（run `20261126T041729Z_4b2acf11e162`、
> `20261126T041937Z_581439180005`、`20261126T042110Z_7904ef813fba`），
> 与环境修复后的成功 run 无关。详见 `ENVIRONMENT.md` §4.5。
>
> 预测：
1. 10 个阶段全部 `succeeded`；
2. `pair_graph.json` 中 `logical_edge_count = 1`，`directed_item_count = 2`；
3. `standard_result.json` 中 `camera_count = 2`，`scale_status = "unverified"`，
   `availability.point_confidence.status = "unavailable"`；
4. `availability.source_pixels_xy.status = "available"`（后端在构造时捕获了 anchor 像素）；
5. 处理后尺寸为 384×512（`long_edge=512`，原图 1448×1086，宽高比 4:3）；
6. `reconstruction.json` 的 `final_losses` 恰好 2 条且均为有限值
   （`mast3r.py:329-331` 强制要求）；
7. `geometry_quality_accepted` 为 `false`（代码中该字段恒为 false，不随质量变化）。

**不能由本次运行证明的**：几何质量、尺度、与真值的一致性。

### 实际观察

- run_id：`20261126T042248Z_3d7558c62c88`，`status: succeeded`
  （产物未记录进程退出码，以 `run_status.status` 为准）
- 10 个阶段全部 `succeeded`，总耗时 18.04 s
- `pair_graph`：`nodes=2`、`logical_edge_count=1`、`directed_item_count=2`，
  连通分量 `[['01.png','02.png']]`
- `camera_count=2`、`point_count=14,734`（每图 7,367）
- `scale_status="unverified"`、`point_confidence.status="unavailable"`、
  `source_pixels_xy.status="available"`
- `processed_hw=[[384,512],[384,512]]`
- `final_losses=[0.00931588001549244, 0.3066957890987396]`（2 条，均有限）
- `geometry_quality_accepted=false`
- `reconstruction_seconds_excluding_model_load=4.467`，
  `peak_allocated_gib=3.168`

### 对账

| 预测 | 实际 | 结论 |
|---|---|---|
| 10 阶段 succeeded | 10 阶段 succeeded | 一致 |
| 1 条无向边 / 2 条有向对 | 1 / 2 | 一致 |
| 2 相机；scale unverified；confidence unavailable | 2；unverified；unavailable | 一致 |
| source_pixels_xy available | available | 一致 |
| 384×512 | `[[384,512],[384,512]]` | 一致 |
| 恰好 2 条有限 loss | 2 条，均有限 | 一致 |
| `geometry_quality_accepted=false` | false | 一致 |

---

## P2. scene_b 首次运行 —— 候选假设

**状态：已执行（B0 run `20261126T042744Z_ffdbbf471afb`）：6 条崩溃假设全部未触发**

### 已排除（静态检查，不需要运行）

（以下检查于 2026-11-26T03:12–03:19Z 在本工作区执行；数据与预处理的复现方式见
`DIAGNOSIS.md` §2.1。）

| 假设 | 排除依据 |
|---|---|
| 配对图断开 | `build_pair_graph` 实测：7 节点、6 条无向边、12 条有向对、单一连通分量 |
| 图像损坏 / 重复 / 离群 | SIFT + BFMatcher(k=2)、ratio 0.75：21 个无序对全部 648–805 good matches，无异常低值；7 张图尺寸、模式一致（早期记录 368–450 来自不同 ratio 阈值，结论相同） |
| 预处理失败 | 静态跑 `load_scene` 成功，7 条记录全部 384×512，顺序 01…07，无 warning |
| manifest 声明的 sha256 不符 | 逐图 SHA256 与 manifest 对照：12/12 OK（三个场景） |
| vendor 哈希固定检查失败 | `vendor/manifest.json` = `140d7489…ec0579`（与 `mast3r.py:25` 一致）；`sparse_ga.py` = `195cc751…73ad74`（与 `sparse_io.py:16` 一致） |

### H1 — 显存不足（OOM）

依据：`scene_b.yaml` 把 `backend.subsample` 从 base 的 8 覆写为 **4**。

需要先区分两个量（静态算术，不需要运行）：

| subsample | anchor 网格（每图 `core_depth` 长度） | 说明 |
|---|---|---|
| 8 | 48×64 = 3,072 | base 配置 |
| 4 | 96×128 = 12,288 | scene_b |
| 16 | 24×32 = 768 | 实验用 |

但**点数不是网格决定的**：`condense_data` 里 `img_anchors` 存的是各对匹配像素的拼接
（`sparse_ga.py:807-813`），历史 2 图 run 每图 8,164 行，远大于网格 3,072。
所以点数主要由**匹配候选数**驱动，而 subsample 是匹配步长（`extract_correspondences`，
`sparse_ga.py:663-664`），步长减半约使每对候选数变为 4 倍。

静态估算：scene_b 有 12 条有向对（scene_a 只有 2 条）；若每对对应数与 scene_a 同量级再乘 4，
链式 7 图总行数约 30–40 万。这个规模本身不大（几十 MB 级），因此
**OOM 若发生，更可能在匹配阶段（候选点 4 倍、NN 检索）而非点云存储**。

倾向：中等偏低。保留为候选，不预先排除。

**如果是 H1，那么应观察到 E1**：
`backend_reconstruct` 阶段失败，异常为 `torch.OutOfMemoryError` 或包含
`CUDA out of memory` 的消息；`run_manifest.json` 中 `phases[*].error.phase == "backend_reconstruct"`；
显存峰值接近 15 GiB。

区分实验：同一份数据，只把 subsample 改回 8（`--set backend.subsample=8`）。
若通过 → 支持 H1；若同样失败且报错相同 → 不支持 H1。

### H2 — 某对图像匹配为空导致张量操作异常

依据：`sparse_ga.py:343` 的 `matching_check(x) = x.max() > matching_conf_thr`，
对空张量调用 `max()` 会抛 `RuntimeError`；本工程配置 `matching_conf_thr: 0.0`，
意味着所有对都被判为"匹配可用"，一旦某对的实际匹配为空，就会走进这条路径。

**如果是 H2，那么应观察到 E2**：
异常消息包含 `max()`、`Expected reduction dim` 或 `numel() == 0`，
且发生在 `sparse_global_alignment` 内部（`backend_reconstruct` 阶段）。

区分实验：
- 实验 2a：把 `matching_conf_thr` 提高到上游默认附近的 5.0（`--set backend.matching_conf_thr=5.0`），
  若报错消失或变为其他错误 → 支持 H2 的"匹配分支"判断；
- 实验 2b：改用 `complete` 配对（需要临时配置，执行前单独记录），
  若报错消失 → 指向 explicit 链上的某一对。

### H3 — 优化数值发散（NaN / 非有限损失）

依据：`optimize_loop` 在迭代中检测到 NaN 会**静默 break**（`sparse_ga.py:462-464`），
随后在循环外用最终参数重算损失（`:468-473`）；`mast3r.py:329-331` 要求恰好解析出
2 条 `>> final loss =` 且均有限，否则报 `Optimizer loss check failed`。

**如果是 H3，那么应观察到 E3**：
`BackendError: Optimizer loss check failed; inspect reconstruction_optimizer.log`；
`reconstruction_optimizer.log` 中 final loss 出现 `nan`/`inf`，或只出现 0–1 条 `>> final loss`。

区分实验：读取该 run 的 `reconstruction_optimizer.log`。
再降学习率重跑（`--set backend.coarse_lr=0.03 --set backend.refine_lr=0.005`），
若 NaN 消失 → 支持数值发散；若仍失败且报错相同 → 不支持。

### H4 — 输入身份 / 顺序在 N>2 时错位

依据：`sparse_pixels.py:48-53` 要求捕获到的图像路径顺序与 anchors 的键顺序严格一致；
`sparse_io.py:51` 要求图像 ID 顺序与配对图节点一致。2 图时这些条件很容易偶然成立。

**如果是 H4，那么应观察到 E4**：
`BackendIOError`，消息为 `Captured image paths/order differ from prepared inputs`、
`Anchor keys/insertion order differ from image order` 或
`Sparse anchor rows differ from optimized point rows`。

区分实验：同一份代码跑 scene_a（2 图）应通过；若 scene_a 通过而 scene_b 失败，
且错误消息属于上述几类 → 支持 H4。

### H5 — `SparseGA` 找不到某张图的 RGB（配对未覆盖）

依据：`sparse_ga.py:36-45` 的 `fetch_img` 遍历 `pairs_in` 按 `instance` 匹配，
匹配不到时返回 `None`，随后 `:55` 的 `im[y, x]` 会抛 `AttributeError`。

**如果是 H5，那么应观察到 E5**：
`AttributeError: 'NoneType' object is not subscriptable`，发生在 `SparseGA.__init__`。

**静态倾向：不太可能**。scene_b 的 explicit 链 01-02-…-06-07 覆盖了全部 7 张图，
每张图至少出现在一条边中。

---

### H6 — 规范化深度出现非正值，触发断言

依据：`sparse_ga.py:916` 的 `assert (core_depth > 0).all()`；
而 `canonical_view`（`sparse_ga.py:866-904`）对多视角深度做置信度加权平均、再用平均夹角重建深度，
图数越多，某个像素的聚合深度越可能落到 ≤0。2 图时几乎不会发生。

**如果是 H6，那么应观察到 E6**：
`AssertionError`（无消息），栈落在 `anchor_depth_offsets` / `prepare_canonical_data`，
发生在 `backend_reconstruct` 阶段、`forward` 完成之后。

区分实验：
- 看 traceback 最后几帧是否包含 `anchor_depth_offsets`；
- 用 `--set backend.subsample=8` 重跑，若同一位置仍断言 → 与采样无关，更支持 H6；
  若消失 → 指向采样步长而非多视图聚合。

### 实际观察（B0 run `20261126T042744Z_ffdbbf471afb`）

产物路径：`outputs/scene_b/20261126T042744Z_ffdbbf471afb/`

- 状态：`succeeded`，10 阶段全部 succeeded，`error: null`；
- 7 相机，354,140 个观测（01: 33,272；02: 65,992；03: 59,716；04: 54,294；
  05: 54,069；06: 56,784；07: 30,013）；
- `final_losses=[0.022823568433523178, 0.8855334520339966]`，两条均有限；
- 重建耗时 24.702 s，allocator 峰值 3.259 GiB（16 GB T4，远未接近显存上限）；
- 6 条无向边 / 12 条有向对，单一连通分量；
- `reconstruction_optimizer.log` 恰好 2 条 `>> final loss`，无 NaN/Inf；
- `source_observation_index` 每图均为 0…n−1，`view_mask_*` 与 `source_image_index` 一致。

### 对账

| 假设 | 预测的可观察证据 | 实际 | 结论 |
|---|---|---|---|
| H1 OOM | `CUDA out of memory`，峰值接近 15 GiB | 峰值 3.259 GiB，无 OOM | **未触发** |
| H2 空匹配 `max()` | 消息含 `max()`/`numel()==0` | 无此错误，日志正常 | **未触发** |
| H3 NaN 损失 | `Optimizer loss check failed` | 两条 loss 有限 | **未触发** |
| H4 身份/顺序 | `BackendIOError` 路径/顺序/形状不符 | 索引 0…n−1，掩码一致 | **未触发** |
| H5 `fetch_img` None | `AttributeError: NoneType` | 构造函数正常 | **未触发** |
| H6 `core_depth` 断言 | 无消息 `AssertionError` | 无断言 | **未触发** |

> 六个崩溃假设全部未触发：**B0 的真实问题是成本（点数/耗时），不是崩溃**。
> 后续 H1–H3（P5–P7）改为针对“输出质量/成本”的设计。
>
> 说明：P2 中为崩溃假设设计的补充实验（H2 的 2a/2b、H3 的降学习率重跑、H6 的
> `subsample=8` 对照）在 B0 未触发对应假设的前提下**没有全部执行**。其中真正执行并产生
> run 的是 `complete`（`20261126T043428Z_87ba5af92253`）与 `thr=5.0`
> （`20261126T044458Z_d7fb4feff3c9`），它们后来作为 P5/P7 的输出诊断实验使用；
> 未执行的实验不出现在任何结论中。

---

## P3. 参数实验预测（`subsample` / `matching_conf_thr`）

**状态：已执行（E1–E4 四个 run，见下文各小节）**（scene_b 已修复、scene_a 基线稳定；一次只改一个参数）

**场景说明**：以下 E1–E4 都在 **scene_a** 上执行，基线为 run
`20261126T042248Z_3d7558c62c88`（`subsample=8`、`matching_conf_thr=0.0`）。
scene_b 的实验只用于诊断，不混入本表。

### P3.1 `backend.subsample` 8 → 16

预测：
1. anchor 网格降为 1/4（24×32=768/图）；匹配候选数约降为 1/4，**点数大致同比例下降但非严格成比例**
   （点数由对应关系决定，见 P2/H1 的说明）；
2. 优化迭代的每次前向/反向开销下降，`reconstruction_seconds_excluding_model_load` 降低；
3. 峰值显存下降；
4. 点云变稀，**无法从本工程产物判断几何质量是否变好**
   （`geometry_quality_accepted` 恒为 false，且没有真值）；
5. 损失数值会变化，但**不同目标下的损失不可直接比较**。

可证伪点：若点数或耗时没有明显变化，说明 subsample 未按预期传播。

**实际观察（run `20261126T045415Z_bc7519ec6124`）**：点数 3,814（基线 14,734 的 0.259×）；
重建耗时 3.298 s（基线 4.467 s）；峰值 3.168 GiB（与基线相同）；2 相机；
`final_losses=[0.009082280099391937, 0.31482994556427]`。
→ 点数约 1/4、耗时下降与预测一致；峰值未下降，原预测第 3 条不成立。

### P3.2 `backend.matching_conf_thr` 0.0 → 5.0

预测：
1. `sparse_ga.py:343-350` 中，低置信度的对会从 `loss3d_slices` 转到 `dust3r_slices`，
   即从 3D 匹配损失退化为 DUSt3R 回归损失；
2. 若某些对的匹配置信度整体低于 5.0，会明显改变优化轨迹（损失曲线、最终点位）；
3. 不改变点数上限，但会改变参与优化的对应数量，从而可能改变耗时与显存；
4. 同样**不能**据此断言几何质量变好或变差。

可证伪点：若所有对的 `confs.max()` 都远大于 5.0，则该参数在本数据上几乎不生效，
应观察到产物几乎不变。

**实际观察（run `20261126T045752Z_a8af27bf9b5f`）**：点数 14,734、2 相机；
`final_losses=[0.00931588001549244, 0.3066957890987396]`，与基线逐位相同；
峰值 3.168 GiB；重建耗时 3.684 s（计时波动）。
→ 阈值 5.0 在 scene_a 上**未触发任何分支切换**，属于预测中的“几乎不生效”情形。

### P3.3 `backend.subsample` 8 → 4（scene_a）

预测：
1. anchor 采样更密，点数上升（可能约 3–4 倍，但不保证严格比例）；
2. 重建耗时上升；峰值显存可能上升；
3. 配对数仍为 1 边 / 2 有向，相机数仍为 2；
4. 损失会变化，但**不可与基线横向比较质量**；
5. `geometry_quality_accepted` 仍为 `false`。

可证伪点：点数/耗时几乎不变 → `subsample` 未按预期传播。

**实际观察（run `20261126T045607Z_b7d39c4a2ae0`）**：点数 53,878（基线 14,734 的 3.66×）；
重建耗时 4.934 s（基线 4.467 s）；峰值 3.175 GiB；2 相机；
`final_losses=[0.008976246230304241, 0.2938293218612671]`。
→ 点数/耗时上升与预测一致；峰值仅轻微上升。

### P3.4 `backend.matching_conf_thr` 0.0 → 1.5（scene_a）

预测：
1. 点数应与基线完全相同（该参数只影响损失分支，不改变对应关系提取）；
2. 若 scene_a 这一对的 `confs.max()` 远大于 1.5，产物几乎不变；
   若接近阈值，损失/位姿会变化；
3. 不能据此断言几何质量变好或变差。

可证伪点：点数和损失都完全不变 → 阈值在 scene_a 上不生效。

**实际观察（run `20261126T045956Z_817a38ed404d`）**：点数 14,734、2 相机；
`final_losses=[0.00931588001549244, 0.3066957890987396]`，与基线逐位相同；
峰值 3.168 GiB；重建耗时 3.639 s（计时波动）。
→ 阈值 1.5 同样未触发，符合“点数/损失完全不变”的证伪条件。

---

## P4. scene_c 迁移（不为它单独调参）

**状态：已执行（run `20261126T083005Z_865eb24fedb1`）**

预测：
1. 配置为 `complete`、3 张图（03/05/07），因此 `logical_edge_count = 3`、
   `directed_item_count = 6`、`camera_count = 3`；
2. 输入图就是 scene_b 的 03/05/07（sha256 完全相同），但配对策略不同
   （`complete` 3 条边 vs `explicit` 链 6 条边），**不能把 scene_b 的结果直接外推**；
3. 沿用基线参数（`subsample=8`、`matching_conf_thr=0.0`），不做任何 `--set`；
   若 scene_b 的根因是配置/代码且已修复，scene_c 应无需额外调整即可完成；
4. 点数应介于 scene_a（2 图）与 scene_b（7 图）之间，但这**不是严格单调保证**
   （点数由对应关系决定，不是图数的线性函数）；
5. `scale_status` 仍为 `unverified`、`point_confidence` 仍为 `unavailable`、
   `geometry_quality_accepted` 仍为 `false`。

**可证伪的观察**：若 scene_c 失败，说明修复不通用或方案依赖场景规模/拓扑；
必须记录为方案的失效条件，而不是靠给 scene_c 调参掩盖。

### 实际观察（run `20261126T083005Z_865eb24fedb1`）

产物路径：`outputs/scene_c/20261126T083005Z_865eb24fedb1/`

- 状态：`succeeded`，10 阶段全部 succeeded，`error: null`；
- 配置来源：`config_sources.json` 的 `load_order` 只有
  `configs/base.yaml`、`configs/t4.yaml`、`configs/scenes/scene_c.yaml`，
  **没有任何 CLI 覆盖**；`subsample` 来源为 `configs/base.yaml`（值 8），
  `pairing.strategy` 来源为 `configs/base.yaml`（`complete`）；
- 配对：`logical_edge_count=3`、`directed_item_count=6`、
  连通分量 `[['03.png','05.png','07.png']]`；
- 相机数：**3**；点数：**55,580**（03: 18,428；05: 18,731；07: 18,421）；
- `final_losses=[0.020873527973890305, 1.2771258354187012]`；
- 重建耗时：**7.892 s**；allocator 峰值：**3.256 GiB**；
- 处理后尺寸：3 张全部 `(384, 512)`；
- `scale_status=unverified`、`point_confidence=unavailable`、
  `source_pixels_xy=available`、`geometry_quality_accepted=false`；
- 导出读回：`readback_checks` 五项均为 `true`（`scene.ply` 1,500,874 B、`cameras.json` 5,373 B、
  `point_sources.npz` 231,249 B）。

### 对账

| 预测 | 实际 | 结论 |
|---|---|---|
| 3 边 / 6 有向 / 3 相机 | 3 / 6 / 3 | 一致 |
| 不调参、沿用基线参数 | `load_order` 无 CLI 覆盖，`subsample=8`、`complete` 来自 base | 一致 |
| 无需额外调整即可完成 | 10 阶段 succeeded，`error: null` | 一致 |
| 点数介于 scene_a 与 scene_b 之间 | 55,580（scene_a 14,734 < 55,580 < scene_b 117,830） | 一致（且非严格单调的说明成立） |
| `unverified` / `unavailable` / `false` | 全部一致 | 一致 |

**说明**：scene_c 成功不证明几何质量；`geometry_quality_accepted` 仍为 `false`。
scene_c 的图片是 scene_b 的 03/05/07 子集，但配对策略为 `complete`（3 边），
与 scene_b 的 explicit 链（6 边）不同，因此结果不能直接互相外推。

---

## P5. scene_b 输出诊断：显式链 vs 完整配对

**状态：已执行（run `20261126T043428Z_87ba5af92253`）**

B0 已跑通（`20261126T042744Z_ffdbbf471afb`）：7 相机 / 354,140 点 / 24.70 s，
`final_losses=[0.022823568, 0.885533452]`，峰值 3.259 GiB。
输出内部一致（每图都有点、索引 0…n-1、掩码与 `source_image_index` 一致、无 NaN）。

**H1：显式链只约束相邻视图，非相邻视图缺少直接约束，导致链式漂移。**

如果是 H1，那么应观察到 E1：

1. `--set pairing.strategy=complete` 后，`logical_edge_count` 从 6 变为 **21**，
   `directed_item_count` 从 12 变为 **42**；
2. 点数随边数增加而上升（约 3–4 倍），耗时上升；
3. 用重建位姿计算的 SIFT 匹配 Sampson 中位误差：非相邻对（尤其 01–07）
   **明显低于 B0 的约 37 px**；
4. 相机焦距/位姿发生可观察的变化，但仍是 7 相机、单连通分量。

可证伪点：若 01–07 的 Sampson 误差不降或反而升高，则不支持 H1。

> 该实验只改 `pairing.strategy` 一个变量；不改 data、不改 manifest 的边。

### 实际观察（run `20261126T043428Z_87ba5af92253`）

- 状态：`succeeded`，7 相机；
- `logical_edge_count=21`、`directed_item_count=42`；
- 点数：**1,221,896**（B0 的 3.45 倍）；
- 重建耗时：**76.396 s**；allocator 峰值 3.262 GiB；
- `final_losses=[0.022955453, 0.927552342]`；
- 焦距相对跨度：**5.14%**（B0 为 7.62%）；
- 正确对极指标（`tools/measure_epipolar.py`，固定 SIFT+RANSAC 内点）：
  01–02 **0.200**（B0 0.254）、04–05 **0.300**（B0 0.257）、
  **01–07 0.517（B0 2.166，改善约 4.2×）**。

> 原预测中的“约 37 px”基于旧算法（方向取反 + `E` 当 `F`），已被 `--legacy` 精确复现为
> 36.970 px，不代表几何结论；**当前指标**下 B0 的 01–07 为 2.166 px（可复现）。
> 若按原措辞“明显低于 B0”，在这一对上的实测成立（2.166→0.517 px）。
>
> 口径说明：脚本在原始 1448×1086 图上提取 SIFT、再按比例映射到处理后坐标，
> 省略了预处理中的亚像素平移（tx=ty=−0.3232，影响 ≈0.002 px）；
> 复核记录中的其他数值原始配方未知、当前脚本不能复现，不作为比较基准。

### 对账

| 预测 | 实际 | 结论 |
|---|---|---|
| 21 边 / 42 有向 | 21 / 42 | 一致 |
| 点数约 3–4 倍 | 3.45 倍 | 一致 |
| 非相邻 01–07 误差明显下降 | 2.166 → 0.517 px | **一致（改善 4.2×）** |
| 焦距/位姿改变、仍 7 相机单连通 | 焦距跨度 7.62%→5.14%，仍 7 相机单连通 | 部分一致 |

**结论：H1 在“受测图像对的自洽性”上成立，但不足以确认因果。** 完整配对把 **01–07**
的该指标从 2.166 降到 0.517 px——这只是**一个图像对上的变化**；01–02（0.254→0.200）
与 04–05（0.257→0.300）一降一升，没有一致方向。缺少几何真值，无法据此判定绝对精度，
也无法确认变化的原因是“补齐了链式漂移”。
成本增至 1,221,896 点 / 76.396 s（点数 3.45×、耗时 3.09×），因此未作为最终修复方案。

---

## P6. scene_b 输出诊断：`subsample` 4 → 8

**状态：已执行（CLI run `20261126T044206Z_4b4ac7dfd49d`；配置修复 run `20261126T044924Z_ff9b3efbfe8e`）**

**H2：`subsample=4` 使 anchor 过密，点数与耗时被放大，但没有带来几何一致性收益。**

如果是 H2，那么应观察到 E2：

1. 6 条无向边 / 12 条有向对不变，仍是 7 相机；
2. 点数从 354,140 降到约 **1/4**（约 8–9 万，允许 7–10 万）；
3. `reconstruction_seconds_excluding_model_load` 明显下降（B0 为 24.70 s）；
4. Sampson 中位误差（01–02、04–05、01–07）与 B0 同量级，**不出现系统性恶化**；
5. 损失值会变化，但不同采样密度下的损失不直接可比。

可证伪点：点数没有明显下降 → `subsample` 未按预期传播；
或 Sampson 误差显著恶化 → 采样密度确实影响几何一致性，H2 的“无收益”判断不成立。

### 实际观察（run `20261126T044206Z_4b4ac7dfd49d`）

- 状态：`succeeded`，6 边 / 12 有向，7 相机；
- 点数：**117,830**（B0 的 0.333 倍，不是严格 1/4）；
- 重建耗时：**14.529 s**（B0 24.702 s）；
- allocator 峰值：3.259 GiB；
- `final_losses=[0.023670070, 0.971985936]`；
- 焦距相对跨度：8.51%（B0 7.62%）；
- 正确对极指标（`tools/measure_epipolar.py`）：01–02 **0.234**（B0 0.254）、
  04–05 **0.247**（B0 0.257）、**01–07 2.026（B0 2.166，略降）**。

### 对账

| 预测 | 实际 | 结论 |
|---|---|---|
| 6 边 / 12 有向、7 相机 | 一致 | 一致 |
| 点数约 1/4（8–9 万） | 11.8 万（约 1/3） | 部分不一致：点数由匹配候选决定，不是网格的严格比例 |
| 耗时明显下降 | 24.70 → 14.53 s | 一致 |
| 对极误差不系统性恶化 | 01–07 2.166 → 2.026 px；04–05 0.257 → 0.247 px | **一致（无退化）** |

**说明**：正确指标下，降采样后三对误差均未恶化，01–07 还略降。
因此 H2 在成本侧成立（354,140→117,830 点、24.70→14.53 s），
几何一致性也没有变差。

**修复后的正式配置 run**：把 `configs/scenes/scene_b.yaml:11` 改为 `subsample: 8` 后，
不带 `--set` 重跑得到 `20261126T044924Z_ff9b3efbfe8e`：
117,830 点、14.614 s、7 相机、6 边/12 有向、10 阶段 succeeded。
它与上面的 CLI 覆盖 run 的产物指标一致，证明修复来自配置文件而不是命令行。

---

## P7. scene_b 输出诊断：`matching_conf_thr` 0.0 → 5.0

**状态：已执行（run `20261126T044458Z_d7fb4feff3c9`）**

**H3：`matching_conf_thr=0.0` 让低置信匹配进入 3D 损失，改变优化轨迹/位姿。**

如果是 H3，那么应观察到 E3：

1. 6 边 / 12 有向、7 相机不变；
2. **点数应与 B0 完全相同**（该参数只影响损失分支，不改变对应关系提取）；
3. `final_losses` 改变，优化轨迹/耗时可能变化；
4. 若低置信匹配确实损害位姿，04–05、01–07 等对的一致性可能改善；
   若所有对的 `confs.max()` 都远大于 5.0，则产物几乎不变。

可证伪点：若点数、位姿、损失都与 B0 完全相同，说明阈值在本数据上几乎不生效，
H3 不支持。

### 实际观察（run `20261126T044458Z_d7fb4feff3c9`）

- 状态：`succeeded`，6 边 / 12 有向，7 相机；
- 点数：**354,140**，与 B0 完全相同（符合预测）；
- 重建耗时：25.406 s（B0 24.702 s）；
- allocator 峰值：3.259 GiB；
- `final_losses=[0.000431214, 0.000251814]`（B0 为 0.022824 / 0.885533）；
- 焦距范围：599.63–646.89，相对跨度 7.59%；
- 相机中心最大两两距离（口径：由 `cameras.json` 的 `T_world_from_camera` 按 `C = −Rᵀt` 换算相机中心；保留 3 位小数）：**0.102**（B0 0.218；4 位小数为 0.1016）；
- 正确对极指标（`tools/measure_epipolar.py`）：01–02 **1.780**（B0 0.254）、
  04–05 **2.250**（B0 0.257）、**01–07 8.642（B0 2.166，全面变差）**。

### 对账

| 预测 | 实际 | 结论 |
|---|---|---|
| 点数与 B0 完全相同 | 354,140 | 一致 |
| 6 边 / 12 有向、7 相机 | 一致 | 一致 |
| 损失改变 | 0.0228/0.8855 → 0.000431/0.000252 | 一致，但目标函数已变，**不可横向比较** |
| 位姿可能改变 | 焦距/相机中心明显变化 | 一致 |

**结论：H3 得到支持（阈值确实强烈影响优化），但不能作为修复方案。**
正确指标下三对全部变差（01–02 0.254→1.780、04–05 0.257→2.250、01–07 2.166→8.642 px）；
损失骤降是大量对转入 DUSt3R 分支、目标函数改变的结果，**损失小不等于质量好**。

---

## 对账表（执行后填写）

| 实验 | 预测要点 | 实际观察 | 结论 |
|---|---|---|---|
| P0 dry-run | 两阶段成功、多几十秒 IO | 两阶段成功；`asset_validation` 仅 2.12 s（预测偏保守） | 通过；哈希耗时预测已修正 |
| P1 scene_a 基线 | 10 阶段成功、2 相机、1 边/2 有向 | `20261126T042248Z_3d7558c62c88`：10 阶段成功、2 相机、1 边/2 有向、14,734 点 | 全部一致 |
| P2 scene_b H1（OOM） | `backend_reconstruct` OOM、峰值接近 15 GiB | B0 `20261126T042744Z_ffdbbf471afb` 10 阶段 succeeded，峰值 3.259 GiB | 未触发；崩溃假设不成立 |
| P2 scene_b H2（空匹配 `max()`） | 消息含 `max()`/`numel()==0` | B0 无此错误；`reconstruction_optimizer.log` 有两条正常 `>> final loss` | 未触发 |
| P2 scene_b H3（NaN 损失） | `Optimizer loss check failed` | B0 `final_losses=[0.022823568, 0.885533452]`，均有限 | 未触发 |
| P2 scene_b H4（身份/顺序） | `BackendIOError` 路径/顺序/形状不符 | B0 7 图均有观测；`source_observation_index` 0…n−1；掩码一致 | 未触发 |
| P2 scene_b H5（`fetch_img` None） | `AttributeError: NoneType` | B0 构造函数正常，7 相机 | 未触发 |
| P2 scene_b H6（core_depth 断言） | 无消息 `AssertionError` | B0 succeeded，无断言 | 未触发 |
| P3.1 subsample=16 | 点数 ~1/4、耗时/显存下降 | `20261126T045415Z_bc7519ec6124`：3,814 点（0.259×）、3.298 s、3.168 GiB | 点数/耗时一致；峰值未下降 |
| P3.2 matching_conf_thr=5.0 | 低置信对转 DUSt3R 分支 | `20261126T045752Z_a8af27bf9b5f`：与基线逐位相同 | 阈值未触发；预测中的“不生效”分支成立 |
| P3.3 subsample=4 | 点数上升、耗时上升 | `20261126T045607Z_b7d39c4a2ae0`：53,878 点（3.66×）、4.934 s、3.175 GiB | 一致 |
| P3.4 matching_conf_thr=1.5 | 点数不变、产物可能不变 | `20261126T045956Z_817a38ed404d`：与基线逐位相同 | 一致；阈值未触发 |
| P4 scene_c | 3 边/6 有向、3 相机、不调参可完成 | `20261126T083005Z_865eb24fedb1`：3/6、3 相机、55,580 点、7.892 s；无 CLI 覆盖 | 全部一致 |
| P5 scene_b complete | 21 边/42 有向、点数上升、01–07 对极误差下降 | 21/42、1,221,896 点；当前指标 01–07 2.166→0.517 px（仅此一对） | 边数/点数一致；仅描述 01–07 的变化，无真值、未做因果隔离 → 不足以确认“链式漂移” |
| P6 scene_b subsample=8 | 点数约 1/4、耗时下降、对极误差不恶化 | 117,830 点（0.333×）、14.53 s；正确指标 01–07 2.166→2.026 px | 成本侧一致；误差无退化 → 一致 |
| P7 scene_b thr=5.0 | 点数不变、损失/位姿可能变化 | 点数 354,140 不变；损失 0.000431/0.000252；正确指标 01–07 2.166→8.642 px | 点数一致；阈值有效但几何全面变差，不能作为修复 |
| 修复验证 | 配置改为 8 后与 CLI 对照一致；scene_a 回归与基线一致 | scene_b `20261126T044924Z_ff9b3efbfe8e`：117,830 点 / 14.614 s / 7 相机；scene_a `20261126T045015Z_65797318eef8`：14,734 点 / 2 相机 / 两条 loss 逐位相同 | 一致 |
