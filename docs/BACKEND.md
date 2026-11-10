# MASt3R sparse-GA 后端适配

独立输入/输出边界与 `prepare / reconstruct / close` 生命周期已经接入重建入口。
稀疏路线此前已用七张实际图片完成 GPU 接入验证，记录见
`benchmarks/scene_01/backend_integrated.json`。该旧运行发生在新的正常 pipeline 入口与 anchor xy 捕获接入之前，不能作为当前入口或 GPU xy 来源验收，也不代表几何质量验收。
独立 probe 接口已完成实际 CUDA FP32 两图匹配验证和显示采样的人工视觉检查。
阶段记录由 `scripts/record_backend_probe.py` 从既有结果校验后生成，见
`benchmarks/scene_01/backend_probe_integrated.json`。上述 probe 阶段记录仍保留当时的验收边界。

当前正常入口、anchor xy 捕获和三种基础配对策略均已接入，并有独立 CPU fixture 阶段记录。CPU 合成匹配、优化或后端替身不能替代实际模型与 GPU 验证。

后来在提交 `6203956b61209f93b72be450c207dcb3cf198465` 的正常入口执行了独立两图 smoke：`outputs/scene_01_smoke/20261010T070434Z_87a89eca26d2`。新记录 `benchmarks/scene_01_smoke/normal_entry_gpu_smoke_verified.json` 仅验收这一次 complete 两图运行的真实 T4/cuda:0/fp32 闭环、构造时 anchor xy 捕获、来源/掩码/颜色/相机对应、导出读回与临时资源清理。两图共 19,848 个观测行均完成颜色对应核对，重复 xy 保留；这不是唯一物理点数或几何质量证明。

原始运行报告、预检和旧阶段记录保留原有 false；本次结论写入独立记录。实际前向次数和重建置信度仍 unavailable。全部 GPU 策略、完整 M2、新实例/重复冷跑门禁、几何质量、尺度和 Golden 冻结仍未验收；来源与阶段说明见 `PIPELINE.md`、`RESULTS.md` 和 `PAIRING.md`。

## 输入边界

`reconlab.backends.sparse_io.prepare_vendor_inputs(batch, pair_graph)`：

- 保持 manifest 的图像 ID、位置及当前策略的固定双向配对顺序；支持 complete、sequential_window 和 explicit，complete 的历史顺序保持兼容。
- 校验输入为 CPU FP32、`1,3,H,W`、有限且在 `[-1,1]` 内。
- 新建上游字典及 int32 `true_shape` 数组，像素张量按只读约定借用。
- `idx` 等于 manifest 位置，`instance` 使用解析后的完整文件路径。
- 字典修改不影响自有 ImageRecord；这不意味着像素张量也作了深拷贝。

固定源码 `convert_dust3r_pairs_naming` 会原地把 `instance` 改为文件路径。
输入检查只从 SHA256 已核对的源码中提取并执行这一函数，
不导入 MASt3R 模型。新输入在该转换后与原入口逐项一致。

## 结果提取边界

`extract_sparse_scene(scene, inputs)` 在上游临时工作目录关闭前执行，
返回自有 `SparseSceneData`，包含：

- manifest 顺序的图像 ID 与每张图的世界点观测和对应 RGB 浮点颜色。
- `K_processed`：`N,3,3`，处理后图像像素坐标的内参。
- `T_world_from_camera`：`N,4,4`，上游相机到世界的变换。
- 正常 adapter 在原 `SparseGA` 构造时捕获真实 anchor processed xy，以逐图 `pixels_xy` 的自有 CPU 副本传入结果提取；与点/颜色保持相同观测行顺序，重复 xy 保留。历史或合成输入未提供 xy 时仍为 unavailable，不能补猜像素坐标。

提取会检查返回文件顺序、每张图点/颜色数量、形状、有限值和正焦距。
张量 detach 后复制为 CPU numpy 数组，结果不依赖上游对象或缓存的存活。
这些数组仍可修改；dataclass 的 frozen 只约束字段重新赋值。

点已经由固定 `make_pts3d` 的 `cam2w` 转到上游世界系，
不作额外坐标规整，也不宣称尺度为米。观测点数允许跨图重复，
不等于唯一物理点数或几何质量评分。

目前不导出完整深度图、点置信度、全场景逐 pair 匹配指标或实际模型前向次数。
这些字段在元数据中标为 `unavailable` 并说明原因，不填零。
`get_depthmaps()` 的优化锚点深度不能当作完整 `H,W` 深度图；
稠密提取需要临时缓存并可能执行额外处理，当前步骤不调用。

## 固定源码与精度边界

- `vendor/mast3r/mast3r/cloud_opt/sparse_ga.py`
  SHA256：`d811614a10fe7920fb41285c3b8b9162b38a5e1ea68823f76383a2f209d3a49f`。
- `vendor/manifest.json`
  SHA256：`cb42fe1ac995b385e8d8c4821a9bfa8470ce5bdd10ea943922436beaecc5a0d8`。

源码中的前向函数局部使用 `torch.no_grad()`；几何优化执行 backward。
后续适配不能把整个重建包在 no_grad 或 inference_mode 中。
下游 head 显式关闭 autocast，并将 token 转为 float；
运行时有效 dtype 仍须实际加载与执行后验证，当前输入检查不证明这一点。
缓存能复用一条边的双向前向结果，因此有向项数不是实际前向次数。

## 当前检查

```bash
python -m unittest discover -s tests/unit -p 'test_*.py' -v
python scripts/check_backend_io.py --config configs/scenes/scene_01.yaml
```

检查以已提交 `benchmarks/scene_01/pairing_integrated.json` 为参照，
比较图像/配置/像素/manifest/配对图及有向顺序，保存新记录到
`work/backend_io_checks/<UTC时间_随机ID>/backend_io_check.json`，不覆盖旧记录。
结果提取测试使用合成 CPU scene，不替代真实 GPU 闭环。
首次独立边界安装阶段只新增文件；本次接入阶段会修改重建入口与文档，
保持配置、vendor 和图像加载不变。安装器不加载实际 MASt3R 模型或运行重建。

## 生命周期与入口接入

`reconlab.backends.mast3r.MASt3RSparseBackend` 提供自有对象接口：

```text
describe() -> BackendCapabilities
prepare(WeightsSpec, RuntimeSpec) -> BackendFingerprint
reconstruct(ImageBatch, PairGraph, ReconstructionOptions, BackendWorkspace) -> BackendResult
close() -> None
```

prepare 再次验证本地权重 SHA256，并核对上述固定 sparse 源码和 vendor manifest。
模型与图像加载文件的 SHA256 会记录在源码指纹中；这里不声称校验了整个 vendor 树。
实际导入的三个模块必须来自对应的仓内文件，加载后设置 eval、FP32 并冻结参数，
检查模型实际浮点参数/缓冲区 dtype 和设备。CUDA 不可用时明确失败，不自动退回 CPU。
未执行 prepare 的预检查没有有效模型指纹，不会声称模型已加载或 dtype 已验证。

优化选项由语义字段显式映射为原调用的 `subsample/device/lr1/niter1/lr2/niter2/
matching_conf_thr/shared_intrinsics`，另外显式传入上游原有默认 `dtype=torch.float32`。
reconstruct 使用 torch.enable_grad 保护几何优化；外层 inference_mode 或 autocast
会明确报错。冻结模型参数不会冻结新建的相机/深度优化参数。

workspace 声明允许写入目录、运行目录、缓存命名空间和可选事件回调。
每次调用分配新的临时目录，成功或失败后清理，不跨运行复用；
优化 stdout 保存为新日志文件，旧日志不覆盖。提取在缓存关闭前完成，
返回的点、颜色和相机是自有 CPU 副本，入口在 close 后导出 PLY 与预览图。
close 清除模型引用并释放可释放的 CUDA allocator 缓存，允许重复调用。

实际入口新增三个文件：

- `backend_fingerprint.json`：源码、权重、设备、版本及实际模型参数状态。
- `backend_result.json`：各图观测数、坐标与 dtype、损失、计时和清理记录。
- `backend_events.jsonl`：关联 run/config 的准备、对齐、提取、清理、关闭阶段事件。

encoder/decoder 内部逐操作 dtype 尚未插桩，明确为 unavailable。
下游 head 关闭 autocast/float token 是固定源码策略；优化输出 dtype 则实际检查。
计时只覆盖对齐调用及设备同步，排除日志保存/打印、CPU 提取、模型加载和导出。
先前入口的测量包含日志保存/打印开销，单次耗时变化不能作为性能验收。

入口预检查使用同一个 `_prepare_backend_inputs` 边界，并与上一阶段保存的
后端输入检查及已提交配对基线比较。它检查实际图片和权重，不导入 MASt3R：

```bash
python scripts/check_backend_entry.py --config configs/scenes/scene_01.yaml --reference work/backend_io_checks/<check_id>/backend_io_check.json
```

结果写入新的 run 目录中的 `backend_entry_check.json`。
预检查通过不代表模型导入、实际 dtype、真实场景结果提取或几何质量验收通过。
## 独立配对 probe

`probe_pair(image_a, image_b, ProbeOptions, BackendWorkspace) -> PairObservation`
复用已 prepare 的 FP32 冻结模型，只调用固定 `symmetric_inference` 和
`extract_correspondences`，不调用场景优化器，不读写上游缓存。
仅 probe 内使用 no_grad，重建仍保留 enable_grad 和几何 backward。

新增固定匹配源码：`vendor/mast3r/mast3r/fast_nn.py`，SHA256 为
`2997b5041853b9ff8d072ef2c0f533a30630cd1ef960c1a3f5b293e6f6870d90`。
调用前检查准备模型时的源码指纹、实际导入文件及匹配函数绑定。
prepare 的原有三文件指纹保持原定义，probe 另记录四文件来源。

结果保存原始、已合并去重的互为最近邻匹配 `xy_a[M,2]`、`xy_b[M,2]`、
`descriptor_confidence[M]` 与独立 `keep_mask[M]`，均为自有 CPU numpy 副本。
坐标是各图处理后的整数像素中心 x,y，尺寸用 H,W；支持两图尺寸不同。
原 manifest 索引保留在观测记录；仅新建的临时上游输入使用 idx=0,1。
稳定 pair_id 与完整场景配对图关联，坐标方向按传入的 image_a/image_b 保留。

描述子置信度为固定源码计算的 `sqrt(desc_conf_a * desc_conf_b)`，
同一坐标对重复时由上游去重索引选择一个观测，不作最大值或均值汇总。
原始分数未经概率校准，可能大于 1。此分数不等于几何置信度或内点正确率。
原始数量是互为最近邻收敛和方向合并后的数量；此前候选数量与几何内点数量不可用。

ProbeOptions 显式传入 subsample；可选描述子置信度下限使用 >=，
可选 border_pixels 对两图分别按自身 H,W 去除边缘。默认下限 None、边缘 0，
保留全部已合并匹配；不会继承优化器的 matching_conf_thr。
空匹配是可记录的诊断结果，统计使用 null；不视作几何质量通过。
输出保留原数组和同索引掩码，调用方不会丢失被过滤观测。

观测明确标为 independent_pair_probe，耗时和结果不计入正常重建。
记录一次 symmetric_inference 调用；不把它称作一次 model.forward，
内部实际前向次数与整场景逐 pair 指标仍不可用。
计时分开记录推理与匹配调用及设备同步，排除模型加载、校验、CPU 拷贝和文件保存。
显存为 probe 前重置的 PyTorch allocator 峰值，包含驻留模型，不代表总显存。

只读模型源码、不导入模型的实际图片预检查：

```bash
python scripts/check_backend_probe.py --config configs/scenes/scene_01.yaml --image-a 01.png --image-b 02.png
```

它核对已提交后端基线 SHA256、实际图片/权重/配置/配对图、输入身份和匹配源码，
写入新的 run 目录中的 backend_probe_preflight.json。
CPU 合成测试不替代真实 MASt3R/GPU 匹配；本次安装不执行 GPU、不重建、不提交代码。

## 独立 GPU probe 验证入口

`scripts/run_pair_probe.py --config <配置> --reference <成功预检查记录>`
使用预检查中明确选定的两张图和 ProbeOptions，不重新选择图片或继承优化器阈值。
它先重新核对实际资源、已提交后端基线、CPU 记录、图像/配对身份与四文件匹配源码，
再 prepare 一次本地 FP32 冻结模型。当前验收要求 CUDA，设备不可用时明确失败。
实际模型指纹须与已提交稀疏重建基线完全相同；不把预检查当作模型验证。

只执行独立匹配，不调用七图场景优化器。成功或失败均 finally close，
成功时先关闭后端再用自有 CPU 数组导出。实际调用前后比较全部输入图的元数据、
CPU 张量与 RGB 字节，检查借用输入没有被修改。

每次使用新 run 目录，保存：

- `backend_fingerprint.json` 和 `probe_events.jsonl`：准备、probe、关闭及 run/config 关联。
- `pair_matches.npz`：无需 pickle 的原始 xy_a/xy_b、描述子置信度和 keep_mask。
- `pair_matches.png`：处理后 RGB 原尺寸并排；上边距 40、图间距 16 像素。
- `pair_observation.json`：坐标、计时、来源、数组/叠加图哈希和读回检查。
- `pair_probe_result.json`：CPU/已提交基线关联与本次实际 GPU 验证结果。

叠加图最多均匀选取 100 条保留匹配，记录所选原始行号及两图显示平移；
不 resize，不在原图坐标中画处理坐标。完整匹配始终保存在 NPZ。
线条是原始互为最近邻匹配，不表示经过几何内点验证。查看叠加图仍是独立验收步骤。
所有产物保存后读回；旧文件不覆盖。空结果会保存诊断产物，然后明确未通过非空真实匹配检查。
失败时保留 `probe_error.log` 和失败阶段状态，不生成成功摘要。
单次独立 probe 通过不代表完整后端最低验收、整场景匹配指标或几何质量通过。

## 独立 probe 阶段记录

2026-10-09 的实际 GPU run 为 `20261009T022343Z_351c800029c4`，
选定 `01.png / 02.png`，处理后 H,W 均为 `384,512`。
实际原始/保留匹配数均为 9924，叠加图显示其中 100 条采样。
用户反馈“目前我人工查看下，是没啥问题的”，仅作为显示采样对应大致合理的视觉接受。
该反馈不证明全部匹配为几何内点，不证明重建尺度或几何质量。

```bash
python scripts/record_backend_probe.py \
  --evidence work/backend_probe_evidence_20261009T024504Z_e689f59f.json \
  --visual-outcome accepted --visual-reviewer user \
  --visual-note '目前我人工查看下，是没啥问题的'
```

记录工具只导入 numpy/Pillow，不导入 torch 或 vendor、不重新加载权重、不重新处理图像。
它核对 CPU/GPU JSON 与文件哈希、Git 中已提交的后端基线、run/image/pair 身份、
模型与匹配源码、实际 NPZ 坐标/唯一性/掩码/统计/dtype、PNG 尺寸与显示行号、
原 GPU 计时/allocator 峰值和准备到关闭的事件关联。
自有源码哈希来自 GPU 执行后单独采集的 snapshot，不声称 GPU run 自己记录了这些哈希。
九个代码/测试文件必须与该 snapshot 一致；阶段文档可以更新，并另记录当前哈希。

旧 GPU 结果里的 `visual_correspondence_quality_accepted=false` 保留，
新的视觉反馈单独写入阶段记录。`--check-only` 只校验、不创建记录。
任何证据错配会失败；既有 run 文件和阶段记录均不覆盖。
`full_scene_pair_metrics_verified / minimum_backend_acceptance_completed /
geometry_quality_accepted` 仍为 false。本阶段不运行推理或重建、不提交代码。
