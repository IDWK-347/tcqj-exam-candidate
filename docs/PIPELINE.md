# Pipeline 正常入口接入（M2 第四步）

`ReconstructionPipeline(project_root, dry_run=False).run(resolved_config) -> RunManifest` 已在代码层连接正常 CLI，业务编排位于自有 reconstruction_pipeline.py。PipelineRun 创建唯一的 `<output_root>/<scene_id>/<UTC+随机ID>/`，记录阶段和资源；不套全局 no_grad 或 inference_mode。

正常阶段：configuration → asset_validation → image_loading → pairing → backend_prepare → backend_reconstruct → backend_close → normalize → export → report。CLI 先解析配置，只有 pipeline 创建 run 目录。dry-run 只执行前两个阶段，不处理图像、不实例化后端。配置解析失败尚无可信输出目录时，CLI 返回错误，不声称已有最终清单；创建目录/初始检查点失败也不能保证落盘。

prepare_run 和 _prepare_backend_inputs 保留给既有检查与独立 probe；正常 CLI 复用相同资源校验、图像/配对桥和参数映射。结果标准化和导出复用已通过组件；最终产物为 artifacts/scene.ply、cameras.json、point_sources.npz、artifacts.json。清单存导出索引，报告阶段索引不可变证据 SHA256/字节数；清单不自哈希。旧运行和 baseline 脚本不改写，新入口报告结构见 RUNNING.md。

每次使用 `with run` 和按既定顺序的 `with run.stage(name)` 执行阶段。阶段记录 pending/running/succeeded/failed、UTC 起止时间、错误类型和信息；禁止乱序、重复和嵌套阶段。所有已声明阶段成功且资源关闭成功时，最终状态才为 succeeded。未完成阶段、中断或业务错误都会使本次运行 failed，后续阶段保留 pending。

获取 backend 对象后、prepare 前立即登记 register_resource(name, close)。成功重建在 backend_close 阶段调用 close_resource(name)，然后才进入 CPU 标准化和导出。退出时逆序关闭剩余回调；已经关闭或关闭失败的回调不重试。清理错误独立记录，不掩盖业务错误；只有关闭失败或被吞掉的关闭异常时也必须 failed。backend_closed 事件只在 close 返回后写出。CPU 合成后端不能证明实际 GPU 资源释放。

最终清单为 `run_manifest.json`，状态摘要为 `run_status.json`，阶段变化时各文件采用临时写入和原子替换。失败也保留最终清单；检查点写入暂时失败且存储恢复后，会写明失败阶段。存储持续不可写时，记录对象保留内存诊断并传播原始错误，不能承诺落盘成功。修复后重跑使用新 ID，不覆盖旧运行。

`scripts/check_pipeline.py` 使用已通过的标准结果与 exporter 证据，走四个真实 CPU 后处理阶段：reference_validation → load_standard_result → export → report。它直接读取已存 NPZ，在新的 work/pipeline_checks 中导出并核验，随后检查最终清单和阶段时间顺序。不调用图像加载、配对或原生后端，也不重做 GPU 重建；该运行的资源列表为空。

上述旧 checker 报告中的“入口未接入”等历史 false 保持原样；本步不重复 replay。

新 scripts/check_normal_pipeline.py 使用合成 CPU 后端、两张合成 PNG 与不可加载为模型的 checkpoint，实际执行资源校验、图像处理、配对、标准化、导出和报告，并检查三点/两相机的参考变换、来源行、提前关闭及损坏资源的失败清单。证据保存在全新 work/normal_pipeline_checks；不读取旧 GPU 产物、不加载 MASt3R 模型。记录实际预处理源码哈希，区别本地 vendor 测试副本和远端完整源码。

正常后端在每次调用的私有函数 globals 副本中捕获 SparseGA 构造时的真实 anchor xy。原代码、参数、默认值和其他 helper 保持原样，vendor 文件及模块 globals 不修改；不读取私有缓存。检查图像与 anchor 键顺序、齐次整数坐标、边界、逐图点数和对应颜色后，将 xy 复制为自有 CPU 数组。重复 xy 保留；错误或缺失捕获导致重建失败。返回场景仅弱引用，提取后清理临时缓存，再关闭后端并进入标准化。

正常入口将每图 pixels_xy 显式传给标准化，所有观测字段使用同一 mask，导出至 artifacts/point_sources.npz 的 source_pixels_xy。旧 fixture 或历史数据缺失 xy 时保持 unavailable，部分图像缺失时拒绝。来源依据写入 backend_result.json 与 reconstruction.json；观测行号不能替代 xy，独立 probe confidence 不能替代重建置信度。

scripts/check_sparse_pixels.py 验证变化后的正常入口：固定来源 SHA，执行原 sparse_global_alignment 函数代码及 SparseGA/几何函数；forward、canonical、condense、optimizer 和运动链 helper 为明确合成替身，prepare 注入非模型对象。实际自有后端 reconstruct/close、标准化与导出执行，检查两图五观测行（含重复 xy）、逐行来源、缓存清理、关闭及置信度 unavailable。新增证据保存到 work/sparse_pixel_checks，不读旧 GPU 输出。该 CPU harness 不验证实际匹配、优化、模型加载或 GPU 来源。

新入口和 xy 接入的 CPU fixture 检查不代替真实 GPU 验收。随后，提交 `6203956b61209f93b72be450c207dcb3cf198465` 的两图正常入口 smoke 在真实 Tesla T4/cuda:0/fp32 上完成全部十阶段；记录为 `benchmarks/scene_01_smoke/normal_entry_gpu_smoke_verified.json`，run 为 `outputs/scene_01_smoke/20261010T070434Z_87a89eca26d2`。

本次验收范围是这一 run 的 GPU 闭环、构造时真实 anchor xy 捕获及导出来源行/颜色/掩码/相机对应、读回和临时缓存/后端清理。19,848 个观测行全部完成颜色对应核对，重复 xy 保留。结论依赖固定源码的实际构造捕获与已核对的导出链；未持久化 live anchors 供第二套独立提取。CPU 文件核对复用了固定图像预处理，未再次运行模型或优化器。

原始运行文件及历史 CPU/GPU 报告保留原有 false，本次验收另建记录。一次 complete 两图 smoke 不验收全部 GPU 策略、完整 M2、新实例/重复冷跑门禁、几何质量、尺度或 Golden 冻结；前向次数、重建置信度和全场景逐 pair 指标仍 unavailable。

基础配对已接入 complete、sequential_window 与 explicit。preflight 从当前 manifest IDs、声明和有效策略生成预期图指纹及有向项数，图像加载后再次核对；不对所有策略套用 complete 项数。顺序窗口须明确 capture_sequence，显式边只按已声明 ID 生成；缺少声明、非法边或图断开在 asset_validation 阶段拒绝并保留失败清单，未进入 backend_prepare。

scripts/check_pairing_strategies.py 在全新目录建立四图 CPU fixture，执行四种配置的正常入口及两种拒绝路径。图像加载、配对、状态、标准化及导出实际执行，backend.prepare/reconstruct 为明确合成替身。真实匹配、优化、策略质量与实际 GPU 逐点来源不因此通过。现有 scene_01 图片、配置与 manifest 不修改，不补写顺序语义；旧 GPU 和 CPU 记录保留。
