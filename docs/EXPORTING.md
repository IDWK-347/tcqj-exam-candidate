# 标准产物 exporter（M2 第二步）

`reconlab.exporter.export_result(result, export_config, run_dir) -> ArtifactIndex` 是 CPU 模块，不推理、不加载权重、不调参、不访问模型缓存。调用方先创建运行目录，并传入已经标准化、共用掩码过滤后的 `ReconstructionResult`。

默认 export_config 为 `{}`，也支持 `{"coordinate_dtype": "float64"}`。本步验证 float64 坐标导出；其他选项会拒绝。标准结果的点必须非空、有限，颜色位于 `[0,1]`；相机内参、刚体姿态、参考姿态、来源索引与原视图掩码须一致。Exporter 不再过滤或截断字段，输入损坏直接报错。

全部产物先写入运行目录下独立临时目录，读回核对后将整个目录原子重命名为 `artifacts/`。同一运行目录用排他 `.export.lock` 防止本 exporter 的并发写入；已有 artifacts 或已有锁会拒绝。失败时删除本次临时目录及本次创建的锁，不删除已有运行文件。该保证不等同跨进程非协作写入或断电恢复协议。

产物为：

- `artifacts/scene.ply`：little-endian 二进制 PLY，xyz 为 double，RGB 为 uint8。颜色从标准结果按最近偶数舍入，单通道最大量化误差 `0.5/255`；坐标不缩放、不裁剪、不作 viewer 翻转。
- `artifacts/cameras.json`：清单顺序下的图像 ID、processed H/W 和 K、camera-to-world 及其逆矩阵、第一台参考相机 ID、右手相机轴/列向量约定、共同世界变换、`scale_status: unverified`。
- `artifacts/point_sources.npz`：每个 PLY 行的来源图像索引和原视图观测行号、清单图像 ID、各原视图掩码。已提供的处理后像素坐标和命名置信度同序保存；不可用字段不写伪值。
- `artifacts/artifacts.json`：三个产物的路径、字节数和 SHA256，数量/范围/RGB/相机/来源读回结果及标准结果元数据。索引自身的 SHA256 由返回的 ArtifactIndex 单独提供，避免自引用。

此前供组件 checker 使用的旧 GPU 结果缺少 anchors xy 和重建置信度；观测行号不是像素，不把独立 probe confidence 混入重建。`scripts/check_exporter.py` 只读取上一阶段已通过的标准化 NPZ，在新的 `work/exporter_checks/<UTC+ID>/` 中导出、检查并记录证据；不重新标准化、不重跑 GPU、不覆盖旧产物。

这一阶段验证 exporter 独立组件，当时正常重建入口尚未接入。随后正常入口已接入，并在提交 `6203956b61209f93b72be450c207dcb3cf198465` 的两图真实 GPU smoke 中通过导出读回及来源行/掩码/xy/颜色/相机核对，见 `benchmarks/scene_01_smoke/normal_entry_gpu_smoke_verified.json`。该 run 的 `point_sources.npz` 保存了真实构造捕获并经同一掩码处理的 xy，重建置信度仍 unavailable；重复 xy 不去重。旧组件与运行报告保留原有边界和 false。完整 M2、几何质量、尺度、资源预算/跨运行缓存/重复性及 Golden 发布验收仍未通过。
