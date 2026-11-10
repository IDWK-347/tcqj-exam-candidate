# 标准结果边界（M2 第一步）

`reconlab.results.normalize_sparse_scene` 只接收已提取的 CPU NumPy 数据，不导入 torch、vendor，不加载模型，不访问缓存，不调用优化器。

输入相机须按图像 ID 的清单顺序对应；矩阵须有限，姿态须为容差内的正旋转刚体矩阵。采用第一台已注册相机作为参考，用同一个参考姿态逆矩阵变换全部相机和世界点。保留 processed-image 内参，采用右手相机坐标 x 右、y 下、z 前和列向量；尺度标记 `unverified`。浮点参考矩阵通过旋转容差检查后取实际逆矩阵，不另行修正旋转或修改几何。

结果为自有 CPU 数组。逐点保留 `source_image_index` 和原视图中的 `source_observation_index`。后者是原观测行号，不能解释为处理后像素坐标。参考相机即使没有保留点仍然是已注册相机，不按点数或文件名字排序更换参考。

点、RGB、可用置信度、可用像素坐标共用同一个掩码。先剔除任一关联字段非有限的行，再应用显式命名的置信度字段及 `>=` 阈值；分别记录非有限移除数量、阈值移除数量和保留比例。有限 RGB 越界、相机错误、数组数量或来源 ID 错误直接报错，不截断颜色或静默丢弃相机。全部点被过滤时为质量失败。

固定上游 `SparseGA.__init__` 使用 `anchors[i][0][..., :2]` 索引 RGB，但返回对象仅保存 points/colors，并未保存 anchors。正常 adapter 现在在构造时捕获这些真实 xy；旧 GPU 产物和旧 CPU 标准化结果仍没有经过验证的逐点像素坐标或重建置信度。未提供字段明确标记 unavailable，并保留原因；不从点数、观测行号、颜色或反投影猜测原像素。置信度不可用时不能开启阈值筛选，独立 probe 的 descriptor confidence 不能用来代替重建置信度。

`scripts/check_result_standardization.py` 使用原七图 GPU run 的已验证 PLY、相机和预处理记录，只在 CPU 做标准化、世界变换回环、相机空间对应、颜色和观测行号检查，写入新的 `work/result_standardization_checks/<UTC+ID>/`。NPZ 与 JSON 读回通过后保存证据；不重建、不覆盖原产物。原 PLY 的 uint8 RGB 不能恢复为量化前的浮点颜色。

上述 checker 是标准结果组件第一步的历史验收；当时正常入口、exporter、pipeline 和逐点来源尚未接入。当前正常入口已调用标准化、导出和状态记录，历史报告保持不变。随后独立两图真实 GPU smoke 的来源对应通过，见 `benchmarks/scene_01_smoke/normal_entry_gpu_smoke_verified.json`：限定提交 `6203956b61209f93b72be450c207dcb3cf198465` 的一个正常入口 run，19,848 行点/颜色/xy/来源掩码及两相机读回核对通过，重复 xy 保留。原始报告的 false 不改写；M2 完成、其他场景/策略、几何质量、尺度、跨运行缓存/性能基线和 Golden 发布验收仍未通过。

正常后端现可在原 SparseGA 构造时捕获真实 processed xy，自有 SparseViewData.pixels_xy 保存原观测行的整数 x,y CPU 副本。SparseSceneData.pixels_xy 为完整逐图元组，部分图像缺失时拒绝。正常入口显式传入 normalize_sparse_scene(pixels_xy=..., processed_hw=...)；同一观测 mask 同步过滤 xy、点、颜色和来源索引，重复 xy 不去重。缺失数据保持 unavailable，不能从行号、数量、颜色或投影生成 xy。重建置信度仍 unavailable。具体来源记录与实际 GPU 验收范围见 PIPELINE.md；CPU harness 不代表实际模型运行。
