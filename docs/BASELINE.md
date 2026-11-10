# 早期七图开发基线（历史记录）

该历史运行入口：`python scripts/validate_scene_01.py`

数据：`data/scene_01/` 中的7张图片，用户确认自行拍摄。
设置：加载尺寸512，完整配对，粗对齐与细化各100步。
验证报告与环境快照：`benchmarks/scene_01/`。

已检查：重建执行、优化损失有限性、点云与相机参数有限性、
PLY导出及读回一致性。已进行初步目视检查，质量验收标准待完善。

环境使用CloudStudio预装包；环境快照用于记录版本，
尚不能作为独立安装清单。
权重放在本地`checkpoints/`，其SHA256记录于验证报告。
运行产物和缓存放在`work/`。

以上保留早期七图开发运行的记录，不作为后来正常入口或 anchor xy 的验收。当前配置驱动入口与使用方式见 `RUNNING.md`。

新增两图正常入口真实 GPU smoke 的限定验收见 `benchmarks/scene_01_smoke/normal_entry_gpu_smoke_verified.json`，执行提交为 `6203956b61209f93b72be450c207dcb3cf198465`，run 为 `outputs/scene_01_smoke/20261010T070434Z_87a89eca26d2`。本次仅验收该两图 complete 运行的 GPU 闭环、来源对应和导出/清理；没有重跑本页七图运行，也没有把旧产物移作新代码证明。完整 M2、几何质量、尺度、环境复现/重复冷跑及 Golden 冻结仍未验收。
