# 多视角三维重建工程｜技术岗考核

**时长 6 小时。允许使用任何 AI 工具。**

---

## 你要交什么

在分支 `submission/<你的名字>` 上提交：

- `ENVIRONMENT.md` — 环境是怎么坏的，你怎么判断的，怎么修的
- `BASELINE.md` — scene_a 跑通后的观测记录
- `PREDICTIONS.md` — 每次关键实验之前的预测
- `DIAGNOSIS.md` — scene_b 的病因分析、实验、排除过程
- `EXPERIMENTS.md` — subsample / matching_conf_thr 的对照表
- `SUBMISSION.md` — 复现步骤、你做了什么、还有什么没解决

只交结论不交 run 的，对应项不得分。

---

## 环境

工作区里的工程**现在跑不起来**。这是题目的一部分。

- 机器：Tesla T4 16GB，CUDA 可用
- 权重：`checkpoints/` 已预置，**不要联网下载**
- 图：`data/scene_a/`（2 图）、`data/scene_b/`（7 图）、`data/scene_c/`（3 图）
- Python：工程要求 **3.11**（系统默认的 `python3` 就是 3.11.1）
- 注意：`pip install` 装到哪个解释器，取决于你当时有没有激活 venv。不在 venv 里装，在当前 venv 里 import 不到
- 入口：`python scripts/run_reconstruction.py --config configs/scenes/<场景>.yaml`

### 允许
- 任何 AI 对话工具、查公开文档、查 MASt3R/DUSt3R 上游源码
- 重建 venv、改 requirements、改任何代码、随时重启环境

### 不允许
- 修改 `data/` 下的图片或 manifest 里声明的 sha256
- 修改 `checkpoints/` 里的权重
- 删掉 `scene_b` 的图让它"跑通"
- 在报告里写你没跑过的数字

> 关于最后一条：写「我没能跑通，因为 X」只丢修复效果的分；**编造数字则全报告作废。**

---

## 四个任务

### 一、让环境能跑
让这条命令成功，然后再真跑一次：

```bash
python scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --dry-run
```

写 `ENVIRONMENT.md`：第一次的确切报错、你判断问题在哪一层的依据、你动了什么没动什么、修完怎么验证。

> 把 `.venv` 整个删掉重装能跑通，但只能拿四分之一的分。我们要看你是不是知道坏在哪。

### 二、建立基线
真跑 `scene_a`，读它的运行目录，填 `BASELINE.md`：run ID、点数、相机数、耗时、峰值显存、实际生效的配置、配置来源、配对边的条数。

再回答：`source_observation_index` 是像素坐标吗？`scale_status` 是什么值、意味着什么？`point_confidence` 是什么状态、为什么？

### 三、诊断 scene_b
`scene_b` **能跑完**，但结果有问题。

1. 列出至少 3 个可能原因，每个写成「如果是 H，那么应该观察到 E」
2. 为每个假设设计一个能区分它的实验，**必须真跑**
3. 跑之前先在 `PREDICTIONS.md` 写下预期
4. 跑完对账
5. 给结论：根因在哪个文件哪一行，怎么修，拿什么证明
6. 回归：改完 `scene_a` 还正常吗

> 这一题有很多分与「你是否答对」无关。答错但方法对，照样拿分。不要因为没找到根因就放弃写过程。

### 四、策略与成本
`configs/base.yaml` 里有两个旋钮：`backend.subsample` 和 `backend.matching_conf_thr`。

1. 分别测它们的影响，**一次只改一个**
2. 每次记：耗时、峰值显存、点数
3. 说明你用什么判断「变好了」，以及这个判断能证明什么、不能证明什么

### 五、迁移与交付
用最终方案跑 `scene_c`（3 张）。**不要为它调参。**

写 `SUBMISSION.md`：从干净环境开始的完整命令、你改了什么、方案在什么条件下失效。

---

## 15:00 之后

提交后有 15 分钟交流，会问你：抽一个数字问它怎么来的、抽一个排除掉的假设问怎么排除的、问你的方案边界。

**答案必须来自你自己跑过的东西。**

---

## 一句实话

「报错粘给 AI、拿回照做、不报错就交」大概能拿 35 分。
「先想我应该看到什么，再跑，再对账」——即使最后没修好，也能拿 70 分以上。

我们要的是后者。
