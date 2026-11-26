# SUBMISSION.md — 交付说明

> 本文件说明：怎么复现、改了什么、方案在什么条件下会失效。
> 所有 run 相关数字与结论以仓库内的 run 产物为准；本文件不替代 `BASELINE.md`、
> `DIAGNOSIS.md`、`EXPERIMENTS.md`，只在它们之上给出索引与交付边界。

---

## 1. 交付概览

| 交付物 | 文件 | 状态 |
|---|---|---|
| 环境配置记录 | `ENVIRONMENT.md` | ✅ 权重校验、dry-run、CUDA 库混用问题与处置均已记录 |
| scene_a 基线 | `BASELINE.md` | ✅ `20261126T042248Z_3d7558c62c88`，10 阶段 succeeded |
| 实验前预测 | `PREDICTIONS.md` | P0–P7 已对账（P2 崩溃假设未触发；P4 scene_c 已执行） |
| scene_b 诊断 | `DIAGNOSIS.md` | ✅ B0 + H1/H2/H3 三个实验 + 根因/修复/回归已完成 |
| 参数实验 | `EXPERIMENTS.md` | ✅ E0–E4 已完成（scene_a 单变量，run ids 见 §4.8） |
| 本文件 | `SUBMISSION.md` | 已补关键 run 索引与改动说明 |
| 改动过的代码 | 见第 3 节 | 配置：`assets/weights.json`、`configs/scenes/scene_b.yaml` |
| scene_c 产物 | `outputs/scene_c/20261126T083005Z_865eb24fedb1/` | ✅ 3 相机 / 55,580 点 / 7.892 s，10 阶段 succeeded |

### 1.1 考核任务 → 交付位置 → 证据 run

| README 任务 | 交付位置 | 证据 run |
|---|---|---|
| 1 环境配置 | `ENVIRONMENT.md` §1–§6（§3.3 为完整运行验证） | dry-run `20261126T035611Z_077779e0fdea`；完整运行 `20261126T042248Z_3d7558c62c88` |
| 2 建立基线 + 三个必答题 | `BASELINE.md` §1–§4 | `20261126T042248Z_3d7558c62c88` |
| 3 诊断 scene_b | `DIAGNOSIS.md` §1–§6；预测与对账见 `PREDICTIONS.md` P2、P5–P7 | B0 `20261126T042744Z_ffdbbf471afb`；完整配对 `20261126T043428Z_87ba5af92253`；subsample=8 `20261126T044206Z_4b4ac7dfd49d`；thr=5.0 `20261126T044458Z_d7fb4feff3c9`；修复 `20261126T044924Z_ff9b3efbfe8e`；回归 `20261126T045015Z_65797318eef8` |
| 4 策略与成本 | `EXPERIMENTS.md` §1–§5 | E0–E4（完整 run_id 见 §4.8）；跨场景阈值效果见 §4.3 |
| 5 迁移与交付 | 本文件 | scene_c `20261126T083005Z_865eb24fedb1` |

> 本表是阅读入口；各文件中每个数字都可指回对应 run 的产物。所有被引用的 run 目录
> 都随提交提供（§8.1），因此结论可按下表与 §4.8 逐条复核。

---

## 2. 环境

- 系统 Python 3.11.1（`/root/.pyenv/versions/3.11.1/bin/python3`），**未使用**仓库自带 `.venv`；
- torch 2.10.0+cu128，`cuda.is_available() = True`，Tesla T4 16 GB；
- 依赖按 `requirements.txt` 安装；`roma` 用 `pip install --no-deps` 补装；
- **所有运行命令必须加 `LD_LIBRARY_PATH=` 前缀**。默认 shell 的
  `LD_LIBRARY_PATH` 含 `/usr/local/cuda-12.2/lib64`，会让 pip 的 cuBLAS 12.8
  加载 12.2 的 `libcublasLt`，在匹配阶段报 `CUBLAS_STATUS_NOT_INITIALIZED`。
  定位过程与证据见 `ENVIRONMENT.md` §4.5；
- 全部细节与遇到的问题见 `ENVIRONMENT.md`。

不激活 `.venv` 的原因：`.venv/bin/python` 是悬空软链，指向 `/workspace/OLD-INSTANCE/.venv/bin/python`，
site-packages 只有 pip/setuptools，无法启动；重建成本高于收益。详见 `ENVIRONMENT.md` §4.1。

---

## 3. 改动清单

### 3.0 须知“开始前”三件事的落实

| 须知条目 | 要求 | 本工程状态 | 证据 |
|---|---|---|---|
| 1. 克隆并建分支 | 从 `main` 拉出 `submission/<名字>` | ✅ 当前分支 `submission/peixinnan`，与 `main` 同一提交 `ac0ba43` | `git branch -vv`（本地工作区即克隆） |
| 2. 下载并校验权重 | 2754910614 字节；SHA256 `e28f91…568eb2`；放入 `checkpoints/` | ✅ 两项同时吻合 | `ENVIRONMENT.md` §3 |
| 3. 改 `assets/weights.json` 的路径 | 须知示例写的是绝对路径 | ⚠️ 按代码契约改为**相对路径**，解析结果仍是 `/workspace/checkpoints/…`；`sha256` 字段未改 | `entry.py:50-52`；`ENVIRONMENT.md` §4.2 |

> 第 3 条是须知与代码冲突之处：`entry.py:50-52` 无条件拒绝绝对路径，照抄须知示例会让
> 每次运行在 `asset_validation` 直接失败。这里以**代码实际行为**为准；冲突与代码依据
> 写明在 `ENVIRONMENT.md` §4.2。须知同条明确要求 `sha256` 保持原样，该字段未改。

| 文件 | 改动 | 原因 |
|---|---|---|
| `assets/weights.json` | `file` 从绝对路径改为 `../checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` | `entry.py:50-52` 无条件拒绝绝对路径；原值 `/workspace/OLD-INSTANCE/...` 在本机不存在 |
| `configs/scenes/scene_b.yaml` | `backend.subsample: 4` → `8` | 诊断结论：该覆写把 7 图输出从 117,830 个观测放大到 354,140，重建耗时从 14.614 s 增加到 24.702 s；改回 base 值。证据见 `DIAGNOSIS.md` §4 |
| `tools/inspect_run.py` | 新增只读 run 汇总脚本 | 便于从产物中取数；放在 `src/` 之外，不进入 `implementation_sources` 指纹 |
| `tools/measure_epipolar.py` | 新增只读对极一致性复算脚本 | 固定 SIFT/RANSAC 内点、像素 `F = K₂⁻ᵀEK₁⁻¹`；`--preprocess-affine` 复算含亚像素平移的口径；`--legacy` 复现旧算法值 36.970 px（仅用于说明历史数值，不作指标） |
| 其他 md | 新增/补写报告 | 交付要求 |

**未改动**（硬规则）：

- `data/` 下任何图片与 `manifest.json` 的 sha256；
- `checkpoints/` 下权重；
- `vendor/` 下任何源码（哈希被 `mast3r.py` 固定校验）。

**代码改动结论**：`src/` 与 `vendor/` **均未改动**（`git status --short src vendor` 为空；
`git diff --stat src vendor` 也为空）。本次全部改动是配置与文档：
`assets/weights.json`（路径改为相对）、`configs/scenes/scene_b.yaml`（`subsample` 4 → 8），
以及新增的复算脚本和报告。scene_b 根因、文件行号与修复理由见 `DIAGNOSIS.md` §4。

### 3.1 交付文件状态（本次提交的内容）

| 状态 | 文件 |
|---|---|
| 已修改（已提交） | `assets/weights.json`、`configs/scenes/scene_b.yaml` |
| 新增（已提交） | 六份 md（`ENVIRONMENT.md`、`BASELINE.md`、`PREDICTIONS.md`、`DIAGNOSIS.md`、`EXPERIMENTS.md`、`SUBMISSION.md`）、`tools/inspect_run.py`、`tools/measure_epipolar.py`、`outputs/scene_a/20261126T*`（10 个 run）、`outputs/scene_b/`（5 个 run）、`outputs/scene_c/20261126T083005Z_865eb24fedb1/` |
| 未修改（保持原样） | `data/`、`vendor/`、`src/`、`scripts/`、`outputs/scene_a/20261010T205345Z_230967b3a998/`（外来历史 run，未采用） |
| gitignored、未修改 | `checkpoints/`（SHA256 与期望一致） |
| 不提交 | `.cargo/`（本地目录）、`CLAUDE.md`（内部交接笔记）、`kaoshengxuzhi.md`（须知副本）；范围见 §8.2 |

**两个新增工具**：

- [`tools/inspect_run.py`](tools/inspect_run.py)：只读汇总 run 的阶段耗时、配对、点数、损失、显存；
- [`tools/measure_epipolar.py`](tools/measure_epipolar.py)：固定 SIFT/RANSAC 内点，
  按像素 `F = K₂⁻ᵀ E K₁⁻¹` 复算对极一致性；`--preprocess-affine` 用
  `image_preprocessing.json` 的仿射（含亚像素平移）替代纯比例映射；
  `--legacy` 仅复现旧算法值 36.970 px。

**`scene_c` 产物**（run `20261126T083005Z_865eb24fedb1`，3 相机 / 55,580 点 / 7.892 s）：

- [run 目录](outputs/scene_c/20261126T083005Z_865eb24fedb1/)
- [`artifacts/scene.ply`](outputs/scene_c/20261126T083005Z_865eb24fedb1/artifacts/scene.ply)（1,500,874 B）
- [`artifacts/cameras.json`](outputs/scene_c/20261126T083005Z_865eb24fedb1/artifacts/cameras.json)（5,373 B）
- [`artifacts/point_sources.npz`](outputs/scene_c/20261126T083005Z_865eb24fedb1/artifacts/point_sources.npz)（231,249 B）
- [`artifacts/artifacts.json`](outputs/scene_c/20261126T083005Z_865eb24fedb1/artifacts/artifacts.json)（3,677 B）

---

## 4. 复现命令

### 4.0 从干净环境开始（完整命令序列）

> **验证范围（如实说明）**：本节命令按仓库文件与 `ENVIRONMENT.md` 整理。
> 本机**实际完整执行过**的是“系统 Python + 环境自带 cu128 torch”这条路径
> （权重下载/校验、dry-run、全部 run 都已实际跑过）；
> **没有**在“全新 clone + 全新 venv”中做过端到端复跑，因此
> §4.0.2 的 venv 创建与 §4.0.3 的 CUDA wheel 安装标注为**未验证**。
> 其余命令均来自本项目已实际使用的写法。

#### 4.0.1 取得代码：两个起点（不要混用）

**A. 取得最终交付版本后复现（推荐）**

```bash
git clone https://github.com/Salih-Biron/tcqj-exam-candidate.git
cd tcqj-exam-candidate
git checkout submission/peixinnan
```

> 该分支必须已在远程存在，或评审环境直接提供当前工作区。
> 截至本文件修订时，本地 `submission/peixinnan` **尚未 push**；
> 若远程没有该分支，请走下面的 B 路径并应用两处配置改动。

**B. 从 `main` 基线开始搭建**

```bash
git clone https://github.com/Salih-Biron/tcqj-exam-candidate.git
cd tcqj-exam-candidate
git checkout main
git checkout -b submission/peixinnan
```

从 `main` 开始时，后续命令依赖的两个文件仍是打包时的状态，**必须先改成最终方案**，
否则无法复现本报告的结果：

- `assets/weights.json` 的 `file` 是绝对路径，入口会直接报 `EntryError`；
- `configs/scenes/scene_b.yaml` 的 `subsample: 4` 会得到 B0 的 354,140 点 / 24.702 s，
  而不是修复后的 117,830 点 / 14.614 s。

> 若要复现原始 B0：先不要执行下面的第 2 条修复，直接按 §4.5 的 B0 命令运行；
> 应用修复后，也可用 `--set backend.subsample=4` 复现旧值。

```bash
# 1) weights.json：绝对路径 -> 相对 assets/ 的路径（sha256 保持原样）
sed -i 's#"file": "/workspace/OLD-INSTANCE/checkpoints/#"file": "../checkpoints/#' assets/weights.json

# 2) scene_b 配置修复：subsample 4 -> 8
sed -i 's/^  subsample: 4$/  subsample: 8/' configs/scenes/scene_b.yaml
```

核对：

```bash
python3 -c "import json; print(json.load(open('assets/weights.json'))['file'])"
# 期望：../checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

grep -n 'subsample' configs/scenes/scene_b.yaml
# 期望：8
```

#### 4.0.2 Python 环境

本机实测：系统 Python 3.11.1，**不使用**仓库自带的坏 `.venv`
（`bin/python` 是悬空软链，site-packages 只有 pip/setuptools；见 `ENVIRONMENT.md` §4.1）。

```bash
python3 --version
# 期望 3.11.x

python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
# 本机实际：2.10.0+cu128 True，架构列表含 sm_75
```

若需要独立环境（**本报告未在全新 venv 中验证**）：

```bash
python3 -m venv .venv-fresh
. .venv-fresh/bin/activate
python3 -m pip install --upgrade pip
```

#### 4.0.3 安装依赖（注意：默认 PyPI 的 torch 是 CPU 版）

`requirements.txt` 顶部已注明 “The default PyPI torch wheel is the CPU build”。
因此顺序必须是：**先有 CUDA 版 torch/torchvision，再装其余依赖，最后补 `roma`**。

```bash
# 1) 确认 CUDA 版 torch 已就绪（本机由环境自带 2.10.0+cu128，未重新安装）
python3 -c "import torch; assert torch.cuda.is_available(); print(torch.__version__)"

# 2) 安装 requirements.txt 中的依赖
python3 -m pip install -r requirements.txt

# 3) 补装 roma：本机执行时 requirements 已装完但 roma 仍缺失
python3 -m pip install --no-deps roma
```

`--no-deps` 的具体用法与原因：它只安装 `roma` 本身，**不重新解析依赖树**。
这里用它是因为不加时 pip 可能顺带升级/替换已安装的 `torch`，
把 cu128 版换成默认索引的 CPU 版。装完必须验证：

```bash
python3 -c "import torch, roma; print(torch.__version__, torch.cuda.is_available())"
# 期望仍为 2.10.0+cu128 True
```

> 如果全新 venv 中 `pip install -r requirements.txt` 已把 torch 装成 CPU 版
> （`cuda.is_available()` 为 False），不要继续跑重建：先按本机驱动对应的 CUDA wheel
> 重装 `torch`/`torchvision`，再重复上面的验证。该重装步骤**本报告未实际执行**。

#### 4.0.4 下载并校验权重

```bash
mkdir -p checkpoints
aria2c -x 16 -s 16 -k 1M --file-allocation=none \
  -o checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  'https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
```

中断后可用同一条 `aria2c -c ...` 续传；单连接可用：

```bash
wget -c -O checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  'https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
```

校验（两个条件必须**同时**满足；`ls -l` 会把稀疏文件误判为已完成）：

```bash
stat -c %s checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
# 期望 2754910614

sha256sum checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
# 期望 e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2
```

#### 4.0.5 首次 dry-run

```bash
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --dry-run
```

通过后再进入 §4.4 的完整运行。`LD_LIBRARY_PATH=` 不能省，
原因与证据见 `ENVIRONMENT.md` §4.5（默认值会让 cuBLAS 12.8 加载 12.2 的 `libcublasLt`）。

### 4.1 环境自检

```bash
cd /workspace
python3 --version
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
python3 -c "import torch; a=torch.randn(4096,4096,device='cuda'); print((a@a).sum().item())"
python3 -c "import torch, torchvision, numpy, scipy, PIL, cv2, einops, roma, tqdm, packaging, huggingface_hub; print('deps OK')"
```

### 4.2 权重校验

```bash
stat -c %s checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
# 期望 2754910614

sha256sum checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
# 期望 e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2
```

两个条件**同时**满足才算下载完成；只看 `ls -l` 会把稀疏文件误判为已完成。

### 4.3 入口预检

```bash
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --dry-run
```

### 4.4 基线

```bash
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml
python3 tools/inspect_run.py outputs/scene_a/<run_id>
```

### 4.5 scene_b：复现原始 B0 与修复后配置

**原始 B0（历史配置 `subsample=4`）**

B0 的真实 `resolved_config.yaml` 是 `backend.subsample: 4`；其 `config_sources.json`
记录该字段最终来源为 `configs/scenes/scene_b.yaml`（raw value `4`），**没有 CLI 覆盖**。
当前交付配置已改为 `8`，因此复现旧值必须用 CLI 覆盖：

```bash
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_b.yaml \
  --set backend.subsample=4
```

> 说明：这里的 `--set` 只是**复现旧值的方式**；新 run 的 `config_sources.json`
> 会额外记录 `source: "<CLI>"`，与原 B0 的“来源=`configs/scenes/scene_b.yaml`”不同，
> 但最终生效配置相同（`subsample=4`）。预期量级：354,140 点 / 24.702 s（耗时会波动）。
>
> 本轮静态核对：对当前 `scene_b.yaml` 应用 `--set backend.subsample=4` 后，
> 配置解析结果为 `backend.subsample=4`（来源链 `base 8 → scene_b 8 → <CLI> 4`）；
> 原 B0 的来源链是 `base 8 → scene_b 4`（无 CLI）。两者最终值相同，来源记录不同。
> **未重跑重建**，所以这里是配置解析核对，不是新的 run 证据。

**修复后（当前交付配置 `subsample=8`）**

```bash
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_b.yaml
```

预期量级：117,830 点 / 14.614 s。

> 从 `main` 起步时：`main` 的 `scene_b.yaml` 仍是 `4`。未应用 §4.0.1 B 的修复前，
> 直接运行上面的“修复后”命令得到的是 B0；应用 4→8 修复后，才得到 117,830 点。
> 两种状态不要混用。

后续区分实验（H1/H2/H3）的完整命令见 `DIAGNOSIS.md` §3。

### 4.6 参数实验（一次只改一个）

```bash
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --set backend.subsample=16
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --set backend.subsample=4
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --set backend.matching_conf_thr=5.0
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --set backend.matching_conf_thr=1.5
```

### 4.7 scene_c（不为它单独调参）

```bash
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_c.yaml
```

scene_c 使用与基线相同的 base 配置，只走通用流程，不做任何 `--set`。
本次实测：run `20261126T083005Z_865eb24fedb1`，`config_sources.json` 的 `load_order`
只有 `configs/base.yaml`、`configs/t4.yaml`、`configs/scenes/scene_c.yaml`，无 CLI 覆盖；
产物在 `outputs/scene_c/20261126T083005Z_865eb24fedb1/`（3 相机 / 55,580 点 / 7.892 s）。

> 未证实的观察：该 run 的 `asset_validation` 阶段耗时 **14.49 s**，而 scene_a 基线
> （`20261126T042248Z_3d7558c62c88`）为 2.10 s、scene_b B0（`20261126T042744Z_ffdbbf471afb`）
> 为 2.13 s，三者都是同一份 2.5 GiB 权重算一遍 SHA256。数值来自
> `run_manifest.json.phases[].started_at_utc/ended_at_utc`；**耗时差异的原因没有独立记录**
> （页缓存冷/热只是推测），故保留为**未证实**，不影响任何结论。

### 4.7.1 对极一致性复算

```bash
python3 tools/measure_epipolar.py --scene scene_b \
  --run outputs/scene_b/20261126T042744Z_ffdbbf471afb \
  --run outputs/scene_b/20261126T043428Z_87ba5af92253 \
  --run outputs/scene_b/20261126T044206Z_4b4ac7dfd49d \
  --run outputs/scene_b/20261126T044458Z_d7fb4feff3c9 \
  --pairs 01-02,04-05,01-07

# 复现旧报告里的 36.970 px（旧算法：方向取反 + E 当 F；仅用于说明，不作指标）
python3 tools/measure_epipolar.py --scene scene_b \
  --run outputs/scene_b/20261126T042744Z_ffdbbf471afb --pairs 01-07 --legacy
```

> **测量口径**：脚本默认在原始 1448×1086 图像上提取 SIFT，再按纯比例映射到处理后坐标，
> 省略了预处理中的亚像素平移（`A_oriented_to_processed` 的 tx=ty=−0.3232）。
> 加 `--preprocess-affine` 可复算补上平移后的值：B0 01–07 由 2.166 变为 2.164 px，
> 12 组 run×pair 的最大变化 0.015 px。全部结论只引用默认口径的可复现值：
> B0 01–07 = **2.166 px**。
> `36.970` 属旧算法，`--legacy` 可精确复现；其他来源的复核数值因原始配方未知、
> 本轮无法逐位复现，**本报告不引用**。

### 4.8 关键 run 索引

| 用途 | run_id | 状态 | 关键值 |
|---|---|---|---|
| 入口 dry-run | `20261126T035611Z_077779e0fdea` | succeeded | 2 阶段，2 图 / 2 有向对 |
| scene_a 基线 | `20261126T042248Z_3d7558c62c88` | succeeded | 2 相机 / 14,734 点 / 重建 4.467 s |
| scene_b B0 | `20261126T042744Z_ffdbbf471afb` | succeeded | 7 相机 / 354,140 点 / 24.702 s；复现命令见 §4.5 |
| scene_b complete | `20261126T043428Z_87ba5af92253` | succeeded | 21 边 / 1,221,896 点 / 76.396 s；01–07 指标 2.166→0.517 px（仅此一对） |
| scene_b subsample=8（CLI） | `20261126T044206Z_4b4ac7dfd49d` | succeeded | 117,830 点 / 14.529 s |
| scene_b thr=5.0 | `20261126T044458Z_d7fb4feff3c9` | succeeded | 点数不变；损失目标已变 |
| scene_b 修复后（配置） | `20261126T044924Z_ff9b3efbfe8e` | succeeded | 117,830 点 / 14.614 s / 7 相机 |
| E1 scene_a subsample=16 | `20261126T045415Z_bc7519ec6124` | succeeded | 3,814 点 / 3.298 s |
| E2 scene_a subsample=4 | `20261126T045607Z_b7d39c4a2ae0` | succeeded | 53,878 点 / 4.934 s |
| E3 scene_a thr=5.0 | `20261126T045752Z_a8af27bf9b5f` | succeeded | 与基线逐位相同 |
| E4 scene_a thr=1.5 | `20261126T045956Z_817a38ed404d` | succeeded | 与基线逐位相同 |
| scene_c 迁移 | `20261126T083005Z_865eb24fedb1` | succeeded | 3 相机 / 55,580 点 / 7.892 s；无 CLI 覆盖 |
| scene_a 回归 | `20261126T045015Z_65797318eef8` | succeeded | 与基线逐位一致（点/相机/loss） |
| scene_a 失败 run（3 个） | `20261126T041729Z_4b2acf11e162`、`20261126T041937Z_581439180005`、`20261126T042110Z_7904ef813fba` | failed | `backend_reconstruct` / `CUBLAS_STATUS_NOT_INITIALIZED`；`ENVIRONMENT.md` §4.5 的证据 |

> 上表每个 run_id 都对应 `outputs/` 下真实存在的目录，并且都纳入提交范围（见 §8.1）；
> 报告中的每个数字都能在上表或对应 run 的产物中找到来源。

---

## 5. 产物与取证

每个 run 的关键文件：

| 文件 | 用途 |
|---|---|
| `run_manifest.json` | run_id、阶段耗时、失败阶段与错误 |
| `resolved_config.yaml` / `config_sources.json` | 实际生效配置、字段来源（`<CLI>` 表示被 `--set` 覆盖） |
| `pair_graph.json` | 无向边、有向对、连通分量 |
| `reconstruction.json` | 最终损失、重建耗时、allocator 峰值 |
| `reconstruction_optimizer.log` | 优化器原始日志（`>> final loss`） |
| `artifacts/artifacts.json` / `standard_result.json` | 点数、相机数、字段可用性 |
| `point_sources.npz` | `source_observation_index` 与 `source_pixels_xy` |
| `run_status.json` | 最终状态 |

**必答题的取证位置**：

1. `source_observation_index` 是“原始稀疏视图中的行号”，不是像素坐标
   （`results.py:55`）；像素坐标在 `point_sources.npz` 的 `source_pixels_xy`。
2. `scale_status` 恒为 `"unverified"`（`exporter.py:200`、`results.py:45`；
   `sparse_io.py:135` 写明 `metric_units_verified: False`）。
3. `availability.point_confidence` 恒为 `"unavailable"`（后端不读上游私有缓存提取置信度）。

---

## 6. 方案失效条件（先验判断，run 之后修订）

> 以下第 1–2 条含**运行后**修订：B0 实测峰值 3.259 GiB（16 GB T4 未 OOM）；
> `subsample=4 → 8` 使点数 354,140→117,830、重建 24.702→14.614 s。
> 其余仍为基于代码约束的先验判断。

1. **显存不足**：T4 16 GB 下，图数或锚点密度继续增大时，`backend_reconstruct`
   可能 OOM；本工程没有 CPU 回退（请求 CUDA 但不可用会直接报错）。
2. **输入规模/拓扑**：`pairing.strategy` 只支持 `complete` / `sequential_window` /
   `explicit`；`explicit` 要求 manifest 提供 `explicit_edges`，且图必须连通。
3. **视觉重叠不足**：配对图连通只证明结构连通，`visual_overlap` 恒为 `unverified`；
   重叠不足时匹配质量无法保证。
4. **分辨率固定**：`preprocess.long_edge` 只能是 `512`，不能通过改分辨率规避显存问题。
5. **精度固定**：`runtime.precision` 只能是 `fp32`，不支持 fp16/bf16。
6. **后端与 vendor 固定**：`backend.name` 只能是 `mast3r_sparse_ga`；`vendor/` 源码被
   SHA256 固定，不能通过改 vendor 代码绕过问题。
7. **尺度与几何质量**：`scale_status` 恒为 `unverified`，`geometry_quality_accepted`
   恒为 `false`；本方案不能给出度量尺度或质量通过结论。
8. **参数取值边界**：`matching_conf_thr` 必须 ≥ 0；`coarse_lr`/`refine_lr` 必须 > 0；
   `subsample` 为正整数。超出白名单的字段会直接 `ConfigError`。

---

## 7. 未采用的材料（避免误当基线）

| 材料 | 未采用原因 |
|---|---|
| `outputs/scene_a/20261010T205345Z_230967b3a998/` | 来自 `/workspace/tcqj-reconlab-golden` 的历史 run；输入图 sha256 与 `data/scene_a/` 不同，且 `resolved_config.yaml` 指向旧工程路径 |
| `docs/BASELINE.md` | 历史 `scene_01` 七图记录，不是本次模板 |
| `.venv/` | 悬空软链，无法启动 |
| `docs/` 引用的 16 个检查脚本 | 仓库中不存在，只有 `scripts/run_reconstruction.py` 可用 |

---

## 8. 提交清单

> 本节给出最终提交的确定范围；该范围已固化为本地提交（尚未 push）。

### 8.1 纳入提交

| 类别 | 路径 | 说明 |
|---|---|---|
| 配置改动 | `assets/weights.json` | `file` 改为相对 `assets/`；`sha256` 未改 |
| 配置改动 | `configs/scenes/scene_b.yaml` | `subsample: 4 → 8`（根因修复） |
| 报告 | `ENVIRONMENT.md`、`BASELINE.md`、`PREDICTIONS.md`、`DIAGNOSIS.md`、`EXPERIMENTS.md`、`SUBMISSION.md` | 六份交付物 |
| 工具 | `tools/inspect_run.py`、`tools/measure_epipolar.py` | 报告引用的只读复算工具（默认输出与报告数值一致） |
| 证据 run | `outputs/scene_a/20261126T*`（10 个） | 含 1 个 dry-run、3 个失败 run（`ENVIRONMENT.md` §4.5 证据）、基线、回归、E1–E4 |
| 证据 run | `outputs/scene_b/*`（5 个） | B0、complete、subsample CLI、thr=5.0、修复后 |
| 证据 run | `outputs/scene_c/20261126T083005Z_865eb24fedb1/` | 迁移产物（考题要求"另附 scene_c 产物"） |

合计约 71 MB；最大单文件为 scene_b `complete` run 的 `scene.ply`（31.5 MB）。
保留全部 run 的原因是：§4.8 中每个 run_id 都是某条结论的证据，删掉任何一个都会让对应结论无法复核。

### 8.2 不纳入提交

| 路径 | 原因 |
|---|---|
| `.cargo/` | 本地工具链目录，非交付物 |
| `CLAUDE.md` | 内部交接笔记（含过程性说明与已删除 run 的引用），不是交付物 |
| `kaoshengxuzhi.md` | 考场须知副本，非交付物 |
| `outputs/scene_a/20261010T205345Z_230967b3a998/` | 外来历史 run，已在 Git 中跟踪；保持原样、未采用（见 §7） |
| `checkpoints/` | `.gitignore` 已排除 |

### 8.3 提交命令（已执行；保留作为复现记录）

```bash
git add assets/weights.json configs/scenes/scene_b.yaml \
  ENVIRONMENT.md BASELINE.md PREDICTIONS.md DIAGNOSIS.md EXPERIMENTS.md SUBMISSION.md \
  tools/inspect_run.py tools/measure_epipolar.py \
  outputs/scene_a/20261126T* outputs/scene_b outputs/scene_c
git status --short   # 确认暂存区不含 .cargo/、CLAUDE.md、kaoshengxuzhi.md
```

> 注：本地提交已完成，但**尚未 push**；远程仓库中尚无这些内容。
