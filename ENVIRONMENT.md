# ENVIRONMENT.md — 环境配置记录

> 记录本机环境、做过的配置改动、遇到的问题与处置。
> 所有结论均可在本文件"复现命令"一节中核对。
> 本文件初稿写于权重下载期间，当时权重校验与首次 `--dry-run` 尚未完成；二者已于
> 2026-11-26T03:51Z / 03:56Z 完成，结果见 §3.2 与 §6，文中未填入任何推测值。

---

## 1. 机器与运行时

| 项目 | 实测值 | 采集命令 |
|---|---|---|
| GPU | Tesla T4，15360 MiB，compute capability (7,5) = sm_75 | `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` |
| 驱动 | 580.178.04 | 同上 |
| Python | 3.11.1，`/root/.pyenv/versions/3.11.1/bin/python3` | `python3 -c "import sys; print(sys.version, sys.executable)"` |
| 虚拟环境 | 无（`VIRTUAL_ENV` 为空，`python3` 解析到 pyenv shims） | `which -a python3; echo $VIRTUAL_ENV` |
| torch | 2.10.0+cu128 | `python3 -c "import torch; print(torch.__version__, torch.version.cuda)"` |
| torchvision | 0.25.0+cu128 | `python3 -c "import torchvision; print(torchvision.__version__)"` |
| CUDA 可用 | True；编译架构列表含 `sm_75` | `python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"` |

**CUDA 冒烟测试**（4×4 千矩阵乘 + 设备同步）：

```
matmul OK: 1832.046875
dev: Tesla T4
peak MiB: 200
```

## 2. 依赖

`requirements.txt` 共 11 项，逐项导入验证结果：

| requirement | 实际版本 | 状态 |
|---|---|---|
| torch>=2.1 | 2.10.0+cu128 | OK |
| torchvision>=0.16 | 0.25.0+cu128 | OK |
| numpy | 2.3.5 | OK |
| scipy | 1.17.0 | OK |
| Pillow | 12.1.0 | OK |
| opencv-python-headless | 4.13.0 | OK |
| einops | 0.8.2 | OK |
| roma | 1.8.0 | OK（本机补装，见 4.3） |
| tqdm | 4.67.3 | OK |
| packaging | 25.0 | OK |
| huggingface-hub | 1.4.0 | OK |

**后端模块导入验证**（不需要权重）：

```
mast3r.model                  OK  -> vendor/mast3r/mast3r/model.py
mast3r.cloud_opt.sparse_ga    OK  -> vendor/mast3r/mast3r/cloud_opt/sparse_ga.py
dust3r.utils.image            OK  -> vendor/mast3r/dust3r/dust3r/utils/image.py
```

附带一条无害警告：`Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead`。
这是缺少 CUDA 扩展时的回退实现提示，不影响正确性，仅影响少量速度。

扫描 `src/` + `vendor/mast3r/` + `scripts/` 的全部导入后，另有 8 个包不可导入：
`asmk`、`faiss`、`pycolmap`、`kapture`、`curope`、`dust3r_visloc`、`pillow_heif`、`models`。
它们全部出现在 vendor 的 demo / visloc / retrieval 分支，**不在本工程使用的 sparse_ga 路线上**，
因此未安装，也不影响入口执行。

## 3. 仓库与权重准备

- 分支：`submission/peixinnan`（从 `main` 拉出）
- 权重目标路径：`checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`
- 期望大小：`2754910614` 字节
- 期望 SHA256：`e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2`

下载方式：先用 `wget -c`，实测单连接约 37 KB/s；改用 `aria2c` 多连接：

```bash
aria2c -x 16 -s 16 -k 1M --file-allocation=none \
  -o MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  'https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
```

实测稳定在 400–550 KiB/s。

**校验结果（2026-11-26T03:51Z）**：

```
stat -c %s   ：2754910614      （期望 2754910614，一致）
sha256sum    ：e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2
               （期望值一致）
aria2 日志   ：Status OK / download completed（无 .aria2 控制文件残留）
```

两个条件同时满足，权重文件视为完整。

### 3.1 静态预检结果（不需要权重，已执行）

以下检查于 **2026-11-26T03:12–03:19Z** 执行，用于在下载期间提前排除“跑到一半才发现”的失败点。

**vendor 快照哈希**（`mast3r.py` 在 `backend_prepare` 阶段会强制校验）：

| 对象 | 实测 SHA256 | 代码常量 | 结果 |
|---|---|---|---|
| `vendor/manifest.json` | `140d74898111e78251fa6b8ea0be59431587e46b773cfe4fd6f67ace39ec0579` | `mast3r.py:25` | 一致 |
| `vendor/mast3r/mast3r/cloud_opt/sparse_ga.py` | `195cc7510b65871f40ea51cb36df40313187fa3ac47c412576c755c44573ad74` | `sparse_io.py:16` | 一致 |

**数据完整性**（证明 `data/` 未改动）：

```bash
python3 - <<'EOF'
import json, hashlib, pathlib
for s in ('scene_a','scene_b','scene_c'):
    m = json.load(open(f'data/{s}/manifest.json'))
    for e in m['images']:
        h = hashlib.sha256((pathlib.Path('data')/s/e['file']).read_bytes()).hexdigest()
        print(s, e['file'], 'OK' if h == e['sha256'] else 'MISMATCH')
EOF
```

结果：**12/12 OK**（scene_a 2 张、scene_b 7 张、scene_c 3 张），无 MISMATCH。

### 3.2 入口预检（`--dry-run`，已通过）

```bash
python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --dry-run
```

- 执行时间：2026-11-26T03:56Z
- run_id：`20261126T035611Z_077779e0fdea`
- 结果：`status: "succeeded"`，`error: null`
- 阶段耗时：
  - `configuration`：0.04 s
  - `asset_validation`：2.12 s（含两遍权重 SHA256 的其中一遍；权重已在页缓存，约 1.3 GB/s）
- 产物：`resolved_config.yaml`、`config_sources.json`、`preflight.json`、
  `run_manifest.json`、`run_status.json`
- 预检值：`images=2`、`directed_pairs=2`、`weight_sha256=e28f91b4…`

说明：dry-run 只验证资源与参数，**不加载模型、不处理图像**，也不证明几何质量
（`geometry_quality_accepted: false`）。此前 02:25 因绝对路径失败的 run 已删除，
本 run 是修正 `assets/weights.json` 后的第一次成功预检。

### 3.3 完整运行验证（任务 1 的第二条要求：产出 `artifacts/scene.ply`）

去掉 `--dry-run` 后完整执行：

```bash
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml
```

- run_id：`20261126T042248Z_3d7558c62c88`（即 `BASELINE.md` 的基线 run）
- `status: "succeeded"`，10 个阶段全部完成；
- 产出 `artifacts/scene.ply`（398,032 B）、`artifacts/cameras.json`（3,887 B）、
  `artifacts/point_sources.npz`（62,019 B）；
- 2 相机 / 14,734 点 / `readback_checks` 五项均为 `true`。

过程中并非一次成功：`--dry-run` 通过后，完整运行曾有 **3 次**在
`backend_reconstruct` 阶段因 `CUBLAS_STATUS_NOT_INITIALIZED` 失败
（`20261126T041729Z_4b2acf11e162`、`20261126T041937Z_581439180005`、
`20261126T042110Z_7904ef813fba`）；清空 `LD_LIBRARY_PATH` 后第 4 次成功。
定位过程见 §4.5，这三个失败 run 的产物保留在 `outputs/scene_a/` 作为证据。

## 4. 遇到的问题与处置

### 4.1 工程自带 `.venv` 失效，未使用虚拟环境

现象：`.venv/bin/python` 是悬空软链，`pyvenv.cfg` 的 `home` 也指向不存在的历史路径：

```
bin/python  -> /workspace/OLD-INSTANCE/.venv/bin/python
pyvenv.cfg  : home = /workspace/OLD-INSTANCE/.venv/bin
site-packages: 只有 pip / setuptools / pkg_resources（共约 13 MB）
```

判断：该 venv 由打包机器的工程副本带入，与 `assets/weights.json` 里 `/workspace/OLD-INSTANCE/` 同源，
内部没有任何项目依赖，且无法启动。

处置：不使用该 venv，直接使用系统 Python 3.11.1。原因：
1. 系统环境已具备正确的 CUDA 版 torch（2.10.0+cu128，sm_75 在支持列表内）；
2. 重建 venv 需重新下载约 2.5 GB 的 CUDA wheel，且 `requirements.txt` 明确注明
   "the default PyPI torch wheel is the CPU build"，直接安装有装成 CPU 版的风险；
3. 本机为一次性考核容器，隔离收益低于重建成本。

### 4.2 `assets/weights.json` 的 `file` 为绝对路径，被入口拒绝

现象：首次 `--dry-run` 报错：

```
执行失败：Asset path must be a nonempty relative path:
'/workspace/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
```

代码依据：

- `src/reconlab/entry.py:50-52`：
  ```python
  def _asset_path(base, raw, scope):
      if not isinstance(raw, str) or not raw.strip() or Path(raw).is_absolute():
          raise EntryError(f"Asset path must be a nonempty relative path: {raw!r}")
  ```
- `src/reconlab/entry.py:130`：`weight = _asset_path(weights_path.parent, weights.get("file"), root)`
  —— base 是 `assets/` 目录，因此 `file` 相对 `assets/` 解析；
- `docs/RUNNING.md:30`：`Weight and notice paths are relative to assets/weights.json`；
- 同文件中的 `license_notice` 本身就是相对路径 `../vendor/mast3r/CHECKPOINTS_NOTICE`。

**与考场须知的差异**：`kaoshengxuzhi.md` 第 3 条要求填绝对路径，但代码无条件拒绝绝对路径。
仓库原始值 `/workspace/OLD-INSTANCE/checkpoints/...` 同样是绝对路径，在交付状态下不可能通过校验。
本工程以**代码实际行为**为准。

处置：`file` 改为 `../checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`；
`sha256`、`schema_version`、`download_url`、`license_notice` 均未改动。

校验（按 `_asset_path` 的逻辑复现）：

```
is_absolute        : False     （必须 False）
resolved           : /workspace/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
inside project_root: True      （必须 True）
license_notice     : /workspace/vendor/mast3r/CHECKPOINTS_NOTICE  exists=True
```

直接调用入口的 `_asset_path()` 复验三种取值（2026-11-26，本机执行）：

| `file` 取值 | 结果 |
|---|---|
| 原始绝对路径（不改） | `EntryError: Asset path must be a nonempty relative path` |
| 按须知字面改成绝对路径 | 同上，同样被拒绝 |
| 当前相对路径 | 解析为 `/workspace/checkpoints/…`，`exists=True` |

因此交付状态下**唯一能通过入口的写法是相对路径**；须知的绝对路径示例与本实现冲突，
以代码为准。`sha256` 字段保持原值未改。

### 4.3 缺少 `roma`

`requirements.txt` 列了 `roma`，但环境未安装（`ModuleNotFoundError: roma`）。

处置：

```bash
pip install --no-deps roma
```

使用 `--no-deps` 是为了防止 pip 重新解析 torch 依赖、把已安装的 cu128 版本替换为 CPU 版。
安装后验证：

```
roma OK
2.10.0+cu128 True
```

### 4.4 单连接下载过慢

`wget` 单连接实测约 37 KB/s，2.57 GiB 需要约 21 小时。服务器返回
`Accept-Ranges: bytes`，支持分段下载，改用 `aria2c -x 16 -s 16` 后
提升到 400–550 KiB/s（约 14 倍）。

期间一个容易误判的点：aria2 以稀疏文件方式写入，`stat -c %s` / `ls -l` 显示的
是**文件表观长度**，下载到一半就会接近 2.5 GiB。真实进度只能看 aria2 日志的 `DL:` 行，
或在完成后用 `sha256sum` 确认。

### 4.5 `LD_LIBRARY_PATH` 混用 CUDA 12.2，导致 cuBLAS 初始化失败

**现象**：scene_a 完整运行在 `backend_reconstruct` 阶段失败：

```
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED
when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```

失败的 run（保留作为证据）：

- `20261126T041729Z_4b2acf11e162`（第一次）
- `20261126T041937Z_581439180005`（原样重跑，稳定复现）
- `20261126T042110Z_7904ef813fba`（第三次，与上两次报错相同）

> 关于 `20261126T042110Z_7904ef813fba`：早期记录称它是 `CUDA_LAUNCH_BLOCKING=1` 的诊断 run。
> **命令行的环境变量没有落进任何产物**（`run_manifest.json` 的 `provenance.environment`
> 只记录 python/executable/platform/模块路径），故这一点**未证实**；能证实的只有它的报错与
> 前两次逐字相同（`phase=backend_reconstruct`、`CUBLAS_STATUS_NOT_INITIALIZED`）。

**定位过程**：

1. 模型加载与 `backend_prepare` 成功，失败点在匹配阶段；
2. 完整 traceback：

   ```
   sparse_ga.py:605  extract_correspondences
     → fast_nn.py:157  tree1.query(...)
       → fast_nn.py:82  bruteforce_reciprocal_nns
         → fast_nn.py:31  return A @ B.T
   ```

3. 检查库解析：

   ```bash
   ldd .../site-packages/nvidia/cublas/lib/libcublas.so.12 | grep cublasLt
   # libcublasLt.so.12 => /usr/local/cuda-12.2/lib64/libcublasLt.so.12
   ```

   即 pip 的 cuBLAS **12.8.4.1** 在运行时加载了系统 CUDA **12.2.5.6** 的 `libcublasLt`。

4. 来源：

   ```
   ~/.zshrc:111   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.2/lib64
   ~/.bashrc:110  同上
   ```

   `LD_LIBRARY_PATH` 的优先级高于 `libcublas.so.12` 自带的 RUNPATH（`$ORIGIN`），
   因此版本被系统 12.2 覆盖。

5. 反证：把 `LD_LIBRARY_PATH` 置空后，同一个库解析到同目录的 12.8 版本：

   ```bash
   LD_LIBRARY_PATH= ldd .../nvidia/cublas/lib/libcublas.so.12 | grep cublasLt
   # libcublasLt.so.12 => .../site-packages/nvidia/cublas/lib/libcublasLt.so.12
   ```

**根因**：cuBLAS 主库 12.8 与 `libcublasLt` 12.2 版本不匹配，导致 legacy GEMM
（`cublasSgemm`）的 handle 初始化失败。不是项目代码、vendor 快照或权重的问题。

**处置**：运行入口时清空该变量：

```bash
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml
```

验证结果：run `20261126T042248Z_3d7558c62c88` 完整通过（10 阶段 succeeded，
2 相机 / 14,734 点，两条 finite 的 `>> final loss`）。

**未做的事**：没有修改 `~/.zshrc` / `~/.bashrc`，也没有改项目代码或 vendor。
所有 run 命令统一加 `LD_LIBRARY_PATH=` 前缀即可复现；如果后续所有命令都容易忘记，
再考虑清理 shell 配置。

### 4.6 文档与实际工程存在漂移

- `docs/` 与 `README.md` 共引用 17 个 `scripts/*.py` 路径，其中 16 个不存在，
  实际只有 `scripts/run_reconstruction.py`；
- `docs/RUNNING.md` 提到的 `python -m unittest discover -s tests/unit`，但仓库没有 `tests/` 目录；
- 文档提到的 `benchmarks/`、`work/` 目录也不存在。

处置：只使用 `scripts/run_reconstruction.py` 入口；不以文档中的检查脚本作为验证依据。

## 5. 复现命令

```bash
# 环境
python3 --version
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "import torch; a=torch.randn(4096,4096,device='cuda'); print((a@a).sum().item())"

# 依赖
python3 -c "import torch, torchvision, numpy, scipy, PIL, cv2, einops, roma, tqdm, packaging, huggingface_hub; print('deps OK')"

# 权重
stat -c %s checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
sha256sum checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

# 入口预检（dry-run 不碰 GPU，但仍建议统一加前缀）
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml --dry-run

# 完整运行（必须清空 LD_LIBRARY_PATH，原因见 §4.5）
LD_LIBRARY_PATH= python3 scripts/run_reconstruction.py --config configs/scenes/scene_a.yaml
```

## 6. 完成项

| 项目 | 状态 |
|---|---|
| vendor 快照哈希预检 | ✅ 已完成（2026-11-26T03:12–03:19Z，见 §3.1） |
| `data/` 12 张图的 sha256 对照 | ✅ 12/12 OK（见 §3.1） |
| 权重 SHA256 校验 | ✅ 完成：大小 2754910614、SHA256 均与期望一致（2026-11-26T03:51Z） |
| 首次 `--dry-run` 通过 | ✅ `20261126T035611Z_077779e0fdea`，两阶段 succeeded（见 §3.2） |
| CUDA 库混用问题 | ✅ 已定位并规避：运行需加 `LD_LIBRARY_PATH=` 前缀（见 §4.5） |
| scene_a 完整运行（任务 1 验证） | ✅ `20261126T042248Z_3d7558c62c88`：10 阶段 succeeded，产出 `artifacts/scene.ply`（见 §3.3） |
| scene_a 基线 | ✅ 同一 run，指标见 `BASELINE.md` |
