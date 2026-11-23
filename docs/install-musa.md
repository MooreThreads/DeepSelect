# DeepSelect MUSA 安装指南（发布到 `docs/install-musa.md` 的草稿）

> 适用于 `deepselect-musa`（DeepSelect 的摩尔线程 MUSA 后端）。CUDA/Ascend 用户请走源码安装。

## 1. 前置条件

| 项 | 要求 |
| --- | --- |
| GPU | MTT S5000（offload arch `mp_31`） |
| 驱动 | 与 MUSA SDK 5.2 匹配的驱动（mccx-204 实测 `mthreads-gmi 2.4.2` / driver `5.2.0-server`） |
| MUSA SDK | 5.2，安装于 `/usr/local/musa`（`MUSA_HOME`），`mcc` 可用 |
| Python | 3.10 / 3.12（其它版本未验证） |
| torch / torch_musa | `2.11.0.post1+musa5.2.0`（必须成对；见 `docs/compatibility-musa52.json`） |

先自检环境：

```bash
mthreads-gmi | head
echo "$MUSA_HOME" ; which mcc ; mcc --version | head -1
python -c "import torch, torch_musa; print(torch.__version__, torch_musa.__version__)"
python -c "import torch; print(torch.musa.is_available())"   # 必须 True
```

## 2. 安装

### 2.1 PyPI（推荐）

```bash
pip install deepselect-musa
# 锁定版本：
pip install "deepselect-musa==0.0.1"
python -m deep_select.compat
```

PyPI 上当前版本是 `0.0.1`，wheel 为 `cp310` / `manylinux_2_35_x86_64`
（MUSA SDK 5.2、`torch`/`torch_musa` 2.11.0.post1、Python 3.10、glibc >= 2.35）。
PyPI 不允许 PEP 440 local version，因此 wheel 版本不带 `+musa...torch...`；
完整构建栈记录在 wheel 内的 `_build_info.py`，`python -m deep_select.compat`
会在 import 时校验。

### 2.2 GitHub Release wheel（离线/固定版本）

每个 release 附带 `linux_x86_64` wheel 与 `SHA256SUMS`，适合无索引/离线环境：

```bash
pip install https://github.com/MooreThreads/DeepSelect/releases/download/v0.0.1/\
deepselect_musa-0.0.1-cp310-cp310-linux_x86_64.whl
sha256sum deepselect_musa-*.whl   # 与 release 里的 SHA256SUMS 对比
```

### 2.3 从源码构建

```bash
git clone https://github.com/MooreThreads/DeepSelect.git
cd DeepSelect
git submodule update --init --recursive
DEEP_SELECT_BUILD_TARGET_PLATFORM=MUSA python3 -m pip install -v .
```

sdist 尚未发布到 PyPI；源码构建需要 MUSA SDK + mcc，且需要与目标环境匹配的
`torch_musa`。

## 3. 安装后自检

```bash
python -m deep_select.compat        # 版本、设备、兼容性结论
python - <<'PY'
import torch, deep_select
x = torch.randn(4, 204800, dtype=torch.bfloat16, device="musa")
vals, idx = deep_select.topk(x, 1024)
print(vals.shape, idx.shape, vals.dtype, idx.dtype)
PY
```

## 4. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ImportError: failed to load the backend ... deep_select_musa` | 装了 `deepselect-musa` 但环境不是 MUSA（`torch.musa.is_available()` False） | 确认驱动/GPU；`python -m deep_select.compat` |
| `RuntimeError: deepselect-musa was built for torch_musa 2.11..., but ... is installed` | torch/torch_musa 版本与 wheel 不匹配 | 安装对应构建栈的版本，或升级/降级 torch_musa |
| `No matching distribution found for deepselect-musa` | 当前 Python/glibc 没有匹配 wheel（仅 cp310 / manylinux_2_35 已发布） | 用 GitHub Release 的 `linux_x86_64` wheel 或源码构建 |
| 跑出 `device assertion` / NaN 相关 abort | 输入含 NaN 且 `abort_when_nan_found=True`（默认） | 语义如此；需要继续计算时传 `abort_when_nan_found=False` |
| `sorted_value` 与 `sorted_index` 同时为 True 报错 | 两者不能同时使用（MUSA 实现限制） | 只保留一个；参考 README |
| `pip install` 很慢 / 编译失败 | 走的是源码构建 | 用 §2.1/§2.2 的预编译 wheel |

## 5. 卸载与版本查询

```bash
pip show deepselect-musa
pip uninstall deepselect-musa
```

兼容矩阵与发布历史见 `docs/compatibility-musa52.json` 与 GitHub Release（`MooreThreads/deepselect-whl`）。
