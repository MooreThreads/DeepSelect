# DeepSelect MUSA 0.0.1

首个 `deepselect-musa` 发布线（MUSA 后端打包 + 短序列 BF16 TopK 优化）。

## 安装

```bash
pip install deepselect-musa
# 或锁定版本：
pip install "deepselect-musa==0.0.1"
python -m deep_select.compat
```

PyPI wheel：`deepselect_musa-0.0.1-cp310-cp310-manylinux_2_35_x86_64.whl`。

前置条件：MUSA SDK 5.2（`MUSA_HOME=/usr/local/musa`）、
`torch`/`torch_musa` 2.11.0.post1、Python 3.10、x86_64（glibc >= 2.35）、
MTT S5000（`mp_31`）。

每个 GitHub Release 同时附带 `linux_x86_64` wheel 与 `SHA256SUMS`，
可用于离线/固定版本安装。

## 变更

- MUSA 后端：BF16/FP32 native TopK、AOT 预编译扩展、MUSA 平台探测。
- 短序列 BF16 TopK 专用路径（`rows<=8`、`width<=2048`、`topk<=1024`、
  `return_value=False`、连续行、无 `end`/NaN abort），其余情况回退通用路径。
- 统一 MUSA benchmark 入口：`benchmarks/musa_bench.py --suite short|wide|all`。
- BF16 NaN payload 检测修复（覆盖 signaling NaN）。
- 打包：确定性版本注入、构建 provenance、运行时兼容性检查
  （`python -m deep_select.compat`）。

## 兼容性

- PyPI 版本使用 public version `0.0.1`（PyPI 不接受 PEP 440 local version），
  一个版本对应一个已验证栈：MUSA 5.2 / torch_musa 2.11.0.post1 / cp310 / glibc 2.35。
  换 torch/SDK 时升版本号发布；完整构建栈记录在 `_build_info.py`。
- `linux_x86_64` wheel 用于 GitHub Release 直链安装；PyPI 上传前会自动
  重打为 `manylinux_2_35_x86_64`。
