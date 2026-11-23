# Changelog

## 0.0.1 — 2026-11-20

First `deepselect-musa` release line (MUSA backend packaging + short BF16 topk).
Published to PyPI as `deepselect-musa==0.0.1` (`cp310` / `manylinux_2_35_x86_64`).

### Added
- **MUSA backend**: AOT `.mu` TopK kernels for BF16/FP32, public `deep_select.topk`
  dispatch, MUSA platform detection, and prebuilt-wheel/JIT diagnostic loading.
- **Short BF16 TopK fast path** (`topk_bf16_short_kernel`): single-CTA-per-row
  shared-key 12+8-bit radix selector for small-batch contiguous short rows
  (`rows <= 8`, `width <= 2048`, `topk <= 1024`, `return_value=False`,
  no `end`/NaN-abort), falling back to the generic path otherwise.
- **Unified MUSA benchmark**: `benchmarks/musa_bench.py --suite short|wide|all`
  with warmup/iterations/profiler/output-JSON support; replaces the previous
  per-scenario benchmark scripts.
- **Packaging**: deterministic, injectable version (`DEEP_SELECT_VERSION` /
  `DEEP_SELECT_LOCAL_LABEL` / `DEEP_SELECT_DIST_NAME`), local-version wheel tag
  `musa<SDK>torch<torch>` (e.g. `musa5.2.0torch2.11.0.post1`), build metadata
  (`deep_select/_build_info.py`), runtime compatibility checks
  (`python -m deep_select.compat`), wheel build/index/publish scripts, and the
  MUSA install guide (`docs/install-musa.md`).

### Fixed
- **BF16 NaN payload detection**: the persistent BF16 path now detects every
  non-zero mantissa payload (including signaling NaN), not only quiet-NaN ranges.
- README benchmark references now match the unified benchmark CLI.

### Notes
- Prebuilt `deepselect-musa` wheels target MUSA SDK 5.2, `torch`/`torch_musa`
  2.11.0.post1, Python 3.10, and MTT S5000 (`mp_31`).
- PyPI distribution of this release line carries the source distribution only;
  binary wheels are published through the GitHub Releases wheel index because
  PEP 440 local versions are not accepted by PyPI.
