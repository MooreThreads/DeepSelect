# DeepSelect

DeepSelect is a high performance implementation of the TopK kernel used in DeepSeek Sparse Attention (DSA) (which is used in DeepSeek V3.2, DeepSeek V4, and DeepSeek V4.1 models) and the sampler. This branch supports NVIDIA CUDA, Huawei Ascend, and Moore Threads MUSA platforms.

DeepSelect 是为 DeepSeek 稀疏注意力（DeepSeek Sparse Attention, DSA）和采样器（Sampler）定制的高性能 TopK 算子。本分支支持 NVIDIA CUDA、华为昇腾（Ascend）和摩尔线程 MUSA 平台。

The MUSA implementation, shape coverage, limitations, and validation are documented in
[`docs/musa-feasibility.zh.md`](docs/musa-feasibility.zh.md).

## News

- 2026.10.29: Added the MUSA backend and optimized native C++ BF16/FP32 radix paths
- 2026.09.30: We've released TopK kernels for Huawei Ascend NPU
- 2026.09.10: We've released a brief analysis of the algorithm and its implementation: [English](docs/DeepSelect-deep-dive.md) | [中文](docs/DeepSelect-deep-dive.zh.md)
- 2026.09.10: We've released DeepSelect v1.0.0

## Supported Cases

TopK workloads vary widely, and the fastest algorithm & implementation highly depends on the input dtype, `batch_size`, `vocab_size`, and `topk`. This repository only focuses on the following cases:

### Lightning Indexer Scenario

This scenario covers:
- Input dtype: `torch.bfloat16`
- `batch_size`: $1 \sim +\infty$ (both large and small batch sizes are optimized)
- `vocab_size`: $1 \sim +\infty$ (both large and small vocabularies are optimized)
- `topk`: small (must be $\le 4096$; larger values are not supported), mainly optimized for `topk=512` (i.e., DeepSeek V4's `topk`)

Recommendations:
- Disable `sorted_index` unless the output has to be ordered by index; enabling it costs performance.
- Set `return_value=False` when the values are not needed. This skips the value output and is faster.

### Sampling Scenario

This scenario covers:
- Input dtype: `torch.float32`
- `batch_size`: $1 \sim +\infty$
- `vocab_size`: around 128K
- `topk`: small (must be $\le 4096$; larger values are not supported)

The CUDA and MUSA implementations support both `torch.float32` and
`torch.bfloat16`. The Ascend implementation supports `torch.bfloat16` only.

## Performance

The CUDA and Ascend graphs use the benchmark in [`tests/test.py`](tests/test.py).
The MUSA graph uses [`benchmarks/musa_library_bench.py`](benchmarks/musa_library_bench.py).
All graphs compare against `torch.topk` on the same device and input. The
metric is effective memory bandwidth: TopK does no floating-point math, so a
FLOP rate would not be meaningful here.

### Lightning Indexer Scenario (CUDA)

bfloat16, `topk = 512`, one subplot per batch size.

![DeepSelect vs torch.topk, bfloat16 Lightning Indexer, CUDA](assets/perf_bf16_cuda.png)

### Lightning Indexer Scenario (Ascend)

bfloat16, `topk = 512`, one subplot per batch size.

![DeepSelect vs torch.topk, bfloat16 Lightning Indexer, Ascend](assets/perf_bf16_ascend.png)

### Lightning Indexer Scenario (MUSA)

bfloat16, `topk = 512`, one subplot per batch size. The plot uses a readable
0–2 TB/s y-axis while keeping the same canvas, colors, and subplot geometry as
the CUDA figure.

![DeepSelect vs torch.topk, bfloat16 Lightning Indexer, MUSA](assets/perf_bf16_musa.png)

The public API wins all 18 README-aligned points. Speedup ranges are
2.82–10.69x for batch 6, 3.53–72.28x for batch 512, and 4.34–131.05x for batch
4096. Batch 512 and 4096 use an exact sampled-survivor selector from 64K
columns onward. Batch 6 uses compact radix at 16K and a 48-CTA persistent grid
from 64K onward. Absolute effective bandwidth is hardware-specific and is not
claimed to match NVIDIA CUDA.

### Sampling Scenario (CUDA)

float32, `vocab_size = 129280`, `topk = 512`.

![DeepSelect vs torch.topk, float32 Sampling, CUDA](assets/perf_fp32_cuda.png)

### Sampling Scenario (MUSA)

float32, `vocab_size = 129280`, `topk = 512`. This uses the CUDA-aligned
sampling semantics: `sorted=True`, `return_value=True`, and int64 indices.

![DeepSelect vs torch.topk, float32 Sampling, MUSA](assets/perf_fp32_musa.png)

DeepSelect is 6.37–11.13x faster than `torch.topk` for batch 256–4096 and 5.34x
faster for batch 6. At batch 4096, the exact sampled-survivor path reads the
full row once and falls back to compact radix when the bounded candidate set is
insufficient.

## Installation

```bash
git clone https://github.com/deepseek-ai/DeepSelect.git
cd DeepSelect
git submodule update --init --recursive
pip install -v .
```

For MUSA, build the native MUSA extension explicitly:

```bash
DEEP_SELECT_BUILD_TARGET_PLATFORM=MUSA python3 -m pip install -v .
```

## Usage

```python
import torch
import deep_select

# input: (batch_size, vocab_size), torch.bfloat16 or torch.float32.
# Its row stride must be a multiple of `deep_select.get_stride_requirement()[0]` bytes, and its last dimension must be contiguous.
batch_size, vocab_size, topk = 4, 204800, 1024

x = torch.randn(batch_size, vocab_size, dtype=torch.bfloat16, device="cuda")

values, indices = deep_select.topk(
    x,
    topk,
    sorted=False,              # sort values descending (FP32 on CUDA/MUSA)
    sorted_index=True,         # return each row's indices in ascending order
    indices_type=torch.int32,  # torch.int32 or torch.int64 (The Ascend implementation only supports `torch.int32`)
    return_value=True,         # False skips the value output (~10% faster)
)
# values:  (batch_size, topk) of x.dtype
# indices: (batch_size, topk) of indices_type
```

Use `device="musa"` after importing `torch_musa` for the MUSA backend. Its
native path covers every non-empty CUDA-supported combination: BF16/FP32,
`topk=1..4096`, int32/int64 output, values, per-row `end`, custom OOB fills,
index offset, both sorting modes, and NaN abort. Selection, short-row copy/fill,
sorting, offset, and the NaN guard all run in the MUSA C++/`.mu` extension; no
TileLang or PyTorch TopK fallback is used for those calls.

The capability matrix in
[`benchmarks/musa_capability_bench.py`](benchmarks/musa_capability_bench.py)
compares representative combinations with `torch.topk` and checks selected
values and indices for every case.

For graph capture of the low-batch multi-CTA path, warm up once on the same
explicit MUSA stream passed to `torch.musa.graph(..., stream=stream)`. This
preallocates the per-stream scratch buffer outside capture.

The row stride of the input tensor (`x`) must be aligned to `deep_select.get_stride_requirement()[0]` bytes. For unaligned inputs, padding is necessary.

Both outputs are allocated by the call, and their strides are aligned to `deep_select.get_stride_requirement()[1]` bytes (so they may be non-contiguous). Pass `output_idx=` to write indices into a buffer you own, and that buffer must satisfy the same stride requirement.

For the full signature, see [`deep_select/interface.py`](deep_select/interface.py).

### Variable-length rows

`end` sets a per-row upper bound (exclusive). Rows shorter than `topk` are padded with
`value_oob_fill_value` / `idx_oob_fill_value`:

```python
batch_size, vocab_size = 2, 129280   # 129280 is a multiple of 256, so float32 is fine
x = torch.randn(batch_size, vocab_size, dtype=torch.float32, device="cuda")

end = torch.tensor([129280, 100000], dtype=torch.int32, device="cuda")  # (batch_size,)
values, indices = deep_select.topk(x, 1000, end=end, sorted=True,
                                   indices_type=torch.int64)
```

### NaN handling

NaN checking is always on. With the default `abort_when_nan_found=True`, CUDA
invokes `trap()` and MUSA invokes a native device assertion from a guard kernel.
CUDA and MUSA skip NaN checking for rows whose valid length is `<= topk`.

### Experimental MUSA FP32 dispatch

Set `DEEP_SELECT_MUSA_FP32_TWO_STAGE=1` to try the two-stage selector on
MP31. The selector accepts contiguous FP32 input without `end`, 1 to 16
rows, widths from 262144 to 524288 divisible by four, and `topk` from
128 to 2048. Results with `topk > 1024` use the existing
value sort when requested. Other calls use the existing dispatch. The flag is off by
default. NaN rows produce the row sentinel and return early after the
partition scan without taking the full-row fallback.

## Citation

```text
@misc{deepselect2026,
    title={DeepSelect: High-Performance TopK Kernels for DeepSeek Sparse Attention and Sampling},
    author={Yi Qian and Shengyu Liu and Yichen Li},
    year={2026},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/deepseek-ai/DeepSelect}},
}
```
