# Environment Variables

DeepSelect reads runtime dispatch options from the current process
environment. Set or export variables before launching the workload or calling
`deep_select.topk`.

## MUSA Kernel Dispatch

| Variable | Default | Meaning |
| --- | --- | --- |
| `DEEP_SELECT_MUSA_FP32_TWO_STAGE` | `0` (unset) | Set to `1` to opt in to the MP31 `(3, 1)` FP32 two-stage TopK selector. Only raw `1` enables it. |

When `DEEP_SELECT_MUSA_FP32_TWO_STAGE=1` is set on MP31 `(3, 1)`, the FP32
path dispatches to the two-stage selector for contiguous inputs without `end`,
`1` to `16` rows, widths from `262144` to `524288` divisible by `4`, and
`topk` from `128` to `2048` (`topk > 1024` uses the existing value sort when
`sorted=True`). It uses cyclic 128-element warp-tile partitioning with
deferred `compact12` fallback, and returns early after the partition scan for
NaN rows.
