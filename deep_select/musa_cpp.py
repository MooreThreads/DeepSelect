"""JIT loader for the MUSA C++ TopK kernel."""

from __future__ import annotations

from collections.abc import Callable
import functools
import hashlib
import importlib
import os
from pathlib import Path
from typing import Any

import torch


_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "csrc"
    / "musa_kernels"
    / "topk.mu"
)
_scratch_cache: dict[tuple[Any, ...], torch.Tensor] = {}
_two_stage_scratch_cache: dict[tuple[Any, ...], torch.Tensor] = {}
_persist_scratch_cache: dict[tuple[Any, ...], torch.Tensor] = {}
_fp32_scratch_cache: dict[tuple[Any, ...], torch.Tensor] = {}
_TWO_STAGE_PARTS = 16
_PERSIST_MAX_ROWS = 8
_PERSIST_MIN_WIDTH = 16384  # the sampler's real widths are 128K/129280
_PERSIST_MAX_TOPK = 2048


@functools.lru_cache(maxsize=None)
def _device_capability(device_index: int) -> tuple[int, int]:
    return torch.musa.get_device_capability(device_index)


def _get_scratch(
    cache: dict[tuple[Any, ...], torch.Tensor],
    device: torch.device,
    key_parts: tuple[int, ...],
    size_fn: Callable[[], int],
    *,
    zero: bool = False,
) -> torch.Tensor:
    with torch.musa.device(device):
        capturing = torch.musa.is_current_stream_capturing()
        key = (device, torch.musa.current_stream().musa_stream, *key_parts)
        scratch = None if capturing else cache.get(key)
        if scratch is None:
            alloc = torch.zeros if zero else torch.empty
            scratch = alloc(size_fn(), dtype=torch.int32, device=device)
            if not capturing:
                cache[key] = scratch
        return scratch


def _persist_parts(rows: int, width: int) -> int | None:
    """Per-row CTA count for the persistent path, or None if unusable.

    The policy uses a larger CTA count for two-row inputs and a smaller count
    otherwise, with more workers for multi-row inputs. A 16 B aligned row slice
    is required; shapes that cannot split evenly use the fallback policy.
    """
    totals = (48,) if rows == 6 and width >= 65536 else (
        (40, 32) if rows == 2 else (32,)
    )
    for total in totals:
        if total % rows:
            continue
        parts = total // rows
        if parts <= 0 or rows * parts > 56:
            continue
        if (width + parts - 1) // parts % 8 == 0:
            return parts
    return None


@functools.lru_cache(maxsize=1)
def _module():
    try:
        return importlib.import_module(".deep_select_musa_cpp", __package__)
    except ImportError as aot_error:
        if not _SOURCE.is_file():
            raise ImportError(
                "DeepSelect MUSA AOT extension is unavailable and the JIT "
                f"source checkout was not found at {_SOURCE}"
            ) from aot_error

    from torch_musa.utils.musa_extension import load

    digest_builder = hashlib.sha256(_SOURCE.read_bytes())
    digest = digest_builder.hexdigest()[:16]
    extra_musa_cflags = [
        "-O3",
        "--offload-arch=mp_31",
        "-ffast-math",
        "-Wno-error=address-of-temporary",
        "-fmusa-flush-denormals-to-zero",
        "-fno-signed-zeros",
        "-mllvm",
        "-mtgpu-opt-level=1",
        "-mllvm",
        "-mtgpu-load-store-opt=1",
        "-mllvm",
        "-mtgpu-fold-global-ldst=1",
    ]
    return load(
        name=f"deep_select_musa_topk_{digest}",
        sources=[str(_SOURCE)],
        extra_include_paths=[
            str(_SOURCE.parents[1] / "3rdparty" / "mutlass" / "include")
        ],
        extra_musa_cflags=extra_musa_cflags,
        verbose=False,
    )


def launch_topk_bf16(
    input: torch.Tensor,
    output: torch.Tensor,
    output_value: torch.Tensor | None = None,
    end: torch.Tensor | None = None,
    idx_fill: int = 2147483647,
    value_fill: float = float("-inf"),
    abort_when_nan_found: bool = True,
) -> None:
    rows = input.shape[0]
    width = input.shape[1]
    topk = output.shape[1]
    parts = (
        _persist_parts(rows, width)
        if (
            end is None
            and input.dtype == torch.bfloat16
            and output.dtype in (torch.int32, torch.int64)
            and 0 < rows <= _PERSIST_MAX_ROWS
            and width >= _PERSIST_MIN_WIDTH
            and width % 8 == 0
            and 0 < topk <= _PERSIST_MAX_TOPK
            and not abort_when_nan_found
            and input.is_contiguous()
            and output.stride(1) == 1
        )
        else None
    )
    if parts is not None:
        threads = 256 if rows == 1 else 1024
        scratch = _get_scratch(
            _persist_scratch_cache,
            input.device,
            (rows, width, parts),
            lambda: _module().persist_scratch_ints(rows),
            zero=True,
        )
        _module().topk_bf16_persist(
            input,
            output,
            scratch,
            output_value,
            parts,
            threads,
            abort_when_nan_found,
        )
        return
    if end is None and rows <= 8 and input.shape[1] > 131072:
        parts = 48 if input.shape[1] >= 1048576 else 32
        scratch = _get_scratch(
            _scratch_cache,
            input.device,
            (rows, input.shape[1], parts),
            lambda: (
                2 * rows * parts * 256
                + 2 * rows * parts
                + 6 * rows
                + rows * input.shape[1]
            ),
        )
        _module().topk_bf16_multi(
            input,
            output,
            scratch,
            parts,
            256,
            output_value,
            abort_when_nan_found,
        )
    else:
        _module().topk_bf16(
            input,
            output,
            output_value,
            end,
            idx_fill,
            value_fill,
            abort_when_nan_found,
        )


def launch_topk_fp32(
    input: torch.Tensor,
    output: torch.Tensor,
    output_value: torch.Tensor | None = None,
    end: torch.Tensor | None = None,
    idx_fill: int = 2147483647,
    value_fill: float = float("-inf"),
    abort_when_nan_found: bool = True,
    sort_by_value: bool = False,
) -> bool:
    """Returns True when the selector already ordered the values itself."""
    rows, width = input.shape
    topk = output.shape[1]
    if (
        os.getenv("DEEP_SELECT_MUSA_FP32_TWO_STAGE", "1") != "0"
        and end is None
        and 0 < rows <= 16
        and 262144 <= width <= 524288
        and width % 4 == 0
        and 128 <= topk <= 2048
        and input.is_contiguous()
        and _device_capability(
            input.device.index
            if input.device.index is not None
            else torch.musa.current_device()
        )
        == (3, 1)
    ):
        parts = 8 if rows > 8 and 512 < topk <= 1024 else _TWO_STAGE_PARTS
        module = _module()
        launch = getattr(module, "topk_fp32_two_stage", None)
        if launch is not None:
            scratch = _get_scratch(
                _two_stage_scratch_cache,
                input.device,
                (rows, parts),
                lambda: module.two_stage_scratch_ints(rows, parts),
            )
            fused_sort = sort_by_value and topk <= 1024
            launch(
                input, output, scratch, output_value,
                fused_sort, abort_when_nan_found, parts,
            )
            return fused_sort
    if (
        end is None
        and rows <= 8
        and width >= 1048576
        and input.is_contiguous()
    ):
        parts = 48
        scratch = _get_scratch(
            _fp32_scratch_cache,
            input.device,
            (rows, width, parts),
            lambda: rows * parts * 256 + rows * parts + 7 * rows,
        )
        _module().topk_fp32_multi(
            input,
            output,
            scratch,
            parts,
            256,
            output_value,
            abort_when_nan_found,
        )
        return False
    return bool(
        _module().topk_fp32(
            input,
            output,
            output_value,
            end,
            idx_fill,
            value_fill,
            abort_when_nan_found,
            sort_by_value,
        )
    )


def launch_output_idx_offset(
    output: torch.Tensor,
    end: torch.Tensor | None,
    offset: torch.Tensor,
    full_length: int,
) -> None:
    _module().output_idx_offset(output, end, offset, full_length)


def launch_sort_output_fused(
    output: torch.Tensor,
    output_value: torch.Tensor | None,
    end: torch.Tensor | None,
    offset: torch.Tensor | None,
    full_length: int,
) -> None:
    _module().sort_output_fused(output, output_value, end, offset, full_length)


def launch_sort_output(
    output: torch.Tensor,
    output_value: torch.Tensor | None,
    end: torch.Tensor | None,
    full_length: int,
    sort_by_value: bool,
) -> None:
    _module().sort_output(
        output, output_value, end, full_length, sort_by_value
    )
