"""JIT loader for the MUSA C++ TopK kernel."""

from __future__ import annotations

import functools
import hashlib
import importlib
import os
from pathlib import Path

import torch


_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "csrc"
    / "musa_kernels"
    / "topk.mu"
)
_scratch_cache: dict[tuple[torch.device, int, int, int, int], torch.Tensor] = {}
_two_stage_scratch_cache: dict[tuple[torch.device, int, int, int], torch.Tensor] = {}
_TWO_STAGE_PARTS = 16
_persist_scratch_cache: dict[
    tuple[torch.device, int, int, int, int], torch.Tensor
] = {}
_PERSIST_MAX_ROWS = 8
_PERSIST_MIN_WIDTH = 16384  # the sampler's real widths are 128K/129280
_PERSIST_MAX_TOPK = 2048


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
_fp32_scratch_cache: dict[
    tuple[torch.device, int, int, int, int], torch.Tensor
] = {}


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
        stream_id = torch.musa.current_stream(input.device).musa_stream
        key = (input.device, stream_id, rows, width, parts)
        scratch = _persist_scratch_cache.get(key)
        if scratch is None:
            scratch = torch.zeros(
                _module().persist_scratch_ints(rows),
                dtype=torch.int32,
                device=input.device,
            )
            _persist_scratch_cache[key] = scratch
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
        stream_id = torch.musa.current_stream(input.device).musa_stream
        key = (input.device, stream_id, rows, input.shape[1], parts)
        scratch = _scratch_cache.get(key)
        if scratch is None:
            elements = (
                2 * rows * parts * 256
                + 2 * rows * parts
                + 6 * rows
                + rows * input.shape[1]
            )
            scratch = torch.empty(elements, dtype=torch.int32, device=input.device)
            _scratch_cache[key] = scratch
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
        end is None
        and 0 < rows <= 16
        and 262144 <= width <= 524288
        and width % 4 == 0
        and 128 <= topk <= 2048
        and input.is_contiguous()
        and os.getenv("DEEP_SELECT_MUSA_FP32_TWO_STAGE") == "1"
        and torch.musa.get_device_capability(input.device) == (3, 1)
    ):
        parts = 8 if rows > 8 and 512 < topk <= 1024 else _TWO_STAGE_PARTS
        module = _module()
        launch = getattr(module, "topk_fp32_two_stage", None)
        if launch is not None:
            with torch.musa.device(input.device):
                capturing = torch.musa.is_current_stream_capturing()
                key = (input.device, torch.musa.current_stream().musa_stream, rows, parts)
                scratch = None if capturing else _two_stage_scratch_cache.get(key)
                if scratch is None:
                    scratch = torch.empty(
                        module.two_stage_scratch_ints(rows, parts),
                        dtype=torch.int32,
                        device=input.device,
                    )
                    if not capturing:
                        _two_stage_scratch_cache[key] = scratch
            fused_sort = sort_by_value and topk <= 1024
            launch(
                input, output, scratch, output_value, output_value is not None,
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
        stream_id = torch.musa.current_stream(input.device).musa_stream
        key = (input.device, stream_id, rows, width, parts)
        scratch = _fp32_scratch_cache.get(key)
        if scratch is None:
            elements = rows * parts * 256 + rows * parts + 7 * rows
            scratch = torch.empty(
                elements, dtype=torch.int32, device=input.device
            )
            _fp32_scratch_cache[key] = scratch
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
