"""Focused public-API regression matrix for the MUSA backend."""

from __future__ import annotations

from contextlib import contextmanager
import os
import subprocess
import sys
import unittest.mock

import torch
import torch_musa  # noqa: F401

import deep_select
from deep_select import deep_select_musa, musa_cpp
import kernelkit as kk

from lib import (
    NormalFloatDistribution,
    TestParam,
)
from test import run_testcase


def _case(
    batch: int,
    width: int,
    topk: int,
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    *,
    sorted_value: bool = False,
    sorted_index: bool = False,
    return_value: bool = True,
    end: bool = False,
    offset: bool = False,
    input_distrib=None,
) -> TestParam:
    return TestParam(
        batch,
        width,
        topk,
        sorted_value,
        sorted_index,
        return_value,
        dtype,
        index_dtype,
        enable_end_position=end,
        enable_output_idx_offset=offset,
        input_distrib=input_distrib or NormalFloatDistribution(),
        num_runs=0,
    )


def _run_semantic_matrix() -> None:
    cases = []
    for dtype in (torch.bfloat16, torch.float32):
        for index_dtype in (torch.int32, torch.int64):
            cases.extend(
                [
                    _case(3, 1, 1, dtype, index_dtype, end=True, offset=True),
                    _case(5, 257, 511, dtype, index_dtype, return_value=False),
                    _case(
                        7,
                        32768,
                        512,
                        dtype,
                        index_dtype,
                        return_value=False,
                    ),
                    _case(
                        3,
                        1602,
                        513,
                        dtype,
                        index_dtype,
                        sorted_index=True,
                        offset=True,
                    ),
                    _case(3, 1602, 1024, dtype, index_dtype, end=True),
                    _case(
                        3,
                        4096,
                        1231,
                        dtype,
                        index_dtype,
                        sorted_index=True,
                        return_value=False,
                    ),
                    _case(2, 1602, 2048, dtype, index_dtype, end=True, offset=True),
                    _case(2, 8192, 4096, dtype, index_dtype),
                    _case(
                        1,
                        204800,
                        512,
                        dtype,
                        index_dtype,
                        return_value=False,
                    ),
                    _case(
                        1,
                        1048576,
                        1024,
                        dtype,
                        index_dtype,
                        return_value=False,
                    ),
                ]
            )
            if dtype == torch.float32:
                cases.append(
                    _case(
                        6,
                        129280,
                        512,
                        dtype,
                        index_dtype,
                        sorted_value=True,
                    )
                )
            elif index_dtype == torch.int32:
                cases.append(
                    _case(
                        512,
                        16384,
                        512,
                        dtype,
                        index_dtype,
                        return_value=False,
                    )
                )

    for case in cases:
        assert run_testcase(case, quiet=True), case


def _run_nan_cases() -> None:
    for dtype in (torch.bfloat16, torch.float32):
        input = torch.randn(3, 32768, dtype=dtype, device="musa")
        input[:, 17] = float("nan")
        _, indices = deep_select.topk(
            input,
            512,
            indices_type=torch.int32,
            return_value=False,
            abort_when_nan_found=False,
        )
        assert torch.all(indices[:, 0] == 0x3F3F3F3F).item()

        offsets = torch.tensor([3, 7, 11], dtype=torch.int32, device="musa")
        _, sorted_indices = deep_select.topk(
            input,
            512,
            sorted_index=True,
            output_idx_offset=offsets,
            indices_type=torch.int64,
            return_value=False,
            abort_when_nan_found=False,
        )
        assert torch.all(sorted_indices[:, 0] == 0x3F3F3F3F).item()

        input[:, 0] = float("nan")
        end = torch.tensor([1, 1, 1], dtype=torch.int32, device="musa")
        _, short_indices = deep_select.topk(
            input,
            512,
            end=end,
            indices_type=torch.int32,
            return_value=False,
            abort_when_nan_found=True,
        )
        assert torch.all(short_indices[:, 0] == 0).item()
def _assert_indices_select_topk(
    input: torch.Tensor,
    topk: int = 512,
    index_dtype: torch.dtype = torch.int32,
) -> None:
    output_alignment = 32 // index_dtype.itemsize
    output_stride = (
        (topk + output_alignment - 1) // output_alignment * output_alignment
    )
    output = torch.empty(
        input.shape[0], output_stride, dtype=index_dtype, device="musa"
    )[:, :topk]
    deep_select.topk(
        input,
        topk,
        output_idx=output,
        indices_type=index_dtype,
        return_value=False,
        abort_when_nan_found=False,
    )
    selected = torch.gather(input, 1, output.to(torch.int64))
    expected = torch.sort(input, dim=1, descending=True).values[:, :topk]
    torch.testing.assert_close(
        torch.sort(selected, dim=1, descending=True).values,
        expected,
        rtol=0,
        atol=0,
    )
    sorted_indices = torch.sort(output, dim=1).values
    assert torch.all(sorted_indices[:, 1:] != sorted_indices[:, :-1]).item()


def _run_native_cpp_adversarial_cases() -> None:
    width = 1048576
    pattern = torch.tensor(
        [-3.0, -1.0, 0.0, 1.0], dtype=torch.bfloat16, device="musa"
    )
    low_entropy = pattern.repeat((6 * width + 3) // 4)[: 6 * width].reshape(
        6, width
    )
    _assert_indices_select_topk(low_entropy)
    _assert_indices_select_topk(
        torch.full((1, width), 2.0, dtype=torch.bfloat16, device="musa")
    )

    bf16_compact_width = 131072
    bf16_low_entropy = pattern.repeat(
        (16 * bf16_compact_width + 3) // 4
    )[: 16 * bf16_compact_width].reshape(16, bf16_compact_width)
    _assert_indices_select_topk(bf16_low_entropy)
    _assert_indices_select_topk(
        torch.full(
            (16, bf16_compact_width),
            2.0,
            dtype=torch.bfloat16,
            device="musa",
        )
    )
    bf16_compact_random = torch.randn(
        16, 65536, dtype=torch.bfloat16, device="musa"
    )
    for compact_topk in (1, 513, 4096):
        _assert_indices_select_topk(
            bf16_compact_random, topk=compact_topk
        )

    fp32_width = 131072
    fp32_pattern = torch.tensor(
        [-3.0, -1.0, 0.0, 1.0], dtype=torch.float32, device="musa"
    )
    fp32_low_entropy = fp32_pattern.repeat(
        (16 * fp32_width + 3) // 4
    )[: 16 * fp32_width].reshape(16, fp32_width)
    _assert_indices_select_topk(fp32_low_entropy)
    _assert_indices_select_topk(
        torch.full(
            (16, fp32_width), 2.0, dtype=torch.float32, device="musa"
        ),
        index_dtype=torch.int64,
    )

    _assert_indices_select_topk(
        torch.full(
            (2, 131072), 2.0, dtype=torch.float32, device="musa"
        )
    )


def _run_native_topk_tiers() -> None:
    input = torch.randn(4, 16384, dtype=torch.bfloat16, device="musa")
    for index_dtype in (torch.int32, torch.int64):
        for topk in (1, 511, 512, 1024, 4096):
            output_alignment = 32 // index_dtype.itemsize
            output_stride = (
                (topk + output_alignment - 1)
                // output_alignment
                * output_alignment
            )
            output = torch.empty(
                4, output_stride, dtype=index_dtype, device="musa"
            )[:, :topk]
            assert deep_select_musa._can_use_fast_path(
                input,
                topk,
                None,
                False,
                False,
                output,
                None,
                False,
                False,
            )
            _assert_indices_select_topk(input, topk, index_dtype)

    pattern = torch.tensor(
        [-3.0, -1.0, 0.0, 1.0], dtype=torch.bfloat16, device="musa"
    )
    width = 131072
    low_entropy = pattern.repeat((2 * width + 3) // 4)[: 2 * width].reshape(
        2, width
    )
    _assert_indices_select_topk(low_entropy, 4096, torch.int32)
    _assert_indices_select_topk(low_entropy, 4096, torch.int64)


def _run_native_value_cases() -> None:
    input = torch.randn(4, 131072, dtype=torch.bfloat16, device="musa")
    for index_dtype in (torch.int32, torch.int64):
        values, indices = deep_select.topk(
            input,
            512,
            indices_type=index_dtype,
            return_value=True,
            abort_when_nan_found=False,
        )
        expected = torch.topk(input, 512, dim=1, sorted=True).values
        torch.testing.assert_close(
            torch.sort(values, dim=1, descending=True).values,
            expected,
            rtol=0,
            atol=0,
        )
        selected = torch.gather(input, 1, indices.to(torch.int64))
        torch.testing.assert_close(
            torch.sort(selected, dim=1, descending=True).values,
            expected,
            rtol=0,
            atol=0,
        )


def _run_native_sort_epilogue_cases() -> None:
    bf16 = torch.randn(4, 131072, dtype=torch.bfloat16, device="musa")
    _, sorted_index = deep_select.topk(
        bf16,
        512,
        sorted_index=True,
        indices_type=torch.int64,
        return_value=False,
        abort_when_nan_found=False,
    )
    assert torch.all(sorted_index[:, 1:] >= sorted_index[:, :-1]).item()

    fp32 = torch.randn(4, 131072, dtype=torch.float32, device="musa")
    values, indices = deep_select.topk(
        fp32,
        512,
        sorted=True,
        indices_type=torch.int32,
        return_value=True,
        abort_when_nan_found=False,
    )
    expected_values, expected_indices = torch.topk(
        fp32, 512, dim=1, sorted=True
    )
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.gather(fp32, 1, indices.to(torch.int64)),
        expected_values,
        rtol=0,
        atol=0,
    )

    large_batch_width = 16384
    fp32_large_batch = torch.zeros(
        4096, large_batch_width, dtype=torch.float32, device="musa"
    )
    fp32_large_batch[:, -512:] = torch.arange(
        1, 513, dtype=torch.float32, device="musa"
    )
    large_values, large_indices = deep_select.topk(
        fp32_large_batch,
        512,
        sorted=True,
        indices_type=torch.int32,
        return_value=True,
        abort_when_nan_found=False,
    )
    expected_large_values = torch.arange(
        512, 0, -1, dtype=torch.float32, device="musa"
    ).expand(4096, -1)
    expected_large_indices = torch.arange(
        large_batch_width - 1,
        large_batch_width - 513,
        -1,
        dtype=torch.int32,
        device="musa",
    ).expand(4096, -1)
    torch.testing.assert_close(
        large_values, expected_large_values, rtol=0, atol=0
    )
    torch.testing.assert_close(
        large_indices, expected_large_indices, rtol=0, atol=0
    )

    short = torch.randn(2, 1024, dtype=torch.float32, device="musa")
    short[0, 17] = float("-inf")
    end = torch.tensor([511, 700], dtype=torch.int32, device="musa")
    value_fill = 1234.0
    short_values, short_indices = deep_select.topk(
        short,
        512,
        sorted=True,
        end=end,
        indices_type=torch.int64,
        idx_oob_fill_value=123456789,
        value_oob_fill_value=value_fill,
        return_value=True,
        abort_when_nan_found=False,
    )
    for row, valid_len in enumerate(end.cpu().tolist()):
        selected_count = min(512, valid_len)
        expected = torch.topk(
            short[row, :valid_len], selected_count, sorted=True
        ).values
        torch.testing.assert_close(
            short_values[row, :selected_count], expected, rtol=0, atol=0
        )
        torch.testing.assert_close(
            torch.gather(
                short[row],
                0,
                short_indices[row, :selected_count].to(torch.int64),
            ),
            expected,
            rtol=0,
            atol=0,
        )
        assert torch.all(
            short_values[row, selected_count:] == value_fill
        ).item()

    wide_fp32 = torch.randn(4, 16384, dtype=torch.float32, device="musa")
    for topk in (1025, 2048, 2049, 4096):
        expected_vals, _ = torch.topk(wide_fp32, topk, dim=1, sorted=True)
        for indices_type in (torch.int32, torch.int64):
            vals, idxs = deep_select.topk(
                wide_fp32,
                topk,
                sorted=True,
                indices_type=indices_type,
                return_value=True,
                abort_when_nan_found=False,
            )
            torch.testing.assert_close(vals, expected_vals, rtol=0, atol=0)
            torch.testing.assert_close(
                torch.gather(wide_fp32, 1, idxs.to(torch.int64)),
                expected_vals,
                rtol=0,
                atol=0,
            )


def _check_fp32_two_stage(
    x: torch.Tensor,
    topk: int,
    *,
    sorted_value: bool,
    return_value: bool,
    indices_type: torch.dtype,
) -> None:
    vals, idxs = deep_select.topk(
        x,
        topk,
        sorted=sorted_value,
        indices_type=indices_type,
        return_value=return_value,
        abort_when_nan_found=False,
    )
    expected_vals = torch.topk(x, topk, dim=1, sorted=True).values
    ordered = torch.sort(idxs, dim=1).values
    assert torch.all(ordered[:, 1:] != ordered[:, :-1]).item()
    gathered = torch.gather(x, 1, idxs.to(torch.int64))
    if sorted_value:
        torch.testing.assert_close(vals, expected_vals, rtol=0, atol=0)
        torch.testing.assert_close(gathered, expected_vals, rtol=0, atol=0)
    else:
        torch.testing.assert_close(
            torch.sort(gathered, dim=1, descending=True).values,
            expected_vals,
            rtol=0,
            atol=0,
        )
        if return_value:
            torch.testing.assert_close(
                torch.sort(vals, dim=1, descending=True).values,
                expected_vals,
                rtol=0,
                atol=0,
            )
        else:
            assert vals is None


def _check_fp32_two_stage_modes(
    x: torch.Tensor, ks: tuple[int, ...]
) -> None:
    for k in ks:
        for sorted_value, return_value, indices_type in (
            (True, True, torch.int32),
            (False, False, torch.int64),
        ):
            _check_fp32_two_stage(
                x,
                k,
                sorted_value=sorted_value,
                return_value=return_value,
                indices_type=indices_type,
            )


@contextmanager
def _forbid_fp32_two_stage_kernel():
    real_module = musa_cpp._module()

    def _unexpected(*args, **kwargs):
        raise AssertionError("topk_fp32_two_stage should not be called")

    class _NoTwoStageModule:
        def __getattr__(self, name: str):
            if name == "topk_fp32_two_stage":
                return _unexpected
            return getattr(real_module, name)

    with unittest.mock.patch.object(
        musa_cpp, "_module", return_value=_NoTwoStageModule()
    ):
        yield


def _run_fp32_two_stage_cases() -> None:
    prior = os.environ.get("DEEP_SELECT_MUSA_FP32_TWO_STAGE")
    os.environ.pop("DEEP_SELECT_MUSA_FP32_TWO_STAGE", None)
    width = 262144
    try:
        default_off_x = torch.randn(
            6, width, dtype=torch.float32, device="musa"
        )
        with _forbid_fp32_two_stage_kernel():
            _check_fp32_two_stage_modes(default_off_x, (128, 512, 1024, 2048))
        del default_off_x

        os.environ["DEEP_SELECT_MUSA_FP32_TWO_STAGE"] = "1"
        for rows in (1, 6, 16):
            x = torch.randn(rows, width, dtype=torch.float32, device="musa")
            for k in (128, 512, 1024, 2048):
                for indices_type in (torch.int32, torch.int64):
                    for sorted_value, return_value in (
                        (False, False),
                        (False, True),
                        (True, True),
                    ):
                        _check_fp32_two_stage(
                            x,
                            k,
                            sorted_value=sorted_value,
                            return_value=return_value,
                            indices_type=indices_type,
                        )
            del x

        wide_x = torch.randn(6, 524288, dtype=torch.float32, device="musa")
        _check_fp32_two_stage_modes(wide_x, (512, 2048))
        del wide_x

        all_equal = torch.full((6, width), 2.0, dtype=torch.float32, device="musa")
        _check_fp32_two_stage_modes(all_equal, (512, 1024, 2048))
        del all_equal

        ties = torch.full((6, width), -1.0, dtype=torch.float32, device="musa")
        ties[:, :200] = 5.0
        ties[:, 200:5000] = 1.0
        _check_fp32_two_stage_modes(ties, (512, 1024, 2048))
        del ties

        adversarial = torch.zeros(6, width, dtype=torch.float32, device="musa")
        adversarial[:, 1:2001:2] = torch.linspace(
            1.0, 2.0, 1000, dtype=torch.float32, device="musa"
        )
        _check_fp32_two_stage_modes(adversarial, (128, 512, 1024, 2048))
        del adversarial

        periodic = torch.zeros(6, width, dtype=torch.float32, device="musa")
        step = width // 2048
        periodic[:, 1::step] = torch.linspace(
            1.0, 2.0, width // step, dtype=torch.float32, device="musa"
        )
        periodic[:, 2::step] = torch.linspace(
            2.0, 3.0, width // step, dtype=torch.float32, device="musa"
        )
        periodic[:, 3::step] = torch.linspace(
            3.0, 4.0, width // step, dtype=torch.float32, device="musa"
        )
        _check_fp32_two_stage_modes(periodic, (128, 512, 1024, 2048))
        del periodic

        nan_x = torch.randn(6, width, dtype=torch.float32, device="musa")
        nan_x[:, 17] = float("nan")
        for k in (512, 1024, 2048):
            for sorted_value, return_value in ((False, False), (True, True)):
                for indices_type in (torch.int32, torch.int64):
                    _, nan_idxs = deep_select.topk(
                        nan_x,
                        k,
                        sorted=sorted_value,
                        indices_type=indices_type,
                        return_value=return_value,
                        abort_when_nan_found=False,
                    )
                    assert torch.all(nan_idxs[:, 0] == 0x3F3F3F3F).item()
        del nan_x

        for rows, k in ((6, 512), (16, 1024), (6, 2048)):
            gx = torch.randn(rows, width, dtype=torch.float32, device="musa")
            gout = torch.empty(rows, k, dtype=torch.int32, device="musa")
            captured: list[tuple[torch.Tensor | None, torch.Tensor]] = []

            def run_graph() -> None:
                captured.clear()
                captured.append(
                    deep_select.topk(
                        gx,
                        k,
                        sorted=True,
                        indices_type=torch.int32,
                        return_value=True,
                        abort_when_nan_found=False,
                    )
                )
                deep_select.topk(
                    gx,
                    k,
                    output_idx=gout,
                    indices_type=torch.int32,
                    return_value=False,
                    abort_when_nan_found=False,
                )

            stream = torch.musa.Stream()
            with torch.musa.stream(stream):
                run_graph()
            stream.synchronize()
            graph = torch.musa.MUSAGraph()
            with torch.musa.graph(graph, stream=stream):
                run_graph()
            gx.normal_()
            graph.replay()
            torch.musa.synchronize()
            gvals, gidxs = captured[0]
            expected_vals = torch.topk(gx, k, dim=1, sorted=True).values
            torch.testing.assert_close(gvals, expected_vals, rtol=0, atol=0)
            torch.testing.assert_close(
                torch.gather(gx, 1, gidxs.to(torch.int64)),
                expected_vals,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                torch.sort(
                    torch.gather(gx, 1, gout.to(torch.int64)),
                    dim=1,
                    descending=True,
                ).values,
                expected_vals,
                rtol=0,
                atol=0,
            )

        mp32_x = torch.randn(6, width, dtype=torch.float32, device="musa")
        with (
            unittest.mock.patch.object(
                musa_cpp, "_device_capability", return_value=(3, 2)
            ),
            _forbid_fp32_two_stage_kernel(),
        ):
            _check_fp32_two_stage_modes(mp32_x, (128, 512, 1024, 2048))
        del mp32_x

        os.environ["DEEP_SELECT_MUSA_FP32_TWO_STAGE"] = "0"
        opt_out_x = torch.randn(6, width, dtype=torch.float32, device="musa")
        with _forbid_fp32_two_stage_kernel():
            _check_fp32_two_stage_modes(opt_out_x, (128, 512, 1024, 2048))
        del opt_out_x
    finally:
        if prior is None:
            os.environ.pop("DEEP_SELECT_MUSA_FP32_TWO_STAGE", None)
        else:
            os.environ["DEEP_SELECT_MUSA_FP32_TWO_STAGE"] = prior


def _run_native_sample_survivor_cases() -> None:
    batch = 4096
    width = 129280
    topk = 512
    all_equal = torch.full(
        (batch, width), 2.0, dtype=torch.float32, device="musa"
    )
    equal_values, equal_indices = deep_select.topk(
        all_equal,
        topk,
        sorted=True,
        indices_type=torch.int64,
        return_value=True,
        abort_when_nan_found=False,
    )
    assert torch.all(equal_values == 2.0).item()
    torch.testing.assert_close(
        torch.gather(all_equal, 1, equal_indices),
        equal_values,
        rtol=0,
        atol=0,
    )
    ordered = torch.sort(equal_indices, dim=1).values
    assert torch.all(ordered[:, 1:] != ordered[:, :-1]).item()
    del all_equal, equal_values, equal_indices, ordered

    shifted = torch.zeros(
        batch, width, dtype=torch.float32, device="musa"
    )
    shifted[:, :16384] = -100.0
    shifted[:, -topk:] = torch.arange(
        1, topk + 1, dtype=torch.float32, device="musa"
    )
    shifted_values, shifted_indices = deep_select.topk(
        shifted,
        topk,
        sorted=True,
        indices_type=torch.int64,
        return_value=True,
        abort_when_nan_found=False,
    )
    expected_values = torch.arange(
        topk, 0, -1, dtype=torch.float32, device="musa"
    ).expand(batch, -1)
    expected_indices = torch.arange(
        width - 1,
        width - topk - 1,
        -1,
        dtype=torch.int64,
        device="musa",
    ).expand(batch, -1)
    torch.testing.assert_close(
        shifted_values, expected_values, rtol=0, atol=0
    )
    torch.testing.assert_close(
        shifted_indices, expected_indices, rtol=0, atol=0
    )

    shifted[:, 0] = float("nan")
    _, nan_indices = deep_select.topk(
        shifted,
        topk,
        sorted=True,
        indices_type=torch.int64,
        return_value=True,
        abort_when_nan_found=False,
    )
    assert torch.all(nan_indices[:, 0] == 0x3F3F3F3F).item()

    shifted[:, 0] = -100.0
    graph_output = torch.empty(
        batch, topk, dtype=torch.int64, device="musa"
    )

    def run_sample_graph() -> None:
        deep_select.topk(
            shifted,
            topk,
            output_idx=graph_output,
            indices_type=torch.int64,
            return_value=False,
            abort_when_nan_found=False,
        )

    capture_stream = torch.musa.Stream()
    with torch.musa.stream(capture_stream):
        run_sample_graph()
    capture_stream.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=capture_stream):
        run_sample_graph()
    graph.replay()
    torch.musa.synchronize()
    selected = torch.gather(shifted, 1, graph_output)
    torch.testing.assert_close(
        torch.sort(selected, dim=1, descending=True).values,
        expected_values,
        rtol=0,
        atol=0,
    )


def _run_native_bf16_sample_survivor_cases() -> None:
    batch = 4096
    width = 1048576
    topk = 512
    period = torch.linspace(
        -10.0, -1.0, 16384, dtype=torch.float32, device="musa"
    ).to(torch.bfloat16)
    base = period.repeat(width // period.numel())
    input = base.expand(batch, -1).clone()
    expected = torch.topk(base, topk, sorted=True).values.expand(batch, -1)

    values, indices = deep_select.topk(
        input,
        topk,
        indices_type=torch.int64,
        return_value=True,
        abort_when_nan_found=False,
    )
    selected = torch.gather(input, 1, indices)
    torch.testing.assert_close(
        torch.sort(values, dim=1, descending=True).values,
        expected,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.sort(selected, dim=1, descending=True).values,
        expected,
        rtol=0,
        atol=0,
    )
    ordered = torch.sort(indices, dim=1).values
    assert torch.all(ordered[:, 1:] != ordered[:, :-1]).item()
    del values, indices, selected, ordered

    input.fill_(2.0)
    _, equal_indices = deep_select.topk(
        input,
        topk,
        indices_type=torch.int32,
        return_value=False,
        abort_when_nan_found=False,
    )
    assert torch.all(equal_indices >= 0).item()
    assert torch.all(equal_indices < width).item()
    ordered = torch.sort(equal_indices, dim=1).values
    assert torch.all(ordered[:, 1:] != ordered[:, :-1]).item()
    assert torch.all(
        torch.gather(input, 1, equal_indices.to(torch.int64)) == 2.0
    ).item()

    row_offsets = (
        torch.arange(batch, dtype=torch.int64, device="musa") * width
    )
    flat_input = input.view(-1)
    for nan_position in (0, width // 2):
        nan_offsets = row_offsets + nan_position
        flat_input.index_fill_(0, nan_offsets, float("nan"))
        _, nan_indices = deep_select.topk(
            input,
            topk,
            indices_type=torch.int32,
            return_value=False,
            abort_when_nan_found=False,
        )
        assert torch.all(nan_indices[:, 0] == 0x3F3F3F3F).item()
        flat_input.index_fill_(0, nan_offsets, 2.0)

    input.copy_(base)
    graph_output = torch.empty(
        batch, topk, dtype=torch.int32, device="musa"
    )

    def run_bf16_sample_graph() -> None:
        deep_select.topk(
            input,
            topk,
            output_idx=graph_output,
            indices_type=torch.int32,
            return_value=False,
            abort_when_nan_found=False,
        )

    capture_stream = torch.musa.Stream()
    with torch.musa.stream(capture_stream):
        run_bf16_sample_graph()
    capture_stream.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=capture_stream):
        run_bf16_sample_graph()
    graph.replay()
    torch.musa.synchronize()
    selected = torch.gather(input, 1, graph_output.to(torch.int64))
    torch.testing.assert_close(
        torch.sort(selected, dim=1, descending=True).values,
        expected,
        rtol=0,
        atol=0,
    )


def _run_native_offset_nan_cases() -> None:
    input = torch.randn(4, 131072, dtype=torch.bfloat16, device="musa")
    offsets = torch.tensor([3, 7, 11, 13], dtype=torch.int32, device="musa")
    _, indices = deep_select.topk(
        input,
        512,
        indices_type=torch.int32,
        output_idx_offset=offsets,
        return_value=False,
        abort_when_nan_found=False,
    )
    unshifted = indices.to(torch.int64) - offsets.to(torch.int64)[:, None]
    selected = torch.gather(input, 1, unshifted)
    expected = torch.topk(input, 512, dim=1, sorted=True).values
    torch.testing.assert_close(
        torch.sort(selected, dim=1, descending=True).values,
        expected,
        rtol=0,
        atol=0,
    )

def _run_native_end_cases() -> None:
    batch, width, topk = 4, 131072, 512
    end = torch.tensor([0, 511, 512, 1000], dtype=torch.int32, device="musa")
    offsets = torch.tensor([3, 7, 11, 13], dtype=torch.int32, device="musa")
    idx_fill = 123456789
    value_fill = -123.0
    for dtype in (torch.bfloat16, torch.float32):
        input = torch.randn(
            batch, width + 16, dtype=dtype, device="musa"
        )[:, :width]
        for indices_type in (torch.int32, torch.int64):
            values, indices = deep_select.topk(
                input,
                topk,
                end=end,
                indices_type=indices_type,
                sorted_index=True,
                output_idx_offset=offsets,
                idx_oob_fill_value=idx_fill,
                value_oob_fill_value=value_fill,
                return_value=True,
                abort_when_nan_found=False,
            )
            for row, valid_len in enumerate(end.cpu().tolist()):
                selected_count = min(topk, valid_len)
                if selected_count:
                    unshifted = (
                        indices[row, :selected_count].to(torch.int64)
                        - offsets[row].to(torch.int64)
                    )
                    expected = torch.topk(
                        input[row, :valid_len], selected_count, sorted=True
                    ).values
                    actual = torch.sort(
                        torch.gather(input[row], 0, unshifted),
                        descending=True,
                    ).values
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        values[row, :selected_count],
                        torch.gather(input[row], 0, unshifted),
                        rtol=0,
                        atol=0,
                    )
                    assert torch.all(
                        indices[row, 1:selected_count]
                        >= indices[row, : selected_count - 1]
                    ).item()
                assert torch.all(
                    indices[row, selected_count:] == idx_fill
                ).item()
                assert torch.all(
                    values[row, selected_count:] == value_fill
                ).item()


def _run_native_shortcut_cases() -> None:
    batch, width, topk = 3, 257, 511
    offsets = torch.tensor([3, 7, 11], dtype=torch.int32, device="musa")
    idx_fill = 123456789
    value_fill = -123.0
    for dtype in (torch.bfloat16, torch.float32):
        input = torch.randn(
            batch, 272, dtype=dtype, device="musa"
        )[:, :width]
        for indices_type in (torch.int32, torch.int64):
            values, indices = deep_select.topk(
                input,
                topk,
                indices_type=indices_type,
                output_idx_offset=offsets,
                idx_oob_fill_value=idx_fill,
                value_oob_fill_value=value_fill,
                return_value=True,
                abort_when_nan_found=True,
            )
            assert deep_select_musa._can_use_fast_path(
                input,
                topk,
                None,
                False,
                False,
                indices,
                offsets,
                True,
                True,
            )
            expected_indices = torch.arange(
                width, dtype=torch.int64, device="musa"
            ).unsqueeze(0) + offsets.to(torch.int64).unsqueeze(1)
            torch.testing.assert_close(
                indices[:, :width].to(torch.int64),
                expected_indices,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                values[:, :width], input, rtol=0, atol=0
            )
            assert torch.all(indices[:, width:] == idx_fill).item()
            assert torch.all(values[:, width:] == value_fill).item()


def _run_native_trap_subprocesses() -> None:
    probe = """
import sys
import torch
import torch_musa
import deep_select
from deep_select import deep_select_musa

dtype = getattr(torch, sys.argv[1])
batch = int(sys.argv[2])
width = int(sys.argv[3])
x = torch.randn(batch, width, dtype=dtype, device="musa")
x[0, 17] = float("nan")
output = torch.empty(batch, 512, dtype=torch.int32, device="musa")
assert deep_select_musa._can_use_fast_path(
    x, 512, None, False, False, output, None, False, True
)
print("FAST_PATH=1", flush=True)
deep_select.topk(
    x,
    512,
    indices_type=torch.int32,
    return_value=False,
    abort_when_nan_found=True,
)
torch.musa.synchronize()
print("NO_DEVICE_TRAP", flush=True)
"""
    cases = (
        ("bf16-single", "bfloat16", 4, 32768, {}),
        ("bf16-multi", "bfloat16", 1, 131072, {}),
        ("fp32-single", "float32", 4, 32768, {}),
        (
            "fp32-two-stage",
            "float32",
            2,
            262144,
            {"DEEP_SELECT_MUSA_FP32_TWO_STAGE": "1"},
        ),
    )
    for name, dtype, batch, width, extra_env in cases:
        env = os.environ.copy()
        env.update(extra_env)
        completed = subprocess.run(
            [sys.executable, "-c", probe, dtype, str(batch), str(width)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        assert "FAST_PATH=1" in completed.stdout, (name, completed.stdout)
        assert "NO_DEVICE_TRAP" not in completed.stdout, (name, completed.stdout)
        assert completed.returncode != 0, (name, completed.stdout)
        print(
            f"native NaN trap passed: {name}, returncode={completed.returncode}"
        )


def _run_output_and_dispatch_cases() -> None:
    input = torch.randn(4, 32768, dtype=torch.bfloat16, device="musa")
    output = torch.empty(4, 512, dtype=torch.int32, device="musa")
    values, indices = deep_select.topk(
        input,
        512,
        output_idx=output,
        indices_type=torch.int32,
        return_value=False,
        abort_when_nan_found=False,
    )
    assert values is None
    assert indices.data_ptr() == output.data_ptr()
    selected = torch.gather(input, 1, indices.to(torch.int64))
    expected = torch.topk(input, 512, dim=1, sorted=True).values
    torch.testing.assert_close(
        torch.sort(selected, dim=1, descending=True).values,
        expected,
        rtol=0,
        atol=0,
    )
    assert deep_select_musa._can_use_fast_path(
        input,
        512,
        None,
        False,
        False,
        output,
        None,
        False,
        False,
    )

    wide = torch.randn(1, 204800, dtype=torch.float32, device="musa")
    wide_output = torch.empty(1, 512, dtype=torch.int32, device="musa")
    assert deep_select_musa._can_use_fast_path(
        wide,
        512,
        None,
        False,
        False,
        wide_output,
        None,
        False,
        False,
    )


def _run_boundary_shapes() -> None:
    cases = [
        _case(
            4096,
            4096,
            512,
            torch.bfloat16,
            torch.int32,
            return_value=False,
        ),
        _case(
            1,
            (1 << 23) - 1,
            512,
            torch.bfloat16,
            torch.int32,
            return_value=False,
        ),
        _case(
            1,
            (1 << 23) - 1,
            512,
            torch.float32,
            torch.int32,
            return_value=False,
        ),
    ]
    for case in cases:
        assert run_testcase(case, quiet=True), case


def _run_graph_shape(batch: int, width: int) -> None:
    input = torch.randn(batch, width, dtype=torch.bfloat16, device="musa")
    output = torch.empty(batch, 512, dtype=torch.int32, device="musa")
    torch.musa.synchronize()

    def run() -> None:
        deep_select.topk(
            input,
            512,
            output_idx=output,
            indices_type=torch.int32,
            return_value=False,
            abort_when_nan_found=False,
        )

    capture_stream = torch.musa.Stream()
    with torch.musa.stream(capture_stream):
        run()
    capture_stream.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=capture_stream):
        run()
    graph.replay()
    torch.musa.synchronize()

    selected = torch.gather(input, 1, output.to(torch.int64))
    expected = torch.topk(input, 512, dim=1, sorted=True).values
    torch.testing.assert_close(
        torch.sort(selected, dim=1, descending=True).values,
        expected,
        rtol=0,
        atol=0,
    )


def _run_graph_case() -> None:
    _run_graph_shape(8, 32768)
    _run_graph_shape(6, 1048576)

    fp32 = torch.randn(6, 262144, dtype=torch.float32, device="musa")
    fp32_output = torch.empty(6, 512, dtype=torch.int64, device="musa")

    def run_fp32() -> None:
        deep_select.topk(
            fp32,
            512,
            output_idx=fp32_output,
            indices_type=torch.int64,
            return_value=False,
            abort_when_nan_found=False,
        )

    fp32_capture_stream = torch.musa.Stream()
    with torch.musa.stream(fp32_capture_stream):
        run_fp32()
    fp32_capture_stream.synchronize()
    fp32_graph = torch.musa.MUSAGraph()
    with torch.musa.graph(fp32_graph, stream=fp32_capture_stream):
        run_fp32()
    fp32_graph.replay()
    torch.musa.synchronize()
    fp32_selected = torch.gather(fp32, 1, fp32_output)
    fp32_expected = torch.topk(fp32, 512, dim=1, sorted=True).values
    torch.testing.assert_close(
        torch.sort(fp32_selected, dim=1, descending=True).values,
        fp32_expected,
        rtol=0,
        atol=0,
    )

    batch, width, topk = 8, 32768, 512
    input = torch.randn(batch, width, dtype=torch.bfloat16, device="musa")
    output = torch.empty(batch, topk, dtype=torch.int64, device="musa")
    end = torch.tensor(
        [0, 1, 511, 512, 513, 1000, 16384, width],
        dtype=torch.int32,
        device="musa",
    )
    offsets = torch.tensor(
        [3, 5, 7, 11, 13, 17, 19, 23],
        dtype=torch.int32,
        device="musa",
    )
    idx_fill = 123456789

    def run_end_offset() -> None:
        deep_select.topk(
            input,
            topk,
            end=end,
            indices_type=torch.int64,
            sorted_index=True,
            output_idx=output,
            output_idx_offset=offsets,
            idx_oob_fill_value=idx_fill,
            return_value=False,
            abort_when_nan_found=False,
        )

    capture_stream = torch.musa.Stream()
    with torch.musa.stream(capture_stream):
        run_end_offset()
    capture_stream.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=capture_stream):
        run_end_offset()
    graph.replay()
    torch.musa.synchronize()

    for row, valid_len in enumerate(end.cpu().tolist()):
        selected_count = min(topk, valid_len)
        if selected_count:
            unshifted = (
                output[row, :selected_count] - offsets[row].to(torch.int64)
            )
            actual = torch.sort(
                torch.gather(input[row], 0, unshifted), descending=True
            ).values
            expected = torch.topk(
                input[row, :valid_len], selected_count, sorted=True
            ).values
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.all(
                output[row, 1:selected_count]
                >= output[row, : selected_count - 1]
            ).item()
        assert torch.all(output[row, selected_count:] == idx_fill).item()


def _run_multi_stream_case() -> None:
    width = 262144
    inputs = [
        torch.randn(2, width, dtype=torch.bfloat16, device="musa")
        for _ in range(2)
    ]
    outputs = [
        torch.empty(2, 512, dtype=torch.int32, device="musa") for _ in range(2)
    ]
    streams = [torch.musa.Stream(), torch.musa.Stream()]

    for _ in range(5):
        for input, output, stream in zip(inputs, outputs, streams):
            with torch.musa.stream(stream):
                deep_select.topk(
                    input,
                    512,
                    output_idx=output,
                    indices_type=torch.int32,
                    return_value=False,
                    abort_when_nan_found=False,
                )
    for stream in streams:
        stream.synchronize()
    for input, output in zip(inputs, outputs):
        selected = torch.gather(input, 1, output.to(torch.int64))
        expected = torch.topk(input, 512, dim=1, sorted=True).values
        torch.testing.assert_close(
            torch.sort(selected, dim=1, descending=True).values,
            expected,
            rtol=0,
            atol=0,
        )

    fp32_inputs = [
        torch.randn(2, width, dtype=torch.float32, device="musa")
        for _ in range(2)
    ]
    fp32_outputs = [
        torch.empty(2, 512, dtype=torch.int64, device="musa")
        for _ in range(2)
    ]
    fp32_streams = [torch.musa.Stream(), torch.musa.Stream()]
    for _ in range(5):
        for input, output, stream in zip(
            fp32_inputs, fp32_outputs, fp32_streams
        ):
            with torch.musa.stream(stream):
                deep_select.topk(
                    input,
                    512,
                    output_idx=output,
                    indices_type=torch.int64,
                    return_value=False,
                    abort_when_nan_found=False,
                )
    for stream in fp32_streams:
        stream.synchronize()
    for input, output in zip(fp32_inputs, fp32_outputs):
        selected = torch.gather(input, 1, output)
        expected = torch.topk(input, 512, dim=1, sorted=True).values
        torch.testing.assert_close(
            torch.sort(selected, dim=1, descending=True).values,
            expected,
            rtol=0,
            atol=0,
        )


def _run_kernelkit_profile_case() -> None:
    input = torch.randn(4, 32768, dtype=torch.bfloat16, device="musa")
    output = torch.empty(4, 512, dtype=torch.int32, device="musa")

    def run() -> None:
        deep_select.topk(
            input,
            512,
            output_idx=output,
            indices_type=torch.int32,
            return_value=False,
            abort_when_nan_found=False,
        )

    result = kk.bench(run, num_tests=3, flush_l2=False)
    kernel_names = result.get_kernel_names()
    assert any(
        "topk_bf16_kernel" in name
        or "topk_bf16_compact12_kernel" in name
        or "topk_bf16_persist_kernel" in name
        for name in kernel_names
    ), kernel_names

    bf16_large_batch = torch.randn(
        32, 131072, dtype=torch.bfloat16, device="musa"
    )
    bf16_large_output = torch.empty(
        32, 512, dtype=torch.int32, device="musa"
    )

    def run_bf16_large_batch() -> None:
        deep_select.topk(
            bf16_large_batch,
            512,
            output_idx=bf16_large_output,
            indices_type=torch.int32,
            return_value=False,
            abort_when_nan_found=False,
        )

    bf16_large_result = kk.bench(
        run_bf16_large_batch, num_tests=3, flush_l2=False
    )
    bf16_large_kernel_names = bf16_large_result.get_kernel_names()
    assert any(
        "topk_bf16_compact12_kernel" in name
        for name in bf16_large_kernel_names
    ), bf16_large_kernel_names

    fp32 = torch.randn(6, 262144, dtype=torch.float32, device="musa")
    fp32_output = torch.empty(6, 512, dtype=torch.int64, device="musa")

    def run_fp32() -> None:
        deep_select.topk(
            fp32,
            512,
            output_idx=fp32_output,
            indices_type=torch.int64,
            return_value=False,
            abort_when_nan_found=False,
        )

    fp32_result = kk.bench(run_fp32, num_tests=3, flush_l2=False)
    fp32_kernel_names = fp32_result.get_kernel_names()
    assert any(
        "partial_fp32_histogram_kernel" in name
        or "topk_fp32_compact12_kernel" in name
        or "topk_fp32_scan_partitions" in name
        for name in fp32_kernel_names
    ), fp32_kernel_names
    assert any(
        "finish_fp32_multi_kernel" in name
        or "topk_fp32_compact12_kernel" in name
        or "topk_fp32_finish_partitions" in name
        for name in fp32_kernel_names
    ), fp32_kernel_names

    fp32_large_batch = torch.randn(
        32, 131072, dtype=torch.float32, device="musa"
    )
    fp32_large_output = torch.empty(
        32, 512, dtype=torch.int64, device="musa"
    )

    def run_fp32_large_batch() -> None:
        deep_select.topk(
            fp32_large_batch,
            512,
            output_idx=fp32_large_output,
            indices_type=torch.int64,
            return_value=False,
            abort_when_nan_found=False,
        )

    fp32_large_result = kk.bench(
        run_fp32_large_batch, num_tests=3, flush_l2=False
    )
    fp32_large_kernel_names = fp32_large_result.get_kernel_names()
    assert any(
        "topk_fp32_compact12_kernel" in name
        for name in fp32_large_kernel_names
    ), fp32_large_kernel_names


def main() -> None:
    torch.set_default_device("musa")
    assert deep_select.get_stride_requirement() == (32, 32)
    _run_semantic_matrix()
    _run_nan_cases()
    _run_native_cpp_adversarial_cases()
    _run_native_topk_tiers()
    _run_native_value_cases()
    _run_native_sort_epilogue_cases()
    _run_fp32_two_stage_cases()
    _run_native_sample_survivor_cases()
    _run_native_bf16_sample_survivor_cases()
    _run_native_offset_nan_cases()
    _run_native_end_cases()
    _run_native_shortcut_cases()
    _run_output_and_dispatch_cases()
    _run_boundary_shapes()
    _run_graph_case()
    _run_multi_stream_case()
    _run_kernelkit_profile_case()
    _run_native_trap_subprocesses()
    print("MUSA backend public API matrix passed")


if __name__ == "__main__":
    main()
