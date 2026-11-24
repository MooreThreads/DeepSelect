#include <torch/extension.h>
#include <torch/library.h>

#include <cstdint>
#include <tuple>

#define DEEP_SELECT_DISABLE_PYBIND
#include "musa_kernels/topk.mu"

namespace {

constexpr int64_t kAlignmentBytes = 32;

void check_row_alignment(const char* name, const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.storage_offset() * tensor.element_size() % kAlignmentBytes == 0,
      name, ".data_ptr() must be aligned to ", kAlignmentBytes, " bytes");
  TORCH_CHECK(
      tensor.stride(0) * tensor.element_size() % kAlignmentBytes == 0,
      name, ".stride(0) must be aligned to ", kAlignmentBytes, " bytes");
}

void topk(
    const at::Tensor& input, int64_t topk,
    const c10::optional<at::Tensor>& begin,
    const c10::optional<at::Tensor>& end, bool sorted_value, bool sorted_index,
    const c10::optional<at::Tensor>& output_value,
    const at::Tensor& output_index,
    const c10::optional<at::Tensor>& output_idx_offset,
    int64_t idx_oob_fill_value, double value_oob_fill_value, bool return_value,
    bool abort_when_nan_found) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
              "input must be a MUSA tensor");
  TORCH_CHECK(input.dim() == 2 && input.stride(1) == 1,
              "input must be a contiguous-last-dimension 2D tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kBFloat16,
              "input must be float32 or bfloat16");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) < (1 << 23),
              "input width must be in [1, 2^23)");
  TORCH_CHECK(!begin.has_value(), "begin is not supported");
  TORCH_CHECK(topk > 0 && topk <= 4096,
              "topk must be in [1, 4096]");
  TORCH_CHECK(!sorted_value || (return_value && !sorted_index),
              "sorted_value requires return_value and excludes sorted_index");
  TORCH_CHECK(!sorted_value || input.scalar_type() == at::kFloat,
              "sorted_value requires float32 input");
  TORCH_CHECK(
      return_value ? output_value.has_value() : !output_value.has_value(),
      return_value ? "output_value must be provided when return_value is true"
                   : "output_value must be omitted when return_value is false");
  TORCH_CHECK(output_index.device() == input.device() &&
                  (output_index.scalar_type() == at::kInt ||
                   output_index.scalar_type() == at::kLong) &&
                  output_index.dim() == 2 &&
                  output_index.stride(1) == 1 &&
                  output_index.size(0) >= input.size(0) &&
                  output_index.size(1) >= topk,
              "output_index must be an int32 or int64 MUSA tensor with at least "
              "[input.size(0), topk] elements");
  TORCH_CHECK(!end.has_value() ||
                  (end->device() == input.device() &&
                   end->scalar_type() == at::kInt && end->dim() == 1 &&
                   end->size(0) == input.size(0) && end->is_contiguous()),
              "end must be a contiguous int32 MUSA tensor with one value per row");
  check_row_alignment("input", input);
  check_row_alignment("output_index", output_index);

  if (output_idx_offset.has_value()) {
    TORCH_CHECK(output_idx_offset->device() == input.device() &&
                    output_idx_offset->scalar_type() == at::kInt &&
                    output_idx_offset->dim() == 1 &&
                    output_idx_offset->size(0) == input.size(0) &&
                    output_idx_offset->is_contiguous(),
                "output_idx_offset must be contiguous int32 [batch] on input.device");
  }
  at::Tensor output = output_index.narrow(0, 0, input.size(0)).narrow(1, 0, topk);
  c10::optional<at::Tensor> values = c10::nullopt;
  if (output_value.has_value()) {
    TORCH_CHECK(output_value->dim() == 2 &&
                    output_value->size(0) >= input.size(0) &&
                    output_value->size(1) >= topk,
                "output_value must have at least [input.size(0), topk] elements");
    TORCH_CHECK(output_value->device() == input.device() &&
                    output_value->scalar_type() == input.scalar_type() &&
                    output_value->stride(1) == 1,
                "output_value must match input device/dtype and be contiguous-last-dimension");
    check_row_alignment("output_value", *output_value);
    if (return_value) {
      values = output_value->narrow(0, 0, input.size(0)).narrow(1, 0, topk);
    }
  }

  bool sorted_in_kernel = false;
  if (input.scalar_type() == at::kFloat) {
    sorted_in_kernel = launch_topk_fp32(
        input, output, values, end, idx_oob_fill_value, value_oob_fill_value,
        abort_when_nan_found, sorted_value);
  } else {
    launch_topk_bf16(input, output, values, end, idx_oob_fill_value,
                     value_oob_fill_value, abort_when_nan_found);
  }
  if ((sorted_value || sorted_index) && !sorted_in_kernel) {
    launch_sort_output(output, values, end, input.size(1), sorted_value);
  }
  if (output_idx_offset.has_value()) {
    launch_output_idx_offset(output, end, *output_idx_offset, input.size(1));
  }
}

std::tuple<int64_t, int64_t> get_alignment_requirement() {
  return {kAlignmentBytes, kAlignmentBytes};
}

void meta_topk(
    const at::Tensor&, int64_t, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, bool, bool,
    const c10::optional<at::Tensor>&, const at::Tensor&,
    const c10::optional<at::Tensor>&, int64_t, double, bool, bool) {}

}

TORCH_LIBRARY(deep_select, m) {
  m.def(
      "topk(Tensor input, int topk, Tensor? begin, Tensor? end, "
      "bool sorted_value, bool sorted_index, "
      "Tensor(a!)? output_value, Tensor(b!) output_index, "
      "Tensor? output_idx_offset, int idx_oob_fill_value, "
      "float value_oob_fill_value, bool return_value, "
      "bool abort_when_nan_found) -> ()");
  m.def("get_alignment_requirement() -> (int, int)");
}

TORCH_LIBRARY_IMPL(deep_select, PrivateUse1, m) {
  m.impl("topk", &topk);
}

TORCH_LIBRARY_IMPL(deep_select, Meta, m) {
  m.impl("topk", &meta_topk);
}

TORCH_LIBRARY_IMPL(deep_select, CompositeExplicitAutograd, m) {
  m.impl("get_alignment_requirement", &get_alignment_requirement);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
