#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <musa_bf16.h>
#include <musa_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <torch/extension.h>
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <mutlass/arch/barrier.hpp>
#include <mute/tensor.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

constexpr int kMaxTopK = 4096;
constexpr int kRadix = 256;
constexpr int kNanSentinel = 0x3F3F3F3F;
constexpr int kSampleFallbackSentinel = -1;

__device__ __forceinline__ void abort_for_nan() {
  __assert_fail("DeepSelect NaN found in valid input range", __FILE__,
                __LINE__, __func__);
}

template <typename OutIndexT>
__global__ void nan_guard_kernel(
    const OutIndexT* __restrict__ output, int rows, int64_t output_stride) {
  const int row_index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row_index < rows &&
      output[static_cast<int64_t>(row_index) * output_stride] ==
          static_cast<OutIndexT>(kNanSentinel))
    abort_for_nan();
}

template <typename OutIndexT>
void launch_nan_guard(
    const OutIndexT* output, int rows, int64_t output_stride,
    musaStream_t stream) {
  constexpr int kGuardThreads = 128;
  nan_guard_kernel<OutIndexT>
      <<<(rows + kGuardThreads - 1) / kGuardThreads,
         kGuardThreads, 0, stream>>>(output, rows, output_stride);
}

union Bf16Bits {
  __mt_bfloat16 value;
  uint16_t bits;
};

struct __align__(16) Bf16x8 {
  union {
    uint4 storage;
    __mt_bfloat16 values[8];
  };
};

struct __align__(16) Floatx4 {
  union {
    float4 storage;
    float values[4];
  };
};

__device__ __forceinline__ Bf16x8 load_bf16x8(
    const __mt_bfloat16* address) {
  return *reinterpret_cast<const Bf16x8*>(address);
}

__device__ __forceinline__ Floatx4 load_floatx4(const float* address) {
  return *reinterpret_cast<const Floatx4*>(address);
}

__device__ __forceinline__ uint16_t raw_bits(__mt_bfloat16 value) {
  Bf16Bits converted;
  converted.value = value;
  return converted.bits;
}

__device__ __forceinline__ uint16_t ordered_key_from_bits(uint16_t bits) {
  return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                          : static_cast<uint16_t>(bits | 0x8000u);
}

__device__ __forceinline__ uint16_t ordered_key(__mt_bfloat16 value) {
  return ordered_key_from_bits(raw_bits(value));
}

__device__ __forceinline__ bool is_nan_key(uint16_t raw_bits) {
  return (raw_bits & 0x7F80u) == 0x7F80u && (raw_bits & 0x007Fu) != 0;
}

template <typename OutIndexT, typename ValueT>
__device__ __forceinline__ void store_topk_result(
    OutIndexT* indices, ValueT* values, bool return_value, int position,
    int64_t index, ValueT value) {
  indices[position] = static_cast<OutIndexT>(index);
  if (return_value)
    values[static_cast<int64_t>(position)] = value;
}

__device__ __host__ __forceinline__ constexpr bool compact12_topk_ok(int topk) {
  return topk <= 1024 || topk >= 4096;
}

__device__ __forceinline__ void warp_select(
    int* histogram, int wanted, int* threshold, int* remain) {
  if ((static_cast<int>(threadIdx.x) >> 5) != 0) return;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  int bins[8];
  int lane_sum = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int index = lane * 8 + j;
    bins[j] = histogram[index];
    histogram[index] = 0;
    lane_sum += bins[j];
  }

  int prefix = lane_sum;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(0xFFFFFFFFu, prefix, offset);
    if (lane >= offset) prefix += other;
  }
  const int total = __shfl_sync(0xFFFFFFFFu, prefix, 31);
  const int target = total - wanted;
  if (target <= 0) {
    if (lane == 0) {
      *threshold = 0;
      *remain = wanted;
    }
    return;
  }

  const int base = prefix - lane_sum;
  if (target > base && target <= prefix) {
    int running = base;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      running += bins[j];
      if (running >= target) {
        *threshold = lane * 8 + j;
        *remain = running - target;
        break;
      }
    }
  }
}

template <typename OutIndexT>
__global__ void topk_bf16_short_kernel(
    const __mt_bfloat16* __restrict__ input, OutIndexT* __restrict__ output,
    int width, int64_t input_stride, int64_t output_stride, int topk,
    bool abort_when_nan_found) {
  __shared__ uint16_t keys[4096];
  __shared__ int histogram[kRadix];
  __shared__ int threshold_hi;
  __shared__ int remain_hi;
  __shared__ int threshold_lo;
  __shared__ int remain_lo;
  __shared__ int output_count;
  __shared__ int nan_count;

  const int tid = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x);
  const __mt_bfloat16* row_input = input + static_cast<int64_t>(row) * input_stride;
  OutIndexT* row_output = output + static_cast<int64_t>(row) * output_stride;

  if (tid < kRadix) histogram[tid] = 0;
  if (tid == 0) {
    output_count = 0;
    nan_count = 0;
  }
  __syncthreads();

  for (int index = tid; index < width; index += blockDim.x) {
    const uint16_t bits = raw_bits(row_input[index]);
    if (is_nan_key(bits)) {
      atomicAdd(&nan_count, 1);
      keys[index] = 0;
    } else {
      const uint16_t key = ordered_key_from_bits(bits);
      keys[index] = key;
      atomicAdd(&histogram[key >> 8], 1);
    }
  }
  __syncthreads();

  if (nan_count != 0) {
    if (tid == 0) row_output[0] = static_cast<OutIndexT>(kNanSentinel);
    if (abort_when_nan_found) __syncthreads();
    return;
  }

  if (width <= topk) {
    for (int index = tid; index < width; index += blockDim.x)
      row_output[index] = static_cast<OutIndexT>(index);
    return;
  }

  warp_select(histogram, topk, &threshold_hi, &remain_hi);
  __syncthreads();

  if (tid < kRadix) histogram[tid] = 0;
  __syncthreads();
  for (int index = tid; index < width; index += blockDim.x) {
    const uint16_t key = keys[index];
    if ((key >> 8) == threshold_hi)
      atomicAdd(&histogram[key & 0xFFu], 1);
  }
  __syncthreads();
  warp_select(histogram, remain_hi, &threshold_lo, &remain_lo);
  __syncthreads();

  if (tid == 0) output_count = 0;
  __syncthreads();
  for (int index = tid; index < width; index += blockDim.x) {
    const uint16_t key = keys[index];
    const int hi = key >> 8;
    const int lo = key & 0xFFu;
    const bool selected = hi > threshold_hi ||
        (hi == threshold_hi && lo > threshold_lo);
    if (hi == threshold_hi && lo == threshold_lo) {
      const int previous = atomicAdd(&remain_lo, -1);
      if (previous > 0)
        row_output[topk - previous] = static_cast<OutIndexT>(index);
    } else if (selected) {
      const int slot = atomicAdd(&output_count, 1);
      if (slot < topk) row_output[slot] = static_cast<OutIndexT>(index);
    }
  }
}

__device__ __forceinline__ void block_select_4096(
    int* histogram, int wanted, int* threshold, int* remain,
    int* group_sums, int* group_threshold, int* group_remain) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < 256) {
    int sum = 0;
#pragma unroll
    for (int item = 0; item < 16; ++item)
      sum += histogram[tid * 16 + item];
    group_sums[tid] = sum;
  }
  __syncthreads();
  warp_select(group_sums, wanted, group_threshold, group_remain);
  __syncthreads();
  if (tid == *group_threshold) {
    int needed = *group_remain;
#pragma unroll
    for (int item = 15; item >= 0; --item) {
      const int count = histogram[tid * 16 + item];
      if (needed > count) {
        needed -= count;
      } else {
        *threshold = tid * 16 + item;
        *remain = needed;
        break;
      }
    }
  }
  __syncthreads();
}

template <int kThreads, typename OutIndexT>
__global__ __launch_bounds__(
    kThreads,
    kThreads >= 1024 ? 1 : (kThreads >= 512 ? 2 : (kThreads >= 256 ? 8 : 12)))
void topk_bf16_kernel(const __mt_bfloat16* __restrict__ input,
                      OutIndexT* __restrict__ output, int64_t width,
                      int64_t input_stride, int64_t output_stride,
                      __mt_bfloat16* __restrict__ values,
                      int64_t value_stride, const int* __restrict__ end_ptr,
                      OutIndexT idx_fill, __mt_bfloat16 value_fill, int topk,
                      bool return_value) {
  __shared__ int histogram[kRadix];
  __shared__ int output_count;
  __shared__ int threshold_hi;
  __shared__ int threshold_lo;
  __shared__ int remain_hi;
  __shared__ int remain_lo;
  __shared__ int nan_count;

  const int tid = static_cast<int>(threadIdx.x);
  const int row_index = static_cast<int>(blockIdx.x);
  const __mt_bfloat16* row =
      input + static_cast<int64_t>(row_index) * input_stride;
  const int64_t row_width = end_ptr ? end_ptr[row_index] : width;
  OutIndexT* indices =
      output + static_cast<int64_t>(row_index) * output_stride;
  __mt_bfloat16* row_values =
      values ? values + static_cast<int64_t>(row_index) * value_stride : nullptr;
  if (row_width <= topk) {
    for (int position = tid; position < topk; position += kThreads) {
      if (position < row_width)
        store_topk_result(indices, row_values, return_value, position,
                          position, row[position]);
      else {
        indices[position] = idx_fill;
        if (return_value) row_values[position] = value_fill;
      }
    }
    return;
  }

  for (int index = tid; index < kRadix; index += kThreads) histogram[index] = 0;
  if (tid == 0) {
    output_count = 0;
    threshold_hi = 0;
    threshold_lo = 0;
    remain_hi = topk;
    remain_lo = topk;
    nan_count = 0;
  }
  __syncthreads();

  const int64_t aligned_width = row_width & ~int64_t{7};

  for (int64_t index = static_cast<int64_t>(tid) * 8;
       index < aligned_width; index += static_cast<int64_t>(kThreads) * 8) {
    const Bf16x8 values = load_bf16x8(row + index);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const __mt_bfloat16 value = values.values[element];
      const uint16_t raw = raw_bits(value);
      if (is_nan_key(raw)) atomicAdd(&nan_count, 1);
      atomicAdd(&histogram[ordered_key(value) >> 8], 1);
    }
  }
  for (int64_t index = aligned_width + tid; index < row_width;
       index += kThreads) {
    const __mt_bfloat16 value = row[index];
    const uint16_t raw = raw_bits(value);
    if (is_nan_key(raw)) atomicAdd(&nan_count, 1);
    atomicAdd(&histogram[ordered_key(value) >> 8], 1);
  }
  __syncthreads();
  warp_select(histogram, topk, &threshold_hi, &remain_hi);
  __syncthreads();

  const int hi = threshold_hi;
  for (int64_t index = static_cast<int64_t>(tid) * 8;
       index < aligned_width; index += static_cast<int64_t>(kThreads) * 8) {
    const Bf16x8 values = load_bf16x8(row + index);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const int64_t element_index = index + element;
      const uint16_t key = ordered_key(values.values[element]);
      const int key_hi = key >> 8;
      if (key_hi > hi) {
        const int position = atomicAdd(&output_count, 1);
        if (position < topk)
          store_topk_result(indices, row_values, return_value, position,
                            element_index, values.values[element]);
      } else if (key_hi == hi) {
        atomicAdd(&histogram[key & 0xFF], 1);
      }
    }
  }
  for (int64_t index = aligned_width + tid; index < row_width;
       index += kThreads) {
    const uint16_t key = ordered_key(row[index]);
    const int key_hi = key >> 8;
    if (key_hi > hi) {
      const int position = atomicAdd(&output_count, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          row[index]);
    } else if (key_hi == hi) {
      atomicAdd(&histogram[key & 0xFF], 1);
    }
  }
  __syncthreads();
  warp_select(histogram, remain_hi, &threshold_lo, &remain_lo);
  __syncthreads();

  const int lo = threshold_lo;
  for (int64_t index = static_cast<int64_t>(tid) * 8;
       index < aligned_width; index += static_cast<int64_t>(kThreads) * 8) {
    const Bf16x8 values = load_bf16x8(row + index);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const int64_t element_index = index + element;
      const uint16_t key = ordered_key(values.values[element]);
      if ((key >> 8) != hi) continue;
      const int key_lo = key & 0xFF;
      if (key_lo > lo) {
        const int position = atomicAdd(&output_count, 1);
        if (position < topk)
          store_topk_result(indices, row_values, return_value, position,
                            element_index, values.values[element]);
      } else if (key_lo == lo) {
        const int previous = atomicAdd(&remain_lo, -1);
        if (previous > 0)
          store_topk_result(indices, row_values, return_value,
                            topk - previous, element_index,
                            values.values[element]);
      }
    }
  }
  for (int64_t index = aligned_width + tid; index < row_width;
       index += kThreads) {
    const uint16_t key = ordered_key(row[index]);
    if ((key >> 8) != hi) continue;
    const int key_lo = key & 0xFF;
    if (key_lo > lo) {
      const int position = atomicAdd(&output_count, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          row[index]);
    } else if (key_lo == lo) {
      const int previous = atomicAdd(&remain_lo, -1);
      if (previous > 0)
        store_topk_result(indices, row_values, return_value, topk - previous,
                          index, row[index]);
    }
  }
  __syncthreads();
  if (tid == 0 && nan_count > 0)
    indices[0] = static_cast<OutIndexT>(kNanSentinel);
}

template <typename OutIndexT>
__device__ __forceinline__ void compact_bf16_top12(
    __mt_bfloat16 value, int index, OutIndexT* indices,
    __mt_bfloat16* row_values, bool return_value, int topk,
    uint16_t pivot12, int* histogram, int* output_count,
    int* candidate_count, int* candidate_indices, int candidate_capacity) {
  const uint16_t key = ordered_key(value);
  const uint16_t prefix12 = key >> 4;
  if (prefix12 > pivot12) {
    const int position = atomicAdd(output_count, 1);
    if (position < topk)
      store_topk_result(indices, row_values, return_value, position, index,
                        value);
  } else if (prefix12 == pivot12) {
    atomicAdd(&histogram[key & 0xFu], 1);
    const int position = atomicAdd(candidate_count, 1);
    if (position < candidate_capacity) candidate_indices[position] = index;
  }
}

template <int kThreads, int kCandidateCapacity, typename OutIndexT,
          bool kSampleFallbackOnly = false>
__global__ __launch_bounds__(kThreads, 1)
void topk_bf16_compact12_kernel(
    const __mt_bfloat16* __restrict__ input,
    OutIndexT* __restrict__ output, int64_t width, int64_t input_stride,
    int64_t output_stride, __mt_bfloat16* __restrict__ values,
    int64_t value_stride, const int* __restrict__ end_ptr,
    OutIndexT idx_fill, __mt_bfloat16 value_fill, int topk,
    bool return_value) {
  constexpr int kRadix12 = 4096;
  __shared__ int histogram[kRadix12];
  __shared__ int candidate_indices[kCandidateCapacity];
  __shared__ int group_sums[256];
  __shared__ int output_count;
  __shared__ int candidate_count;
  __shared__ int threshold12;
  __shared__ int threshold16;
  __shared__ int remain;
  __shared__ int nan_count;
  __shared__ int group_threshold;
  __shared__ int group_remain;

  const int tid = static_cast<int>(threadIdx.x);
  const int row_index = static_cast<int>(blockIdx.x);
  const __mt_bfloat16* row =
      input + static_cast<int64_t>(row_index) * input_stride;
  const int64_t row_width = end_ptr ? end_ptr[row_index] : width;
  OutIndexT* indices =
      output + static_cast<int64_t>(row_index) * output_stride;
  __mt_bfloat16* row_values =
      values ? values + static_cast<int64_t>(row_index) * value_stride : nullptr;
  if constexpr (kSampleFallbackOnly) {
    if (indices[0] !=
        static_cast<OutIndexT>(kSampleFallbackSentinel))
      return;
  }
  if (row_width <= topk) {
    for (int position = tid; position < topk; position += kThreads) {
      if (position < row_width)
        store_topk_result(indices, row_values, return_value, position,
                          position, row[position]);
      else {
        indices[position] = idx_fill;
        if (return_value) row_values[position] = value_fill;
      }
    }
    return;
  }

  for (int bin = tid; bin < kRadix12; bin += kThreads)
    histogram[bin] = 0;
  if (tid == 0) {
    output_count = 0;
    candidate_count = 0;
    threshold12 = threshold16 = 0;
    remain = topk;
    nan_count = 0;
  }
  __syncthreads();

  const int64_t aligned_width = row_width & ~int64_t{7};
  const bool row_is_aligned =
      (reinterpret_cast<uintptr_t>(row) & uintptr_t{15}) == 0;
  if (row_is_aligned) {
    for (int64_t index = static_cast<int64_t>(tid) * 8;
         index < aligned_width;
         index += static_cast<int64_t>(kThreads) * 8) {
      const Bf16x8 values8 = load_bf16x8(row + index);
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        const __mt_bfloat16 value = values8.values[item];
        const uint16_t raw = raw_bits(value);
        if (is_nan_key(raw)) atomicAdd(&nan_count, 1);
        atomicAdd(&histogram[ordered_key(value) >> 4], 1);
      }
    }
    for (int64_t index = aligned_width + tid; index < row_width;
         index += kThreads) {
      const __mt_bfloat16 value = row[index];
      const uint16_t raw = raw_bits(value);
      if (is_nan_key(raw)) atomicAdd(&nan_count, 1);
      atomicAdd(&histogram[ordered_key(value) >> 4], 1);
    }
  } else {
    for (int64_t index = tid; index < row_width; index += kThreads) {
      const __mt_bfloat16 value = row[index];
      const uint16_t raw = raw_bits(value);
      if (is_nan_key(raw)) atomicAdd(&nan_count, 1);
      atomicAdd(&histogram[ordered_key(value) >> 4], 1);
    }
  }
  __syncthreads();
  block_select_4096(histogram, topk, &threshold12, &remain, group_sums,
                    &group_threshold, &group_remain);
  for (int bin = tid; bin < kRadix12; bin += kThreads)
    histogram[bin] = 0;
  __syncthreads();

  if (row_is_aligned) {
    for (int64_t index = static_cast<int64_t>(tid) * 8;
         index < aligned_width;
         index += static_cast<int64_t>(kThreads) * 8) {
      const Bf16x8 values8 = load_bf16x8(row + index);
#pragma unroll
      for (int item = 0; item < 8; ++item)
        compact_bf16_top12(
            values8.values[item], static_cast<int>(index) + item, indices,
            row_values, return_value, topk,
            static_cast<uint16_t>(threshold12), histogram, &output_count,
            &candidate_count, candidate_indices, kCandidateCapacity);
    }
    for (int64_t index = aligned_width + tid; index < row_width;
         index += kThreads)
      compact_bf16_top12(
          row[index], static_cast<int>(index), indices, row_values,
          return_value, topk, static_cast<uint16_t>(threshold12), histogram,
          &output_count, &candidate_count, candidate_indices,
          kCandidateCapacity);
  } else {
    for (int64_t index = tid; index < row_width; index += kThreads)
      compact_bf16_top12(
          row[index], static_cast<int>(index), indices, row_values,
          return_value, topk, static_cast<uint16_t>(threshold12), histogram,
          &output_count, &candidate_count, candidate_indices,
          kCandidateCapacity);
  }
  __syncthreads();
  warp_select(histogram, remain, &threshold16, &remain);
  __syncthreads();

  if (candidate_count <= kCandidateCapacity) {
    for (int position = tid; position < candidate_count;
         position += kThreads) {
      const int index = candidate_indices[position];
      const __mt_bfloat16 value = row[index];
      const uint16_t low4 = ordered_key(value) & 0xFu;
      if (low4 > static_cast<uint16_t>(threshold16)) {
        const int output_position = atomicAdd(&output_count, 1);
        if (output_position < topk)
          store_topk_result(indices, row_values, return_value,
                            output_position, index, value);
      } else if (low4 == static_cast<uint16_t>(threshold16)) {
        const int previous = atomicAdd(&remain, -1);
        if (previous > 0)
          store_topk_result(indices, row_values, return_value,
                            topk - previous, index, value);
      }
    }
  } else {
    for (int64_t index = tid; index < row_width; index += kThreads) {
      const __mt_bfloat16 value = row[index];
      const uint16_t key = ordered_key(value);
      if ((key >> 4) != static_cast<uint16_t>(threshold12)) continue;
      const uint16_t low4 = key & 0xFu;
      if (low4 > static_cast<uint16_t>(threshold16)) {
        const int output_position = atomicAdd(&output_count, 1);
        if (output_position < topk)
          store_topk_result(indices, row_values, return_value,
                            output_position, index, value);
      } else if (low4 == static_cast<uint16_t>(threshold16)) {
        const int previous = atomicAdd(&remain, -1);
        if (previous > 0)
          store_topk_result(indices, row_values, return_value,
                            topk - previous, index, value);
      }
    }
  }
  __syncthreads();
  if (tid == 0 && nan_count > 0)
    indices[0] = static_cast<OutIndexT>(kNanSentinel);
}

__device__ __forceinline__ void compact_bf16_survivor_high8(
    __mt_bfloat16 value, int index, uint16_t pivot_hi,
    __mt_bfloat16* next_values, int* next_indices, int* histogram,
    int* output_count) {
  const uint16_t key = ordered_key(value);
  const uint16_t hi = key >> 8;
  if (hi > pivot_hi) {
    const int position = atomicAdd(output_count, 1);
    if (position < 512) {
      next_values[position] = value;
      next_indices[position] = index;
    }
  } else if (hi == pivot_hi) {
    atomicAdd(&histogram[key & 0xFFu], 1);
  }
}

__device__ __forceinline__ void finish_bf16_survivor_high8(
    __mt_bfloat16 value, int index, uint16_t pivot_hi, uint16_t pivot_lo,
    __mt_bfloat16* next_values, int* next_indices, int* output_count,
    int* remain) {
  const uint16_t key = ordered_key(value);
  if ((key >> 8) != pivot_hi) return;
  const uint16_t lo = key & 0xFFu;
  if (lo > pivot_lo) {
    const int position = atomicAdd(output_count, 1);
    if (position < 512) {
      next_values[position] = value;
      next_indices[position] = index;
    }
  } else if (lo == pivot_lo) {
    const int previous = atomicAdd(remain, -1);
    if (previous > 0) {
      const int position = atomicAdd(output_count, 1);
      if (position < 512) {
        next_values[position] = value;
        next_indices[position] = index;
      }
    }
  }
}

template <bool kCheckTileNan>
__device__ __forceinline__ uint16_t select_bf16_survivor_high8(
    int survivor_count, const __mt_bfloat16* previous_values,
    const int* previous_indices, int incoming_count,
    const __mt_bfloat16* incoming_values, const int* incoming_indices,
    const __mt_bfloat16* tile, int tile_count, int tile_index_base,
    uint16_t tile_min_key,
    __mt_bfloat16* next_values, int* next_indices, int* histogram,
    int* output_count, int* threshold_hi, int* threshold_lo, int* remain,
    int* nan_count) {
  constexpr int kTopK = 512;
  constexpr int kRadix = 256;
  const int tid = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);
  for (int position = tid; position < kTopK; position += threads)
    next_indices[position] = -1;
  for (int bin = tid; bin < kRadix; bin += threads) histogram[bin] = 0;
  if (tid == 0) {
    *output_count = 0;
    *threshold_hi = *threshold_lo = 0;
    *remain = kTopK;
  }
  __syncthreads();

  for (int position = tid; position < survivor_count;
       position += threads)
    atomicAdd(&histogram[ordered_key(previous_values[position]) >> 8], 1);
  for (int position = tid; position < incoming_count;
       position += threads)
    atomicAdd(&histogram[ordered_key(incoming_values[position]) >> 8], 1);
  for (int position = tid; position < tile_count; position += threads) {
    const uint16_t bits = raw_bits(tile[position]);
    if constexpr (kCheckTileNan) {
      if (is_nan_key(bits)) atomicAdd(nan_count, 1);
    }
    const uint16_t key = ordered_key_from_bits(bits);
    if (key >= tile_min_key) atomicAdd(&histogram[key >> 8], 1);
  }
  __syncthreads();
  warp_select(histogram, kTopK, threshold_hi, remain);
  __syncthreads();

  const uint16_t pivot_hi = static_cast<uint16_t>(*threshold_hi);
  for (int position = tid; position < survivor_count;
       position += threads)
    compact_bf16_survivor_high8(
        previous_values[position], previous_indices[position], pivot_hi,
        next_values, next_indices, histogram, output_count);
  for (int position = tid; position < incoming_count;
       position += threads)
    compact_bf16_survivor_high8(
        incoming_values[position], incoming_indices[position], pivot_hi,
        next_values, next_indices, histogram, output_count);
  for (int position = tid; position < tile_count; position += threads) {
    const __mt_bfloat16 value = tile[position];
    if (ordered_key(value) >= tile_min_key)
      compact_bf16_survivor_high8(
          value, tile_index_base + position, pivot_hi, next_values,
          next_indices, histogram, output_count);
  }
  __syncthreads();
  warp_select(histogram, *remain, threshold_lo, remain);
  __syncthreads();

  const uint16_t pivot_lo = static_cast<uint16_t>(*threshold_lo);
  for (int position = tid; position < survivor_count;
       position += threads)
    finish_bf16_survivor_high8(
        previous_values[position], previous_indices[position], pivot_hi,
        pivot_lo, next_values, next_indices, output_count, remain);
  for (int position = tid; position < incoming_count;
       position += threads)
    finish_bf16_survivor_high8(
        incoming_values[position], incoming_indices[position], pivot_hi,
        pivot_lo, next_values, next_indices, output_count, remain);
  for (int position = tid; position < tile_count; position += threads) {
    const __mt_bfloat16 value = tile[position];
    if (ordered_key(value) >= tile_min_key)
      finish_bf16_survivor_high8(
          value, tile_index_base + position, pivot_hi, pivot_lo, next_values,
          next_indices, output_count, remain);
  }
  __syncthreads();
  return static_cast<uint16_t>((pivot_hi << 8) | pivot_lo);
}

template <bool kPositivePivot>
__device__ __forceinline__ void collect_bf16_sample_candidate(
    __mt_bfloat16 value, int index, uint16_t sample_pivot,
    __mt_bfloat16* candidate_values, int* candidate_indices,
    int* candidate_count, int candidate_capacity, int* nan_count) {
  const uint16_t bits = raw_bits(value);
  if ((bits & 0x7FFFu) > 0x7F80u) atomicAdd(nan_count, 1);
  bool selected;
  if constexpr (kPositivePivot)
    selected = (bits & 0x8000u) == 0 && bits >= sample_pivot;
  else
    selected = ordered_key_from_bits(bits) >= sample_pivot;
  if (selected) {
    const int position = atomicAdd(candidate_count, 1);
    if (position < candidate_capacity) {
      candidate_values[position] = value;
      candidate_indices[position] = index;
    }
  }
}

template <bool kPositivePivot, int kThreads, int kScanVectors = 1>
__device__ __forceinline__ void scan_bf16_sample_candidates(
    const __mt_bfloat16* row, int64_t width, uint16_t sample_pivot,
    __mt_bfloat16* candidate_values, int* candidate_indices,
    int* candidate_count, int candidate_capacity, int* nan_count) {
  const int tid = static_cast<int>(threadIdx.x);
  constexpr int kVectorElements = 8;
  constexpr int64_t kIterationElements =
      static_cast<int64_t>(kThreads) * kVectorElements * kScanVectors;
  const int64_t aligned_width = width / kIterationElements * kIterationElements;
  const bool row_is_aligned =
      (reinterpret_cast<uintptr_t>(row) & uintptr_t{15}) == 0;
  if (row_is_aligned) {
    for (int64_t tile_base = 0; tile_base < aligned_width;
         tile_base += kIterationElements) {
      const int64_t index =
          tile_base + static_cast<int64_t>(tid) * kVectorElements;
      Bf16x8 vectors[kScanVectors];
#pragma unroll
      for (int vector = 0; vector < kScanVectors; ++vector)
        vectors[vector] = load_bf16x8(
            row + index + static_cast<int64_t>(vector) * kThreads *
                              kVectorElements);
#pragma unroll
      for (int item = 0; item < 8; ++item)
#pragma unroll
        for (int vector = 0; vector < kScanVectors; ++vector) {
          const int64_t vector_index =
              index + static_cast<int64_t>(vector) * kThreads *
                          kVectorElements;
          collect_bf16_sample_candidate<kPositivePivot>(
              vectors[vector].values[item],
              static_cast<int>(vector_index) + item, sample_pivot,
              candidate_values, candidate_indices, candidate_count,
              candidate_capacity, nan_count);
        }
    }
    for (int64_t index = aligned_width + tid; index < width;
         index += kThreads)
      collect_bf16_sample_candidate<kPositivePivot>(
          row[index], static_cast<int>(index), sample_pivot,
          candidate_values, candidate_indices, candidate_count,
          candidate_capacity, nan_count);
  } else {
    for (int64_t index = tid; index < width; index += kThreads)
      collect_bf16_sample_candidate<kPositivePivot>(
          row[index], static_cast<int>(index), sample_pivot,
          candidate_values, candidate_indices, candidate_count,
          candidate_capacity, nan_count);
  }
}

template <int kThreads, typename OutIndexT, int kScanVectors = 1>
__global__ __launch_bounds__(kThreads, 1)
void topk_bf16_sample_survivor_kernel(
    const __mt_bfloat16* __restrict__ input,
    OutIndexT* __restrict__ output, int64_t width, int64_t input_stride,
    int64_t output_stride, __mt_bfloat16* __restrict__ values,
    int64_t value_stride, bool return_value, int sample_elements,
    int candidate_target, bool use_positive_key_scan) {
  constexpr int kCandidateCapacity = 4096;
  constexpr int kTopK = 512;

  __shared__ __mt_bfloat16 candidate_values[kCandidateCapacity];
  __shared__ int candidate_indices[kCandidateCapacity];
  __shared__ __mt_bfloat16 selected_values[kTopK];
  __shared__ int selected_indices[kTopK];
  __shared__ int histogram[256];
  __shared__ int candidate_count;
  __shared__ int output_count;
  __shared__ int sample_threshold_hi;
  __shared__ int sample_threshold_lo;
  __shared__ int sample_remain;
  __shared__ int threshold_hi;
  __shared__ int threshold_lo;
  __shared__ int remain;
  __shared__ int nan_count;

  const int tid = static_cast<int>(threadIdx.x);
  const int row_index = static_cast<int>(blockIdx.x);
  const __mt_bfloat16* row =
      input + static_cast<int64_t>(row_index) * input_stride;
  OutIndexT* indices =
      output + static_cast<int64_t>(row_index) * output_stride;
  __mt_bfloat16* row_values =
      values ? values + static_cast<int64_t>(row_index) * value_stride : nullptr;
  const int sample_count = static_cast<int>(
      width < sample_elements ? width : sample_elements);
  for (int bin = tid; bin < 256; bin += kThreads) histogram[bin] = 0;
  if (tid == 0) {
    sample_remain = static_cast<int>(
        (static_cast<int64_t>(candidate_target) * sample_count + width - 1) /
        width);
    if (sample_remain < 1) sample_remain = 1;
    if (sample_remain > sample_count) sample_remain = sample_count;
    sample_threshold_hi = sample_threshold_lo = 0;
    nan_count = 0;
  }
  __syncthreads();
  const int sample_wanted = sample_remain;
  for (int position = tid; position < sample_count; position += kThreads)
    atomicAdd(&histogram[ordered_key(row[position]) >> 8], 1);
  __syncthreads();
  warp_select(
      histogram, sample_wanted, &sample_threshold_hi, &sample_remain);
  __syncthreads();
  const uint16_t sample_hi = static_cast<uint16_t>(sample_threshold_hi);
  for (int position = tid; position < sample_count; position += kThreads) {
    const uint16_t key = ordered_key(row[position]);
    if ((key >> 8) == sample_hi)
      atomicAdd(&histogram[key & 0xFFu], 1);
  }
  __syncthreads();
  warp_select(
      histogram, sample_remain, &sample_threshold_lo, &sample_remain);
  __syncthreads();
  const uint16_t sample_pivot = static_cast<uint16_t>(
      (sample_hi << 8) | static_cast<uint16_t>(sample_threshold_lo));

  if (tid == 0) candidate_count = 0;
  __syncthreads();
  if (use_positive_key_scan && (sample_pivot & 0x8000u) != 0)
    scan_bf16_sample_candidates<true, kThreads, kScanVectors>(
        row, width, static_cast<uint16_t>(sample_pivot & 0x7FFFu),
        candidate_values, candidate_indices, &candidate_count,
        kCandidateCapacity, &nan_count);
  else
    scan_bf16_sample_candidates<false, kThreads, kScanVectors>(
        row, width, sample_pivot, candidate_values, candidate_indices,
        &candidate_count, kCandidateCapacity, &nan_count);
  __syncthreads();

  if (candidate_count < kTopK || candidate_count > kCandidateCapacity) {
    if (tid == 0)
      indices[0] = static_cast<OutIndexT>(kSampleFallbackSentinel);
    return;
  }

  select_bf16_survivor_high8<false>(
      0, selected_values, selected_indices, candidate_count,
      candidate_values, candidate_indices, nullptr, 0, 0, 0,
      selected_values, selected_indices, histogram, &output_count,
      &threshold_hi, &threshold_lo, &remain, &nan_count);
  for (int position = tid; position < kTopK; position += kThreads)
    store_topk_result(
        indices, row_values, return_value, position,
        selected_indices[position], selected_values[position]);
  __syncthreads();
  if (tid == 0 && nan_count > 0)
    indices[0] = static_cast<OutIndexT>(kNanSentinel);
}

template <typename OutIndexT>
__global__ __launch_bounds__(1024, 1)
void topk_bf16_tme_survivor_kernel(
    const __mt_bfloat16* __restrict__ input,
    OutIndexT* __restrict__ output, int64_t width, int64_t input_stride,
    int64_t output_stride, __mt_bfloat16* __restrict__ values,
    int64_t value_stride, bool return_value) {
  constexpr int kTileElements = 24384;
  constexpr int kIncomingCapacity = 1408;
  constexpr int kTopK = 512;

  __shared__ __align__(256) __mt_bfloat16 tile[kTileElements];
  __shared__ __mt_bfloat16 incoming_values[kIncomingCapacity];
  __shared__ int incoming_indices[kIncomingCapacity];
  __shared__ __mt_bfloat16 survivor_values[2][kTopK];
  __shared__ int survivor_indices[2][kTopK];
  __shared__ int histogram[256];
  __shared__ int incoming_count;
  __shared__ int incoming_before;
  __shared__ int incoming_overflow;
  __shared__ int output_count;
  __shared__ int threshold_hi;
  __shared__ int threshold_lo;
  __shared__ int remain;
  __shared__ int nan_count;

  mutlass::arch::allocate_async_barriers(1);
  mutlass::arch::AsyncTransactionBarrier tme_barrier0{0};
  if (threadIdx.x == 0) {
    tme_barrier0.init(1);
    incoming_count = 0;
    nan_count = 0;
  }
  __syncthreads();

  const int tid = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);
  const int row_index = static_cast<int>(blockIdx.x);
  const __mt_bfloat16* row =
      input + static_cast<int64_t>(row_index) * input_stride;
  OutIndexT* indices =
      output + static_cast<int64_t>(row_index) * output_stride;
  __mt_bfloat16* row_values =
      values ? values + static_cast<int64_t>(row_index) * value_stride : nullptr;
  int survivor_count = 0;
  int survivor_buffer = 0;
  uint16_t pivot_key = 0;

  for (int64_t chunk_base = 0; chunk_base < width;
       chunk_base += kTileElements) {
    const int chunk_count = static_cast<int>(
        width - chunk_base < kTileElements ? width - chunk_base
                                           : kTileElements);
    __mt_bfloat16* current_tile = tile;
    if (chunk_count == kTileElements) {
      if (tid < 32) {
        tme_barrier0.expect_transaction(
            kTileElements * sizeof(__mt_bfloat16));
        mute::MP31_BLK_COPY_G2S::copy(
            row + chunk_base, tme_barrier0.get_barrier_id(), tile,
            kTileElements * sizeof(__mt_bfloat16));
        const auto phase = tme_barrier0.arrive<true>();
        tme_barrier0.wait(phase);
      }
    } else {
      for (int position = tid; position < chunk_count;
           position += threads)
        current_tile[position] = row[chunk_base + position];
    }
    __syncthreads();

    if (survivor_count == 0) {
      const int next_buffer = survivor_buffer ^ 1;
      pivot_key = select_bf16_survivor_high8<true>(
          0, survivor_values[survivor_buffer],
          survivor_indices[survivor_buffer], 0, incoming_values,
          incoming_indices, current_tile, chunk_count,
          static_cast<int>(chunk_base), 0,
          survivor_values[next_buffer], survivor_indices[next_buffer],
          histogram, &output_count, &threshold_hi, &threshold_lo, &remain,
          &nan_count);
      survivor_buffer = next_buffer;
      survivor_count = kTopK;
      if (tid == 0) incoming_count = 0;
      __syncthreads();
    } else {
      if (tid == 0) {
        incoming_before = incoming_count;
        incoming_overflow = 0;
      }
      __syncthreads();
      for (int position = tid; position < chunk_count;
           position += threads) {
        const __mt_bfloat16 value = current_tile[position];
        if (is_nan_key(raw_bits(value))) atomicAdd(&nan_count, 1);
        if (ordered_key(value) >= pivot_key) {
          const int incoming = atomicAdd(&incoming_count, 1);
          if (incoming < kIncomingCapacity) {
            incoming_values[incoming] = value;
            incoming_indices[incoming] =
                static_cast<int>(chunk_base) + position;
          } else {
            atomicExch(&incoming_overflow, 1);
          }
        }
      }
      __syncthreads();

      if (incoming_overflow != 0) {
        const int next_buffer = survivor_buffer ^ 1;
        pivot_key = select_bf16_survivor_high8<false>(
            survivor_count, survivor_values[survivor_buffer],
            survivor_indices[survivor_buffer], incoming_before,
            incoming_values, incoming_indices, current_tile, chunk_count,
            static_cast<int>(chunk_base), pivot_key,
            survivor_values[next_buffer], survivor_indices[next_buffer],
            histogram, &output_count, &threshold_hi, &threshold_lo, &remain,
            &nan_count);
        survivor_buffer = next_buffer;
        survivor_count = kTopK;
        if (tid == 0) incoming_count = 0;
        __syncthreads();
      }
    }

    const bool last_chunk = chunk_base + chunk_count == width;
    if (last_chunk && incoming_count != 0) {
      const int next_buffer = survivor_buffer ^ 1;
      pivot_key = select_bf16_survivor_high8<false>(
          survivor_count, survivor_values[survivor_buffer],
          survivor_indices[survivor_buffer], incoming_count,
          incoming_values, incoming_indices, current_tile, 0, 0, 0,
          survivor_values[next_buffer], survivor_indices[next_buffer],
          histogram, &output_count, &threshold_hi, &threshold_lo, &remain,
          &nan_count);
      survivor_buffer = next_buffer;
      survivor_count = kTopK;
      if (tid == 0) incoming_count = 0;
      __syncthreads();
    }

  }

  for (int position = tid; position < kTopK; position += threads)
    store_topk_result(
        indices, row_values, return_value, position,
        survivor_indices[survivor_buffer][position],
        survivor_values[survivor_buffer][position]);
  __syncthreads();
  if (tid == 0 && nan_count > 0)
    indices[0] = static_cast<OutIndexT>(kNanSentinel);
}

struct MultiScratch {
  int* high_hist;
  int* low_hist;
  int* partial_nan;
  int* candidate_counts;
  int* threshold_hi;
  int* remain_hi;
  int* threshold_lo;
  int* remain_lo;
  int* output_count;
  int* has_nan;
  int* candidate_indices;
};

MultiScratch split_scratch(int* scratch, int rows, int parts, int64_t width) {
  const int hist_size = rows * parts * kRadix;
  const int partial_size = rows * parts;
  MultiScratch result;
  result.high_hist = scratch;
  result.low_hist = result.high_hist + hist_size;
  result.partial_nan = result.low_hist + hist_size;
  result.candidate_counts = result.partial_nan + partial_size;
  result.threshold_hi = result.candidate_counts + partial_size;
  result.remain_hi = result.threshold_hi + rows;
  result.threshold_lo = result.remain_hi + rows;
  result.remain_lo = result.threshold_lo + rows;
  result.output_count = result.remain_lo + rows;
  result.has_nan = result.output_count + rows;
  result.candidate_indices = result.has_nan + rows;
  return result;
}

__global__ void partial_high_histogram_kernel(
    const __mt_bfloat16* __restrict__ input, int* __restrict__ partial_hist,
    int* __restrict__ partial_nan, int64_t width, int parts) {
  __shared__ int histogram[kRadix];
  __shared__ int nan_count;
  const int tid = static_cast<int>(threadIdx.x);
  const int task = static_cast<int>(blockIdx.x);
  const int row_index = task / parts;
  const int part = task - row_index * parts;
  const int64_t begin = width * part / parts;
  const int64_t end = width * (part + 1) / parts;
  const __mt_bfloat16* row = input + static_cast<int64_t>(row_index) * width;
  for (int bin = tid; bin < kRadix; bin += blockDim.x) histogram[bin] = 0;
  if (tid == 0) nan_count = 0;
  __syncthreads();
  const int64_t aligned_begin = (begin + 7) & ~int64_t{7};
  const int64_t aligned_end = end & ~int64_t{7};
  for (int64_t index = begin + tid; index < min(end, aligned_begin);
       index += blockDim.x) {
    const __mt_bfloat16 value = row[index];
    const uint16_t raw = raw_bits(value);
    if (is_nan_key(raw)) atomicAdd(&nan_count, 1);
    atomicAdd(&histogram[ordered_key(value) >> 8], 1);
  }
  for (int64_t index = aligned_begin + static_cast<int64_t>(tid) * 8;
       index < aligned_end; index += static_cast<int64_t>(blockDim.x) * 8) {
    const Bf16x8 values = load_bf16x8(row + index);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const __mt_bfloat16 value = values.values[element];
      const uint16_t raw = raw_bits(value);
      if (is_nan_key(raw)) atomicAdd(&nan_count, 1);
      atomicAdd(&histogram[ordered_key(value) >> 8], 1);
    }
  }
  for (int64_t index = aligned_end + tid; index < end; index += blockDim.x) {
    const __mt_bfloat16 value = row[index];
    const uint16_t raw = raw_bits(value);
    if (is_nan_key(raw)) atomicAdd(&nan_count, 1);
    atomicAdd(&histogram[ordered_key(value) >> 8], 1);
  }
  __syncthreads();
  int* destination = partial_hist + static_cast<int64_t>(task) * kRadix;
  for (int bin = tid; bin < kRadix; bin += blockDim.x)
    destination[bin] = histogram[bin];
  if (tid == 0) partial_nan[task] = nan_count;
}

template <typename OutIndexT>
__global__ void reduce_high_histogram_kernel(
    const int* __restrict__ partial_hist,
    const int* __restrict__ partial_nan, OutIndexT* __restrict__ output,
    int* __restrict__ threshold_hi, int* __restrict__ remain_hi,
    int* __restrict__ output_count, int* __restrict__ has_nan,
    int64_t output_stride, int parts, int topk) {
  __shared__ int histogram[kRadix];
  const int tid = static_cast<int>(threadIdx.x);
  const int row_index = static_cast<int>(blockIdx.x);
  int count = 0;
  if (tid < kRadix) {
    for (int part = 0; part < parts; ++part)
      count += partial_hist[(static_cast<int64_t>(row_index) * parts + part) *
                            kRadix + tid];
    histogram[tid] = count;
  }
  if (tid == 0) {
    int nan_count = 0;
    for (int part = 0; part < parts; ++part)
      nan_count += partial_nan[row_index * parts + part];
    has_nan[row_index] = nan_count != 0;
    output_count[row_index] = 0;
    if (nan_count != 0)
      output[static_cast<int64_t>(row_index) * output_stride] =
          static_cast<OutIndexT>(kNanSentinel);
  }
  __syncthreads();
  warp_select(histogram, topk, threshold_hi + row_index,
              remain_hi + row_index);
}

template <typename OutIndexT>
__global__ void partial_low_histogram_kernel(
    const __mt_bfloat16* __restrict__ input, OutIndexT* __restrict__ output,
    int* __restrict__ partial_hist, const int* __restrict__ threshold_hi,
    int* __restrict__ output_count, const int* __restrict__ has_nan,
    int* __restrict__ candidate_counts,
    int* __restrict__ candidate_indices, int64_t width,
    int64_t output_stride, __mt_bfloat16* __restrict__ values,
    int64_t value_stride, int parts, int topk, bool return_value) {
  __shared__ int histogram[kRadix];
  __shared__ int local_candidate_count;
  const int tid = static_cast<int>(threadIdx.x);
  const int task = static_cast<int>(blockIdx.x);
  const int row_index = task / parts;
  if (has_nan[row_index]) return;
  const int part = task - row_index * parts;
  const int64_t begin = width * part / parts;
  const int64_t end = width * (part + 1) / parts;
  const int hi = threshold_hi[row_index];
  const __mt_bfloat16* row = input + static_cast<int64_t>(row_index) * width;
  OutIndexT* indices =
      output + static_cast<int64_t>(row_index) * output_stride;
  __mt_bfloat16* row_values =
      values ? values + static_cast<int64_t>(row_index) * value_stride : nullptr;
  for (int bin = tid; bin < kRadix; bin += blockDim.x) histogram[bin] = 0;
  if (tid == 0) local_candidate_count = 0;
  __syncthreads();
  const int64_t aligned_begin = (begin + 7) & ~int64_t{7};
  const int64_t aligned_end = end & ~int64_t{7};
  for (int64_t index = begin + tid; index < min(end, aligned_begin);
       index += blockDim.x) {
    const uint16_t key = ordered_key(row[index]);
    const int key_hi = key >> 8;
    if (key_hi > hi) {
      const int position = atomicAdd(output_count + row_index, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          row[index]);
    } else if (key_hi == hi) {
      atomicAdd(&histogram[key & 0xFF], 1);
      const int position = atomicAdd(&local_candidate_count, 1);
      candidate_indices[static_cast<int64_t>(row_index) * width + begin +
                        position] = static_cast<int32_t>(index);
    }
  }
  for (int64_t index = aligned_begin + static_cast<int64_t>(tid) * 8;
       index < aligned_end; index += static_cast<int64_t>(blockDim.x) * 8) {
    const Bf16x8 values = load_bf16x8(row + index);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const int64_t element_index = index + element;
      const uint16_t key = ordered_key(values.values[element]);
      const int key_hi = key >> 8;
      if (key_hi > hi) {
        const int output_position = atomicAdd(output_count + row_index, 1);
        if (output_position < topk)
          store_topk_result(indices, row_values, return_value, output_position,
                            element_index, values.values[element]);
      } else if (key_hi == hi) {
        atomicAdd(&histogram[key & 0xFF], 1);
        const int candidate_position = atomicAdd(&local_candidate_count, 1);
        candidate_indices[static_cast<int64_t>(row_index) * width + begin +
                          candidate_position] =
            static_cast<int32_t>(element_index);
      }
    }
  }
  for (int64_t index = aligned_end + tid; index < end; index += blockDim.x) {
    const uint16_t key = ordered_key(row[index]);
    const int key_hi = key >> 8;
    if (key_hi > hi) {
      const int position = atomicAdd(output_count + row_index, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          row[index]);
    } else if (key_hi == hi) {
      atomicAdd(&histogram[key & 0xFF], 1);
      const int position = atomicAdd(&local_candidate_count, 1);
      candidate_indices[static_cast<int64_t>(row_index) * width + begin +
                        position] = static_cast<int32_t>(index);
    }
  }
  __syncthreads();
  int* destination = partial_hist + static_cast<int64_t>(task) * kRadix;
  for (int bin = tid; bin < kRadix; bin += blockDim.x)
    destination[bin] = histogram[bin];
  if (tid == 0) candidate_counts[task] = local_candidate_count;
}

__global__ void reduce_low_histogram_kernel(
    const int* __restrict__ partial_hist, int* __restrict__ threshold_lo,
    int* __restrict__ remain_lo, const int* __restrict__ remain_hi,
    const int* __restrict__ has_nan, int parts) {
  __shared__ int histogram[kRadix];
  const int tid = static_cast<int>(threadIdx.x);
  const int row_index = static_cast<int>(blockIdx.x);
  if (has_nan[row_index]) return;
  if (tid < kRadix) {
    int count = 0;
    for (int part = 0; part < parts; ++part)
      count += partial_hist[(static_cast<int64_t>(row_index) * parts + part) *
                            kRadix + tid];
    histogram[tid] = count;
  }
  __syncthreads();
  warp_select(histogram, remain_hi[row_index], threshold_lo + row_index,
              remain_lo + row_index);
}

template <typename OutIndexT>
__global__ void finish_topk_kernel(
    const __mt_bfloat16* __restrict__ input, OutIndexT* __restrict__ output,
    const int* __restrict__ threshold_hi,
    const int* __restrict__ threshold_lo, int* __restrict__ remain_lo,
    int* __restrict__ output_count, const int* __restrict__ has_nan,
    const int* __restrict__ candidate_counts,
    const int* __restrict__ candidate_indices, int64_t width,
    int64_t output_stride, __mt_bfloat16* __restrict__ values,
    int64_t value_stride, int parts, int topk, bool return_value) {
  const int tid = static_cast<int>(threadIdx.x);
  const int task = static_cast<int>(blockIdx.x);
  const int row_index = task / parts;
  if (has_nan[row_index]) return;
  const int part = task - row_index * parts;
  const int64_t begin = width * part / parts;
  const int count = candidate_counts[task];
  const int hi = threshold_hi[row_index];
  const int lo = threshold_lo[row_index];
  const __mt_bfloat16* row = input + static_cast<int64_t>(row_index) * width;
  OutIndexT* indices =
      output + static_cast<int64_t>(row_index) * output_stride;
  __mt_bfloat16* row_values =
      values ? values + static_cast<int64_t>(row_index) * value_stride : nullptr;
  for (int position = tid; position < count; position += blockDim.x) {
    const int64_t index = candidate_indices[
        static_cast<int64_t>(row_index) * width + begin + position];
    const uint16_t key = ordered_key(row[index]);
    if ((key >> 8) != hi) continue;
    const int key_lo = key & 0xFF;
    if (key_lo > lo) {
      const int position = atomicAdd(output_count + row_index, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          row[index]);
    } else if (key_lo == lo) {
      const int previous = atomicAdd(remain_lo + row_index, -1);
      if (previous > 0)
        store_topk_result(indices, row_values, return_value, topk - previous,
                          index, row[index]);
    }
  }
}

int select_threads(int64_t rows, int64_t width) {
  if (rows <= 96 && width >= 8192) return 1024;
  if (rows <= 192) return 512;
  if (rows <= 512 && width >= 131072) return 1024;
  if (rows <= 1024 && width >= 65536) return 512;
  if (rows > 320 && width <= 16384) return 128;
  return 256;
}

template <typename OutIndexT>
void launch_topk_bf16_compact12_typed(
    const __mt_bfloat16* input_ptr, OutIndexT* output_ptr, int rows,
    int threads, int64_t width, int64_t input_stride, int64_t output_stride,
    __mt_bfloat16* values, int64_t value_stride, const int* end_ptr,
    OutIndexT idx_fill, __mt_bfloat16 value_fill, int topk,
    bool return_value, musaStream_t stream) {
  if (rows <= 1024) {
    if (threads == 1024)
      topk_bf16_compact12_kernel<1024, 4096, OutIndexT>
          <<<rows, 1024, 0, stream>>>(
              input_ptr, output_ptr, width, input_stride, output_stride,
              values, value_stride, end_ptr, idx_fill, value_fill, topk,
              return_value);
    else
      topk_bf16_compact12_kernel<512, 4096, OutIndexT>
          <<<rows, 512, 0, stream>>>(
              input_ptr, output_ptr, width, input_stride, output_stride,
              values, value_stride, end_ptr, idx_fill, value_fill, topk,
              return_value);
  } else {
    topk_bf16_compact12_kernel<256, 1024, OutIndexT>
        <<<rows, 256, 0, stream>>>(
            input_ptr, output_ptr, width, input_stride, output_stride, values,
            value_stride, end_ptr, idx_fill, value_fill, topk, return_value);
  }
}

template <typename OutIndexT>
void launch_topk_bf16_typed(
    const __mt_bfloat16* input_ptr, OutIndexT* output_ptr, int rows,
    int64_t width, int64_t input_stride, int64_t output_stride,
    __mt_bfloat16* values, int64_t value_stride, const int* end_ptr,
    OutIndexT idx_fill, __mt_bfloat16 value_fill, int topk,
    bool return_value, bool abort_when_nan_found, musaStream_t stream) {
  const bool use_short = rows <= 8 && width <= 2048 && topk <= 1024 &&
      topk <= width && end_ptr == nullptr && input_stride == width &&
      values == nullptr && !return_value && !abort_when_nan_found;
  if (use_short) {
    topk_bf16_short_kernel<OutIndexT><<<rows, 256, 0, stream>>>(
        input_ptr, output_ptr, static_cast<int>(width), input_stride,
        output_stride, topk, abort_when_nan_found);
    return;
  }
  const int threads = select_threads(rows, width);
  const bool use_sample_survivor =
      rows >= 512 && width >= 65536 && topk == 512 && end_ptr == nullptr &&
      input_stride == width;
  if (use_sample_survivor) {
    const bool sample_threads_1024 = rows < 2048 && width >= 524288;
    if (sample_threads_1024)
      topk_bf16_sample_survivor_kernel<1024, OutIndexT, 2>
          <<<rows, 1024, 0, stream>>>(
              input_ptr, output_ptr, width, input_stride, output_stride,
              values, value_stride, return_value, 16384, 1536, true);
    else
      topk_bf16_sample_survivor_kernel<512, OutIndexT, 2>
          <<<rows, 512, 0, stream>>>(
              input_ptr, output_ptr, width, input_stride, output_stride,
              values, value_stride, return_value, 16384, 1536, true);
    topk_bf16_compact12_kernel<256, 1024, OutIndexT, true>
        <<<rows, 256, 0, stream>>>(
            input_ptr, output_ptr, width, input_stride, output_stride, values,
            value_stride, end_ptr, idx_fill, value_fill, topk, return_value);
    return;
  }
  const bool use_compact12 =
      (rows > 8 && width >= 65536 && compact12_topk_ok(topk)) ||
      (rows == 6 && width == 16384 && topk == 512 && end_ptr == nullptr &&
       input_stride == width);
  if (use_compact12) {
    launch_topk_bf16_compact12_typed(
        input_ptr, output_ptr, rows, threads, width, input_stride,
        output_stride, values, value_stride, end_ptr, idx_fill, value_fill,
        topk, return_value, stream);
    return;
  }
  if (threads == 1024)
    topk_bf16_kernel<1024, OutIndexT><<<rows, 1024, 0, stream>>>(
        input_ptr, output_ptr, width, input_stride, output_stride, values,
        value_stride, end_ptr, idx_fill, value_fill, topk, return_value);
  else if (threads == 512)
    topk_bf16_kernel<512, OutIndexT><<<rows, 512, 0, stream>>>(
        input_ptr, output_ptr, width, input_stride, output_stride, values,
        value_stride, end_ptr, idx_fill, value_fill, topk, return_value);
  else if (threads == 256)
    topk_bf16_kernel<256, OutIndexT><<<rows, 256, 0, stream>>>(
        input_ptr, output_ptr, width, input_stride, output_stride, values,
        value_stride, end_ptr, idx_fill, value_fill, topk, return_value);
  else
    topk_bf16_kernel<128, OutIndexT><<<rows, 128, 0, stream>>>(
        input_ptr, output_ptr, width, input_stride, output_stride, values,
        value_stride, end_ptr, idx_fill, value_fill, topk, return_value);
}

union F32Bits {
  float value;
  uint32_t bits;
};

__device__ __forceinline__ uint32_t raw_bits(float value) {
  F32Bits converted;
  converted.value = value;
  return converted.bits;
}

__device__ __forceinline__ uint32_t ordered_key(float value) {
  const uint32_t bits = raw_bits(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ float value_from_ordered_key(uint32_t key) {
  F32Bits converted;
  converted.bits = (key & 0x80000000u) ? (key & 0x7FFFFFFFu) : ~key;
  return converted.value;
}

__device__ __forceinline__ bool is_nan_key(uint32_t bits) {
  return (bits & 0x7F800000u) == 0x7F800000u &&
         (bits & 0x007FFFFFu) != 0;
}

__device__ __forceinline__ bool is_nan_ordered_key(uint32_t key) {
  return key > 0xff800000u || key < 0x007fffffu;
}

__device__ __forceinline__ int warp_lane0_atomic_add(int* ptr, int delta) {
  int base = 0;
  if ((threadIdx.x & 31) == 0) base = atomicAdd(ptr, delta);
  return __shfl_sync(0xffffffffu, base, 0);
}

__device__ __forceinline__ int warp_reserve(int* count, uint32_t mask,
                                            uint32_t lanes_before) {
  return warp_lane0_atomic_add(count, __popc(mask)) +
         __popc(mask & lanes_before);
}

__device__ __forceinline__ void histogram_fp32_top12(
    float value, int* histogram, int* nan_count) {
  const uint32_t bits = raw_bits(value);
  if (is_nan_key(bits)) atomicAdd(nan_count, 1);
  atomicAdd(&histogram[ordered_key(value) >> 20], 1);
}

template <typename OutIndexT>
__device__ __forceinline__ void compact_fp32_top12(
    float value, int index, OutIndexT* indices, float* row_values,
    bool return_value, int topk, uint32_t pivot12, int* histogram,
    int* output_count, int* candidate_count, int* candidate_indices,
    int candidate_capacity) {
  const uint32_t key = ordered_key(value);
  const uint32_t prefix12 = key >> 20;
  if (prefix12 > pivot12) {
    const int position = atomicAdd(output_count, 1);
    if (position < topk)
      store_topk_result(indices, row_values, return_value, position, index,
                        value);
  } else if (prefix12 == pivot12) {
    atomicAdd(&histogram[(key >> 12) & 0xFFu], 1);
    const int position = atomicAdd(candidate_count, 1);
    if (position < candidate_capacity) candidate_indices[position] = index;
  }
}

template <int kThreads, typename OutIndexT>
__global__ __launch_bounds__(
    kThreads,
    kThreads >= 1024 ? 1 : (kThreads >= 512 ? 2 : (kThreads >= 256 ? 8 : 12)))
void topk_fp32_kernel(const float* __restrict__ input,
                      OutIndexT* __restrict__ output, int64_t width,
                      int64_t input_stride, int64_t output_stride,
                      float* __restrict__ values, int64_t value_stride,
                      const int* __restrict__ end_ptr, OutIndexT idx_fill,
                      float value_fill, int topk, bool return_value) {
  __shared__ int histogram[kRadix];
  __shared__ int output_count;
  __shared__ int threshold[4];
  __shared__ int remain;
  __shared__ int nan_count;
  constexpr int kCandidateCapacity = 1024;
  __shared__ int candidate_count;
  __shared__ int candidate_indices[kCandidateCapacity];

  const int tid = static_cast<int>(threadIdx.x);
  const int row_index = static_cast<int>(blockIdx.x);
  const float* row = input + static_cast<int64_t>(row_index) * input_stride;
  const int64_t row_width = end_ptr ? end_ptr[row_index] : width;
  OutIndexT* indices =
      output + static_cast<int64_t>(row_index) * output_stride;
  float* row_values =
      values ? values + static_cast<int64_t>(row_index) * value_stride : nullptr;
  if (row_width <= topk) {
    for (int position = tid; position < topk; position += kThreads) {
      if (position < row_width)
        store_topk_result(indices, row_values, return_value, position,
                          position, row[position]);
      else {
        indices[position] = idx_fill;
        if (return_value) row_values[position] = value_fill;
      }
    }
    return;
  }
  for (int bin = tid; bin < kRadix; bin += kThreads) histogram[bin] = 0;
  if (tid == 0) {
    output_count = 0;
    threshold[0] = threshold[1] = threshold[2] = threshold[3] = 0;
    remain = topk;
    nan_count = 0;
    candidate_count = 0;
  }
  __syncthreads();

  for (int64_t index = tid; index < row_width; index += kThreads) {
    const float value = row[index];
    const uint32_t bits = raw_bits(value);
    if (is_nan_key(bits)) atomicAdd(&nan_count, 1);
    atomicAdd(&histogram[(ordered_key(value) >> 24) & 0xFF], 1);
  }
  __syncthreads();
  warp_select(histogram, topk, &threshold[3], &remain);
  __syncthreads();

  for (int64_t index = tid; index < row_width; index += kThreads) {
    const float value = row[index];
    const uint32_t key = ordered_key(value);
    const int b3 = (key >> 24) & 0xFF;
    if (b3 == threshold[3])
      atomicAdd(&histogram[(key >> 16) & 0xFF], 1);
  }
  __syncthreads();
  warp_select(histogram, remain, &threshold[2], &remain);
  __syncthreads();

  const uint32_t pivot_hi16 =
      (static_cast<uint32_t>(threshold[3]) << 8) |
      static_cast<uint32_t>(threshold[2]);
  for (int64_t index = tid; index < row_width; index += kThreads) {
    const float value = row[index];
    const uint32_t key = ordered_key(value);
    const uint32_t key_hi16 = key >> 16;
    if (key_hi16 > pivot_hi16) {
      const int position = atomicAdd(&output_count, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          value);
    } else if (key_hi16 == pivot_hi16) {
      atomicAdd(&histogram[(key >> 8) & 0xFF], 1);
      const int position = atomicAdd(&candidate_count, 1);
      if (position < kCandidateCapacity)
        candidate_indices[position] = static_cast<int>(index);
    }
  }
  __syncthreads();
  warp_select(histogram, remain, &threshold[1], &remain);
  __syncthreads();

  if (candidate_count <= kCandidateCapacity) {
    for (int position = tid; position < candidate_count;
         position += kThreads) {
      const int index = candidate_indices[position];
      const float value = row[index];
      const uint32_t key = ordered_key(value);
      const uint32_t byte1 = (key >> 8) & 0xFFu;
      if (byte1 > static_cast<uint32_t>(threshold[1])) {
        const int output_position = atomicAdd(&output_count, 1);
        if (output_position < topk)
          store_topk_result(indices, row_values, return_value,
                            output_position, index, value);
      } else if (byte1 == static_cast<uint32_t>(threshold[1])) {
        atomicAdd(&histogram[key & 0xFFu], 1);
      }
    }
    __syncthreads();
    warp_select(histogram, remain, &threshold[0], &remain);
    __syncthreads();

    for (int position = tid; position < candidate_count;
         position += kThreads) {
      const int index = candidate_indices[position];
      const float value = row[index];
      const uint32_t key = ordered_key(value);
      if (((key >> 8) & 0xFFu) !=
          static_cast<uint32_t>(threshold[1]))
        continue;
      const uint32_t byte0 = key & 0xFFu;
      if (byte0 > static_cast<uint32_t>(threshold[0])) {
        const int output_position = atomicAdd(&output_count, 1);
        if (output_position < topk)
          store_topk_result(indices, row_values, return_value,
                            output_position, index, value);
      } else if (byte0 == static_cast<uint32_t>(threshold[0])) {
        const int previous = atomicAdd(&remain, -1);
        if (previous > 0)
          store_topk_result(indices, row_values, return_value,
                            topk - previous, index, value);
      }
    }
    __syncthreads();
    if (tid == 0 && nan_count > 0)
      indices[0] = static_cast<OutIndexT>(kNanSentinel);
    return;
  }

  for (int64_t index = tid; index < row_width; index += kThreads) {
    const float value = row[index];
    const uint32_t key = ordered_key(value);
    if ((key >> 16) != pivot_hi16) continue;
    const uint32_t byte1 = (key >> 8) & 0xFFu;
    if (byte1 > static_cast<uint32_t>(threshold[1])) {
      const int position = atomicAdd(&output_count, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          value);
    } else if (byte1 == static_cast<uint32_t>(threshold[1])) {
      atomicAdd(&histogram[key & 0xFF], 1);
    }
  }
  __syncthreads();
  warp_select(histogram, remain, &threshold[0], &remain);
  __syncthreads();

  for (int64_t index = tid; index < row_width; index += kThreads) {
    const float value = row[index];
    const uint32_t key = ordered_key(value);
    if ((key >> 8) !=
        ((pivot_hi16 << 8) | static_cast<uint32_t>(threshold[1])))
      continue;
    const uint32_t byte0 = key & 0xFFu;
    if (byte0 > static_cast<uint32_t>(threshold[0])) {
      const int position = atomicAdd(&output_count, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          value);
    } else if (byte0 == static_cast<uint32_t>(threshold[0])) {
      const int previous = atomicAdd(&remain, -1);
      if (previous > 0)
        store_topk_result(indices, row_values, return_value, topk - previous,
                          index, value);
    }
  }
  __syncthreads();
  if (tid == 0 && nan_count > 0)
    indices[0] = static_cast<OutIndexT>(kNanSentinel);
}

constexpr int kValueSortFastMax = 1024;
constexpr int kFusedSortMax = 512;
constexpr int kValueSortBuckets = 256;
constexpr int kValueSortMaxBucket = 32;
constexpr int kValueSortScratchInts =
    3 * kValueSortFastMax + 3 * kValueSortBuckets + 3;
template <typename OutIndexT>
__device__ __forceinline__ void sort_selected_row_by_value(
    OutIndexT* __restrict__ indices, float* __restrict__ row_values,
    int valid_count, int topk, int* __restrict__ scratch);

template <typename OutIndexT, bool kParallelPrefix>
__device__ __forceinline__ void sort_selected_row_by_value_impl(
    OutIndexT* __restrict__ indices, float* __restrict__ row_values,
    int valid_count, int topk, int* __restrict__ scratch);

template <int kByte>
__device__ __forceinline__ bool fp32_sample_prefix_matches(
    uint32_t key, const int* thresholds) {
  if constexpr (kByte == 3) {
    return true;
  } else if constexpr (kByte == 2) {
    return (key >> 24) == static_cast<uint32_t>(thresholds[3]);
  } else if constexpr (kByte == 1) {
    const uint32_t prefix =
        (static_cast<uint32_t>(thresholds[3]) << 8) |
        static_cast<uint32_t>(thresholds[2]);
    return (key >> 16) == prefix;
  } else {
    const uint32_t prefix =
        (static_cast<uint32_t>(thresholds[3]) << 16) |
        (static_cast<uint32_t>(thresholds[2]) << 8) |
        static_cast<uint32_t>(thresholds[1]);
    return (key >> 8) == prefix;
  }
}

template <int kByte, bool kCheckNan, typename ElemT = float>
__device__ __forceinline__ void fp32_sample_radix_pass(
    const ElemT* values, int count, int* histogram, int* thresholds,
    int* remain, int* nan_count, int first_wanted) {
  const int tid = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);
  if constexpr (kByte == 3) {
    for (int bin = tid; bin < 256; bin += threads) histogram[bin] = 0;
    __syncthreads();
  }
  for (int position = tid; position < count; position += threads) {
    uint32_t key;
    if constexpr (std::is_same_v<ElemT, uint32_t>) {
      key = values[position];
      if constexpr (kCheckNan && kByte == 3) {
        if (is_nan_ordered_key(key)) atomicAdd(nan_count, 1);
      }
    } else {
      const float value = values[position];
      if constexpr (kCheckNan && kByte == 3) {
        if (is_nan_key(raw_bits(value))) atomicAdd(nan_count, 1);
      }
      key = ordered_key(value);
    }
    if (fp32_sample_prefix_matches<kByte>(key, thresholds))
      atomicAdd(&histogram[(key >> (8 * kByte)) & 0xFFu], 1);
  }
  __syncthreads();
  const int wanted = kByte == 3 ? first_wanted : *remain;
  warp_select(histogram, wanted, thresholds + kByte, remain);
  __syncthreads();
}

__device__ __forceinline__ void collect_fp32_sample_candidate(
    float value, int index, uint32_t sample_pivot, float* candidate_values,
    int* candidate_indices, int* candidate_count, int candidate_capacity,
    int* nan_count) {
  if (is_nan_key(raw_bits(value))) atomicAdd(nan_count, 1);
  if (ordered_key(value) >= sample_pivot) {
    const int position = atomicAdd(candidate_count, 1);
    if (position < candidate_capacity) {
      candidate_values[position] = value;
      candidate_indices[position] = index;
    }
  }
}

template <int kThreads, typename OutIndexT>
__global__ __launch_bounds__(kThreads, 1)
void topk_fp32_sample_survivor_kernel(
    const float* __restrict__ input, OutIndexT* __restrict__ output,
    int64_t width, int64_t input_stride, int64_t output_stride,
    float* __restrict__ values, int64_t value_stride, bool return_value,
    bool sort_by_value) {
  constexpr int kSampleElements = 16384;
  constexpr int kCandidateCapacity = 3072;
  constexpr int kCandidateTarget = 1536;

  union SampleStorage {
    struct Selection {
      float values[kCandidateCapacity];
      int indices[kCandidateCapacity];
      int histogram[256];
    } selection;
    int sort_scratch[4096];
  };
  __shared__ SampleStorage sample_storage;
  __shared__ int thresholds[4];
  __shared__ int candidate_count;
  __shared__ int output_count;
  __shared__ int remain;
  __shared__ int nan_count;
  float* candidate_values = sample_storage.selection.values;
  int* candidate_indices = sample_storage.selection.indices;
  int* histogram = sample_storage.selection.histogram;

  const int tid = static_cast<int>(threadIdx.x);
  const int row_index = static_cast<int>(blockIdx.x);
  const float* row = input + static_cast<int64_t>(row_index) * input_stride;
  OutIndexT* indices =
      output + static_cast<int64_t>(row_index) * output_stride;
  float* row_values =
      values ? values + static_cast<int64_t>(row_index) * value_stride : nullptr;
  const int sample_count = static_cast<int>(
      width < kSampleElements ? width : kSampleElements);
  if (tid == 0) {
    remain = static_cast<int>(
        (static_cast<int64_t>(kCandidateTarget) * sample_count + width - 1) /
        width);
    if (remain < 1) remain = 1;
    if (remain > sample_count) remain = sample_count;
    nan_count = 0;
  }
  __syncthreads();
  const int sample_wanted = remain;
  fp32_sample_radix_pass<3, true>(
      row, sample_count, histogram, thresholds, &remain, &nan_count,
      sample_wanted);
  fp32_sample_radix_pass<2, false>(
      row, sample_count, histogram, thresholds, &remain, &nan_count,
      sample_wanted);
  fp32_sample_radix_pass<1, false>(
      row, sample_count, histogram, thresholds, &remain, &nan_count,
      sample_wanted);
  fp32_sample_radix_pass<0, false>(
      row, sample_count, histogram, thresholds, &remain, &nan_count,
      sample_wanted);
  const uint32_t sample_pivot =
      (static_cast<uint32_t>(thresholds[3]) << 24) |
      (static_cast<uint32_t>(thresholds[2]) << 16) |
      (static_cast<uint32_t>(thresholds[1]) << 8) |
      static_cast<uint32_t>(thresholds[0]);

  if (tid == 0) candidate_count = 0;
  __syncthreads();
  const int64_t wide_width = width & ~int64_t{15};
  const bool row_is_aligned =
      (reinterpret_cast<uintptr_t>(row) & uintptr_t{15}) == 0;
  if (row_is_aligned) {
    for (int64_t index = static_cast<int64_t>(tid) * 16;
         index < wide_width;
         index += static_cast<int64_t>(kThreads) * 16) {
      const Floatx4 v0 = load_floatx4(row + index);
      const Floatx4 v1 = load_floatx4(row + index + 4);
      const Floatx4 v2 = load_floatx4(row + index + 8);
      const Floatx4 v3 = load_floatx4(row + index + 12);
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        collect_fp32_sample_candidate(
            v0.values[item], static_cast<int>(index) + item, sample_pivot,
            candidate_values, candidate_indices, &candidate_count,
            kCandidateCapacity, &nan_count);
        collect_fp32_sample_candidate(
            v1.values[item], static_cast<int>(index) + 4 + item, sample_pivot,
            candidate_values, candidate_indices, &candidate_count,
            kCandidateCapacity, &nan_count);
        collect_fp32_sample_candidate(
            v2.values[item], static_cast<int>(index) + 8 + item, sample_pivot,
            candidate_values, candidate_indices, &candidate_count,
            kCandidateCapacity, &nan_count);
        collect_fp32_sample_candidate(
            v3.values[item], static_cast<int>(index) + 12 + item, sample_pivot,
            candidate_values, candidate_indices, &candidate_count,
            kCandidateCapacity, &nan_count);
      }
    }
    for (int64_t index = wide_width + tid; index < width;
         index += kThreads)
      collect_fp32_sample_candidate(
          row[index], static_cast<int>(index), sample_pivot, candidate_values,
          candidate_indices, &candidate_count, kCandidateCapacity,
          &nan_count);
  } else {
    for (int64_t index = tid; index < width; index += kThreads)
      collect_fp32_sample_candidate(
          row[index], static_cast<int>(index), sample_pivot, candidate_values,
          candidate_indices, &candidate_count, kCandidateCapacity,
          &nan_count);
  }
  __syncthreads();

  if (candidate_count < 512 || candidate_count > kCandidateCapacity) {
    if (tid == 0)
      indices[0] = static_cast<OutIndexT>(kSampleFallbackSentinel);
    return;
  }

  if (tid == 0) remain = 512;
  __syncthreads();
  fp32_sample_radix_pass<3, false>(
      candidate_values, candidate_count, histogram, thresholds, &remain,
      &nan_count, 512);
  fp32_sample_radix_pass<2, false>(
      candidate_values, candidate_count, histogram, thresholds, &remain,
      &nan_count, 512);
  fp32_sample_radix_pass<1, false>(
      candidate_values, candidate_count, histogram, thresholds, &remain,
      &nan_count, 512);
  fp32_sample_radix_pass<0, false>(
      candidate_values, candidate_count, histogram, thresholds, &remain,
      &nan_count, 512);
  const uint32_t pivot_key =
      (static_cast<uint32_t>(thresholds[3]) << 24) |
      (static_cast<uint32_t>(thresholds[2]) << 16) |
      (static_cast<uint32_t>(thresholds[1]) << 8) |
      static_cast<uint32_t>(thresholds[0]);
  if (tid == 0) output_count = 0;
  __syncthreads();
  for (int position = tid; position < candidate_count;
       position += kThreads) {
    const float value = candidate_values[position];
    const uint32_t key = ordered_key(value);
    if (key > pivot_key) {
      const int output_position = atomicAdd(&output_count, 1);
      if (output_position < 512)
        store_topk_result(
            indices, row_values, return_value, output_position,
            candidate_indices[position], value);
    } else if (key == pivot_key) {
      const int previous = atomicAdd(&remain, -1);
      if (previous > 0)
        store_topk_result(
            indices, row_values, return_value, 512 - previous,
            candidate_indices[position], value);
    }
  }
  __syncthreads();
  if (tid == 0 && nan_count > 0)
    indices[0] = static_cast<OutIndexT>(kNanSentinel);
  __syncthreads();
  if (sort_by_value &&
      indices[0] != static_cast<OutIndexT>(kNanSentinel))
    sort_selected_row_by_value_impl<OutIndexT, true>(
        indices, row_values, 512, 512, sample_storage.sort_scratch);
}

template <int kThreads, int kCandidateCapacity, typename OutIndexT,
          bool kParallelValuePrefix = false,
          bool kSampleFallbackOnly = false>
__global__ __launch_bounds__(kThreads, 1)
void topk_fp32_compact12_kernel(
    const float* __restrict__ input, OutIndexT* __restrict__ output,
    int64_t width, int64_t input_stride, int64_t output_stride,
    float* __restrict__ values, int64_t value_stride,
    const int* __restrict__ end_ptr, OutIndexT idx_fill, float value_fill,
    int topk, bool return_value, bool sort_by_value) {
  static_assert(!kParallelValuePrefix || kThreads == kValueSortBuckets,
                "parallel value prefix requires one thread per bucket");
  constexpr int kRadix12 = 4096;
  __shared__ int histogram[kRadix12];
  __shared__ int candidate_indices[kCandidateCapacity];
  __shared__ int group_sums[256];
  __shared__ int output_count;
  __shared__ int candidate_count;
  __shared__ int threshold12;
  __shared__ int threshold20;
  __shared__ int threshold28;
  __shared__ int threshold32;
  __shared__ int remain;
  __shared__ int nan_count;
  __shared__ int group_threshold;
  __shared__ int group_remain;

  const int tid = static_cast<int>(threadIdx.x);
  const int row_index = static_cast<int>(blockIdx.x);
  const float* row = input + static_cast<int64_t>(row_index) * input_stride;
  const int64_t row_width = end_ptr ? end_ptr[row_index] : width;
  OutIndexT* indices =
      output + static_cast<int64_t>(row_index) * output_stride;
  float* row_values =
      values ? values + static_cast<int64_t>(row_index) * value_stride : nullptr;
  if constexpr (kSampleFallbackOnly) {
    if (indices[0] !=
        static_cast<OutIndexT>(kSampleFallbackSentinel))
      return;
  }
  if (row_width <= topk) {
    for (int position = tid; position < topk; position += kThreads) {
      if (position < row_width)
        store_topk_result(indices, row_values, return_value, position,
                          position, row[position]);
      else {
        indices[position] = idx_fill;
        if (return_value) row_values[position] = value_fill;
      }
    }
    if (sort_by_value) {
      __syncthreads();
      const int valid = static_cast<int>(row_width < topk ? row_width : topk);
      if (valid > 1 && indices[0] != static_cast<OutIndexT>(kNanSentinel)) {
        if constexpr (kParallelValuePrefix)
          sort_selected_row_by_value_impl<OutIndexT, true>(
              indices, row_values, valid, topk, histogram);
        else
          sort_selected_row_by_value<OutIndexT>(
              indices, row_values, valid, topk, histogram);
      }
    }
    return;
  }

  for (int bin = tid; bin < kRadix12; bin += kThreads)
    histogram[bin] = 0;
  if (tid == 0) {
    output_count = 0;
    candidate_count = 0;
    threshold12 = threshold20 = threshold28 = threshold32 = 0;
    remain = topk;
    nan_count = 0;
  }
  __syncthreads();

  const int64_t aligned_width = row_width & ~int64_t{3};
  const int64_t wide_width = row_width & ~int64_t{15};
  const bool row_is_aligned =
      (reinterpret_cast<uintptr_t>(row) & uintptr_t{15}) == 0;
  if (row_is_aligned) {
    for (int64_t index = static_cast<int64_t>(tid) * 16; index < wide_width;
         index += static_cast<int64_t>(kThreads) * 16) {
      const Floatx4 v0 = load_floatx4(row + index);
      const Floatx4 v1 = load_floatx4(row + index + 4);
      const Floatx4 v2 = load_floatx4(row + index + 8);
      const Floatx4 v3 = load_floatx4(row + index + 12);
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        histogram_fp32_top12(v0.values[item], histogram, &nan_count);
        histogram_fp32_top12(v1.values[item], histogram, &nan_count);
        histogram_fp32_top12(v2.values[item], histogram, &nan_count);
        histogram_fp32_top12(v3.values[item], histogram, &nan_count);
      }
    }
    for (int64_t index = wide_width + tid; index < row_width;
         index += kThreads)
      histogram_fp32_top12(row[index], histogram, &nan_count);
  } else {
    for (int64_t index = tid; index < row_width; index += kThreads)
      histogram_fp32_top12(row[index], histogram, &nan_count);
  }
  __syncthreads();
  block_select_4096(histogram, topk, &threshold12, &remain, group_sums,
                    &group_threshold, &group_remain);
  for (int bin = tid; bin < kRadix12; bin += kThreads)
    histogram[bin] = 0;
  __syncthreads();

  if (row_is_aligned) {
    for (int64_t index = static_cast<int64_t>(tid) * 16; index < wide_width;
         index += static_cast<int64_t>(kThreads) * 16) {
      const Floatx4 v0 = load_floatx4(row + index);
      const Floatx4 v1 = load_floatx4(row + index + 4);
      const Floatx4 v2 = load_floatx4(row + index + 8);
      const Floatx4 v3 = load_floatx4(row + index + 12);
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        compact_fp32_top12(
            v0.values[item], static_cast<int>(index) + item, indices,
            row_values, return_value, topk,
            static_cast<uint32_t>(threshold12), histogram, &output_count,
            &candidate_count, candidate_indices, kCandidateCapacity);
        compact_fp32_top12(
            v1.values[item], static_cast<int>(index) + 4 + item, indices,
            row_values, return_value, topk,
            static_cast<uint32_t>(threshold12), histogram, &output_count,
            &candidate_count, candidate_indices, kCandidateCapacity);
        compact_fp32_top12(
            v2.values[item], static_cast<int>(index) + 8 + item, indices,
            row_values, return_value, topk,
            static_cast<uint32_t>(threshold12), histogram, &output_count,
            &candidate_count, candidate_indices, kCandidateCapacity);
        compact_fp32_top12(
            v3.values[item], static_cast<int>(index) + 12 + item, indices,
            row_values, return_value, topk,
            static_cast<uint32_t>(threshold12), histogram, &output_count,
            &candidate_count, candidate_indices, kCandidateCapacity);
      }
    }
    for (int64_t index = wide_width + tid; index < row_width;
         index += kThreads)
      compact_fp32_top12(
          row[index], static_cast<int>(index), indices, row_values,
          return_value, topk, static_cast<uint32_t>(threshold12), histogram,
          &output_count, &candidate_count, candidate_indices,
          kCandidateCapacity);
  } else {
    for (int64_t index = tid; index < row_width; index += kThreads)
      compact_fp32_top12(
          row[index], static_cast<int>(index), indices, row_values,
          return_value, topk, static_cast<uint32_t>(threshold12), histogram,
          &output_count, &candidate_count, candidate_indices,
          kCandidateCapacity);
  }
  __syncthreads();
  warp_select(histogram, remain, &threshold20, &remain);
  __syncthreads();

  if (candidate_count <= kCandidateCapacity) {
    for (int position = tid; position < candidate_count;
         position += kThreads) {
      const int index = candidate_indices[position];
      const float value = row[index];
      const uint32_t key = ordered_key(value);
      const uint32_t byte20 = (key >> 12) & 0xFFu;
      if (byte20 > static_cast<uint32_t>(threshold20)) {
        const int output_position = atomicAdd(&output_count, 1);
        if (output_position < topk)
          store_topk_result(indices, row_values, return_value,
                            output_position, index, value);
      } else if (byte20 == static_cast<uint32_t>(threshold20)) {
        atomicAdd(&histogram[key & 0xFFFu], 1);
      }
    }
    __syncthreads();
    block_select_4096(histogram, remain, &threshold32, &remain, group_sums,
                      &group_threshold, &group_remain);

    for (int position = tid; position < candidate_count;
         position += kThreads) {
      const int index = candidate_indices[position];
      const float value = row[index];
      const uint32_t key = ordered_key(value);
      if (((key >> 12) & 0xFFu) !=
          static_cast<uint32_t>(threshold20))
        continue;
      const uint32_t low12 = key & 0xFFFu;
      if (low12 > static_cast<uint32_t>(threshold32)) {
        const int output_position = atomicAdd(&output_count, 1);
        if (output_position < topk)
          store_topk_result(indices, row_values, return_value,
                            output_position, index, value);
      } else if (low12 == static_cast<uint32_t>(threshold32)) {
        const int previous = atomicAdd(&remain, -1);
        if (previous > 0)
          store_topk_result(indices, row_values, return_value,
                            topk - previous, index, value);
      }
    }
    __syncthreads();
    if (tid == 0 && nan_count > 0)
      indices[0] = static_cast<OutIndexT>(kNanSentinel);
    if (sort_by_value) {
      __syncthreads();
      const int valid = static_cast<int>(row_width < topk ? row_width : topk);
      if (valid > 1 && indices[0] != static_cast<OutIndexT>(kNanSentinel)) {
        if constexpr (kParallelValuePrefix)
          sort_selected_row_by_value_impl<OutIndexT, true>(
              indices, row_values, valid, topk, histogram);
        else
          sort_selected_row_by_value<OutIndexT>(
              indices, row_values, valid, topk, histogram);
      }
    }
    return;
  }

  const uint32_t pivot20 =
      (static_cast<uint32_t>(threshold12) << 8) |
      static_cast<uint32_t>(threshold20);
  for (int64_t index = tid; index < row_width; index += kThreads) {
    const float value = row[index];
    const uint32_t key = ordered_key(value);
    const uint32_t prefix20 = key >> 12;
    if (prefix20 > pivot20 &&
        (key >> 20) == static_cast<uint32_t>(threshold12)) {
      const int position = atomicAdd(&output_count, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          value);
    } else if (prefix20 == pivot20) {
      atomicAdd(&histogram[(key >> 4) & 0xFFu], 1);
    }
  }
  __syncthreads();
  warp_select(histogram, remain, &threshold28, &remain);
  __syncthreads();

  for (int64_t index = tid; index < row_width; index += kThreads) {
    const float value = row[index];
    const uint32_t key = ordered_key(value);
    if ((key >> 12) != pivot20) continue;
    const uint32_t byte28 = (key >> 4) & 0xFFu;
    if (byte28 > static_cast<uint32_t>(threshold28)) {
      const int position = atomicAdd(&output_count, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          value);
    } else if (byte28 == static_cast<uint32_t>(threshold28)) {
      atomicAdd(&histogram[key & 0xFu], 1);
    }
  }
  __syncthreads();
  warp_select(histogram, remain, &threshold32, &remain);
  __syncthreads();

  const uint32_t pivot28 =
      (pivot20 << 8) | static_cast<uint32_t>(threshold28);
  for (int64_t index = tid; index < row_width; index += kThreads) {
    const float value = row[index];
    const uint32_t key = ordered_key(value);
    if ((key >> 4) != pivot28) continue;
    const uint32_t low4 = key & 0xFu;
    if (low4 > static_cast<uint32_t>(threshold32)) {
      const int position = atomicAdd(&output_count, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          value);
    } else if (low4 == static_cast<uint32_t>(threshold32)) {
      const int previous = atomicAdd(&remain, -1);
      if (previous > 0)
        store_topk_result(indices, row_values, return_value, topk - previous,
                          index, value);
    }
  }
  __syncthreads();
  if (tid == 0 && nan_count > 0)
    indices[0] = static_cast<OutIndexT>(kNanSentinel);
  if (sort_by_value) {
    __syncthreads();
    const int valid = static_cast<int>(row_width < topk ? row_width : topk);
    if (valid > 1 && indices[0] != static_cast<OutIndexT>(kNanSentinel)) {
      if constexpr (kParallelValuePrefix)
        sort_selected_row_by_value_impl<OutIndexT, true>(
            indices, row_values, valid, topk, histogram);
      else
        sort_selected_row_by_value<OutIndexT>(
            indices, row_values, valid, topk, histogram);
    }
  }
}

template <typename OutIndexT>
bool launch_topk_fp32_typed(
    const float* input_ptr, OutIndexT* output_ptr, int rows, int64_t width,
    int64_t input_stride, int64_t output_stride, float* values,
    int64_t value_stride, const int* end_ptr, OutIndexT idx_fill,
    float value_fill, int topk, bool return_value, bool sort_by_value,
    musaStream_t stream) {
  const bool use_sample_survivor =
      rows >= 4096 && width >= 129280 && topk == 512 && end_ptr == nullptr &&
      input_stride == width;
  if (use_sample_survivor) {
    topk_fp32_sample_survivor_kernel<256, OutIndexT>
        <<<rows, 256, 0, stream>>>(
            input_ptr, output_ptr, width, input_stride, output_stride, values,
            value_stride, return_value, sort_by_value);
    topk_fp32_compact12_kernel<256, 1024, OutIndexT, true, true>
        <<<rows, 256, 0, stream>>>(
            input_ptr, output_ptr, width, input_stride, output_stride, values,
            value_stride, end_ptr, idx_fill, value_fill, topk, return_value,
            sort_by_value);
    return sort_by_value;
  }
  const int threads = select_threads(rows, width);
  const bool use_compact12 =
      (width >= 16384 && topk % 32 == 0 && compact12_topk_ok(topk)) ||
      (rows > 8 && width >= 65536 && compact12_topk_ok(topk));
  const bool fuse_sort = sort_by_value && topk <= kFusedSortMax;
  if (use_compact12) {
    if (rows <= 1024) {
      if (threads == 1024)
        topk_fp32_compact12_kernel<1024, 4096, OutIndexT>
            <<<rows, 1024, 0, stream>>>(
                input_ptr, output_ptr, width, input_stride, output_stride,
                values, value_stride, end_ptr, idx_fill, value_fill, topk,
                return_value, fuse_sort);
      else
        topk_fp32_compact12_kernel<512, 4096, OutIndexT>
            <<<rows, 512, 0, stream>>>(
                input_ptr, output_ptr, width, input_stride, output_stride,
                values, value_stride, end_ptr, idx_fill, value_fill, topk,
                return_value, fuse_sort);
    } else {
      if (rows >= 4096)
        topk_fp32_compact12_kernel<256, 1024, OutIndexT, true>
            <<<rows, 256, 0, stream>>>(
                input_ptr, output_ptr, width, input_stride, output_stride,
                values, value_stride, end_ptr, idx_fill, value_fill, topk,
                return_value, fuse_sort);
      else
        topk_fp32_compact12_kernel<256, 1024, OutIndexT>
            <<<rows, 256, 0, stream>>>(
                input_ptr, output_ptr, width, input_stride, output_stride,
                values, value_stride, end_ptr, idx_fill, value_fill, topk,
                return_value, fuse_sort);
    }
    return fuse_sort;
  }
  if (threads == 1024)
    topk_fp32_kernel<1024, OutIndexT><<<rows, 1024, 0, stream>>>(
        input_ptr, output_ptr, width, input_stride, output_stride, values,
        value_stride, end_ptr, idx_fill, value_fill, topk, return_value);
  else if (threads == 512)
    topk_fp32_kernel<512, OutIndexT><<<rows, 512, 0, stream>>>(
        input_ptr, output_ptr, width, input_stride, output_stride, values,
        value_stride, end_ptr, idx_fill, value_fill, topk, return_value);
  else if (threads == 256)
    topk_fp32_kernel<256, OutIndexT><<<rows, 256, 0, stream>>>(
        input_ptr, output_ptr, width, input_stride, output_stride, values,
        value_stride, end_ptr, idx_fill, value_fill, topk, return_value);
  else
    topk_fp32_kernel<128, OutIndexT><<<rows, 128, 0, stream>>>(
        input_ptr, output_ptr, width, input_stride, output_stride, values,
        value_stride, end_ptr, idx_fill, value_fill, topk, return_value);
  return false;
}

constexpr int kTwoStageThreads = 256;
constexpr int kTwoStageCapacity = 768;
constexpr int kTwoStageMaxParts = 16;
constexpr int kTwoStageMaxTopK = 2048;
constexpr int kTwoStageSmallTopK = 512;
constexpr int kTwoStageMedTopK = 1024;

struct TwoStagePart {
  int strict_count;
  int equal_count;
  int has_nan;
  int strict_cols[kTwoStageCapacity];
  uint32_t strict_keys[kTwoStageCapacity];
  int equal_cols[kTwoStageCapacity];
};
static_assert(sizeof(TwoStagePart) % sizeof(int) == 0);
constexpr int kTwoStageWords =
    static_cast<int>(sizeof(TwoStagePart) / sizeof(int));

template <int kSamplesVal, int kFinishCapacityVal, bool kPrecheckVal,
          bool kAllowFusedSortVal, int kSigmaTenthsVal>
struct TwoStagePolicy {
  static constexpr int kSamples = kSamplesVal;
  static constexpr int kFinishCapacity = kFinishCapacityVal;
  static constexpr bool kPrecheck = kPrecheckVal;
  static constexpr bool kAllowFusedSort = kAllowFusedSortVal;
  static constexpr float kSigma = static_cast<float>(kSigmaTenthsVal) * 0.1f;
};

using TwoStageSmallPolicy = TwoStagePolicy<2048, 4096, false, true, 25>;
using TwoStageMedPolicy = TwoStagePolicy<4096, 8192, true, true, 35>;
using TwoStageLargePolicy = TwoStagePolicy<8192, 8192, true, false, 35>;

template <typename Policy>
__global__ void topk_fp32_scan_partitions(
    const float* input, int width, int64_t stride, int topk, int* workspace) {
  const int tid = threadIdx.x;
  const float* row = input + static_cast<int64_t>(blockIdx.y) * stride;
  TwoStagePart* part =
      reinterpret_cast<TwoStagePart*>(workspace) +
      static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  __shared__ uint32_t sample[Policy::kSamples];
  __shared__ int hist[kRadix], thresholds[4], remain;
  __shared__ int strict_count, equal_count, has_nan;
  if (tid == 0) strict_count = equal_count = has_nan = 0;
  for (int i = tid; i < Policy::kSamples; i += blockDim.x) {
    const int lo = static_cast<int64_t>(i) * width / Policy::kSamples;
    const int hi = static_cast<int64_t>(i + 1) * width / Policy::kSamples;
    const uint32_t h = static_cast<uint32_t>(i) * 0x9e3779b9u;
    const int offset =
        static_cast<uint64_t>(h ^ (h >> 16)) * (hi - lo) >> 32;
    sample[i] = ordered_key(row[lo + offset]);
  }
  __syncthreads();

  const float mean = static_cast<float>(topk * Policy::kSamples) / width;
  const int rank = min(
      Policy::kSamples,
      max(1, static_cast<int>(mean + Policy::kSigma * sqrtf(mean + 1)) + 2));
  fp32_sample_radix_pass<3, false, uint32_t>(
      sample, Policy::kSamples, hist, thresholds, &remain, nullptr, rank);
  fp32_sample_radix_pass<2, false, uint32_t>(
      sample, Policy::kSamples, hist, thresholds, &remain, nullptr, rank);
  fp32_sample_radix_pass<1, false, uint32_t>(
      sample, Policy::kSamples, hist, thresholds, &remain, nullptr, rank);
  fp32_sample_radix_pass<0, false, uint32_t>(
      sample, Policy::kSamples, hist, thresholds, &remain, nullptr, rank);
  const uint32_t pivot = (static_cast<uint32_t>(thresholds[3]) << 24) |
      (thresholds[2] << 16) | (thresholds[1] << 8) | thresholds[0];

  const uint32_t lanes_before = (1u << (tid & 31)) - 1;
  const auto collect = [&](int col, uint32_t key, bool valid) {
    const bool nan = valid && is_nan_ordered_key(key);
    const bool strict = valid && !nan && key > pivot;
    const bool equal = valid && key == pivot;
    if (nan) atomicOr(&has_nan, 1);

    const uint32_t hits = __ballot_sync(
        0xffffffffu, Policy::kPrecheck ? (strict || equal) : strict);
    if constexpr (Policy::kPrecheck) {
      if (!hits) return;
    }
    const uint32_t equal_mask = __ballot_sync(0xffffffffu, equal);
    const uint32_t strict_mask =
        Policy::kPrecheck ? (hits & ~equal_mask) : hits;
    if (strict_mask) {
      const int pos = warp_reserve(&strict_count, strict_mask, lanes_before);
      if (strict && pos < kTwoStageCapacity) {
        part->strict_cols[pos] = col;
        part->strict_keys[pos] = key;
      }
    }
    if (equal_mask) {
      const int pos = warp_reserve(&equal_count, equal_mask, lanes_before);
      if (equal && pos < kTwoStageCapacity) part->equal_cols[pos] = col;
    }
  };
  const int vecs = width / 4;
  const bool aligned = (reinterpret_cast<uintptr_t>(row) & 15) == 0;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warps = blockDim.x >> 5;
  for (int tile = blockIdx.x * warps + warp; tile * 32 < vecs;
       tile += gridDim.x * warps) {
    const int i = tile * 32 + lane;
    const bool in_bounds = i < vecs;
    Floatx4 v{};
    if (in_bounds) {
      if (aligned) v = load_floatx4(row + 4 * i);
      else for (int j = 0; j < 4; ++j) v.values[j] = row[4 * i + j];
    }
    uint32_t keys[4];
    bool any = false;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      keys[j] = ordered_key(v.values[j]);
      if constexpr (Policy::kPrecheck) {
        any |= keys[j] >= pivot || is_nan_ordered_key(keys[j]);
      }
    }
    bool run_warp = true;
    if constexpr (Policy::kPrecheck) {
      run_warp = __ballot_sync(0xffffffffu, in_bounds && any) != 0;
    }
    if (run_warp) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) collect(4 * i + j, keys[j], in_bounds);
    }
  }
  __syncthreads();
  if (tid == 0) {
    part->strict_count = strict_count;
    part->equal_count = equal_count;
    part->has_nan = has_nan;
  }
}

template <typename OutIndexT, typename Policy>
__global__ void topk_fp32_finish_partitions(
    const float* input, int width, int64_t stride, OutIndexT* output,
    int64_t output_stride, float* values, int64_t value_stride, int topk,
    bool sort_by_value, const int* workspace, int parts) {
  const int tid = threadIdx.x;
  const float* row = input + static_cast<int64_t>(blockIdx.x) * stride;
  OutIndexT* indices = output + static_cast<int64_t>(blockIdx.x) * output_stride;
  float* row_values =
      values ? values + static_cast<int64_t>(blockIdx.x) * value_stride : nullptr;
  const TwoStagePart* row_parts =
      reinterpret_cast<const TwoStagePart*>(workspace) +
      static_cast<int64_t>(blockIdx.x) * parts;
  __shared__ int offsets[kTwoStageMaxParts + 1], eq_offsets[kTwoStageMaxParts + 1];
  __shared__ int exact_full, fill_ties, has_nan;
  __shared__ int thresholds[4], remain, output_count;
  union FinishStorage {
    struct {
      int hist[kRadix];
      uint32_t keys[Policy::kFinishCapacity];
      int cols[Policy::kFinishCapacity];
    } sel;
    int sort_scratch[Policy::kAllowFusedSort ? kValueSortScratchInts : 1];
  };
  __shared__ FinishStorage storage;

  storage.sel.hist[tid] = 0;
  if (tid == 0) {
    offsets[0] = eq_offsets[0] = 0;
    exact_full = has_nan = output_count = 0;
    for (int p = 0; p < parts; ++p) {
      const TwoStagePart& part = row_parts[p];
      exact_full |= part.strict_count > kTwoStageCapacity;
      has_nan |= part.has_nan;
      offsets[p + 1] = offsets[p] + min(part.strict_count, kTwoStageCapacity);
      eq_offsets[p + 1] =
          eq_offsets[p] + min(part.equal_count, kTwoStageCapacity);
    }
    const int strict = offsets[parts];
    exact_full |= strict > Policy::kFinishCapacity ||
                  strict + eq_offsets[parts] < topk;
    fill_ties = !exact_full && strict <= topk;
  }
  __syncthreads();

  if (has_nan) {
    if (tid == 0) indices[0] = static_cast<OutIndexT>(kNanSentinel);
    return;
  }
  if (exact_full) {
    if (tid == 0) indices[0] = static_cast<OutIndexT>(kSampleFallbackSentinel);
    return;
  }

  const int count = offsets[parts];

  if (fill_ties) {
    for (int p = 0; p < parts; ++p) {
      const TwoStagePart& part = row_parts[p];
      const int n_strict = offsets[p + 1] - offsets[p];
      for (int s = tid; s < n_strict; s += blockDim.x) {
        const int col = part.strict_cols[s];
        store_topk_result(
            indices, row_values, values != nullptr, offsets[p] + s, col,
            values ? value_from_ordered_key(part.strict_keys[s]) : 0.0f);
      }
    }
    const int strict_total = offsets[parts];
    for (int p = 0; p < parts; ++p) {
      const int rem = topk - (strict_total + eq_offsets[p]);
      if (rem <= 0) break;
      const TwoStagePart& part = row_parts[p];
      const int take = min(rem, eq_offsets[p + 1] - eq_offsets[p]);
      for (int e = tid; e < take; e += blockDim.x) {
        const int col = part.equal_cols[e];
        store_topk_result(
            indices, row_values, values != nullptr,
            strict_total + eq_offsets[p] + e, col, values ? row[col] : 0.0f);
      }
    }
  } else {
    for (int p = 0; p < parts; ++p) {
      const TwoStagePart& part = row_parts[p];
      for (int i = offsets[p] + tid; i < offsets[p + 1]; i += blockDim.x) {
        const int slot = i - offsets[p];
        storage.sel.cols[i] = part.strict_cols[slot];
        storage.sel.keys[i] = part.strict_keys[slot];
      }
    }
    __syncthreads();
    uint32_t prefix = 0;
    uint32_t mask = 0;
    for (int byte = 3; byte >= 0; --byte) {
      for (int i = tid; i < count; i += blockDim.x) {
        const uint32_t key = storage.sel.keys[i];
        if ((key & mask) == prefix)
          atomicAdd(storage.sel.hist + ((key >> (8 * byte)) & 255), 1);
      }
      __syncthreads();
      warp_select(storage.sel.hist, byte == 3 ? topk : remain,
                  thresholds + byte, &remain);
      __syncthreads();
      const uint32_t byte_mask = 255u << (8 * byte);
      mask |= byte_mask;
      prefix |= static_cast<uint32_t>(thresholds[byte]) << (8 * byte);
    }
    const int rounds = (count + blockDim.x - 1) / blockDim.x;
    const uint32_t lanes_before = (1u << (tid & 31)) - 1;
    for (int round = 0; round < rounds; ++round) {
      const int i = round * blockDim.x + tid;
      const bool valid = i < count;
      const int col = valid ? storage.sel.cols[i] : 0;
      const uint32_t key = valid ? storage.sel.keys[i] : 0;
      const uint32_t strict_mask =
          __ballot_sync(0xffffffffu, valid && key > prefix);
      if (strict_mask) {
        const int pos = warp_reserve(&output_count, strict_mask, lanes_before);
        if (valid && key > prefix) {
          store_topk_result(
              indices, row_values, values != nullptr, pos, col,
              values ? value_from_ordered_key(key) : 0.0f);
        }
      }
      const uint32_t equal_mask =
          __ballot_sync(0xffffffffu, valid && key == prefix);
      if (equal_mask) {
        const int left = warp_lane0_atomic_add(&remain, -__popc(equal_mask));
        if (valid && key == prefix) {
          const int pos = topk - left + __popc(equal_mask & lanes_before);
          if (pos < topk) {
            store_topk_result(
                indices, row_values, values != nullptr, pos, col,
                values ? value_from_ordered_key(key) : 0.0f);
          }
        }
      }
    }
  }
  if constexpr (Policy::kAllowFusedSort) {
    if (sort_by_value) {
      __syncthreads();
      sort_selected_row_by_value_impl<OutIndexT, true>(
          indices, row_values, topk, topk, storage.sort_scratch);
    }
  }
}

int64_t two_stage_scratch_ints(int64_t rows, int64_t parts) {
  TORCH_CHECK(rows > 0 && rows <= 65535 && parts >= 1 && parts <= kTwoStageMaxParts,
              "invalid two-stage grid");
  return rows * parts * kTwoStageWords;
}

template <typename OutIndexT, typename Policy>
void launch_two_stage_partitions(
    const at::Tensor& input, at::Tensor& output, float* values,
    int64_t value_stride, bool sort_by_value, bool abort_when_nan_found,
    int* workspace, int parts, musaStream_t stream) {
  const dim3 grid(parts, input.size(0));
  topk_fp32_scan_partitions<Policy><<<grid, kTwoStageThreads, 0, stream>>>(
      input.data_ptr<float>(), input.size(1), input.stride(0), output.size(1),
      workspace);
  auto* indices = output.data_ptr<OutIndexT>();
  const int rows = input.size(0);
  const int topk = output.size(1);
  topk_fp32_finish_partitions<OutIndexT, Policy>
      <<<rows, kTwoStageThreads, 0, stream>>>(
          input.data_ptr<float>(), input.size(1), input.stride(0), indices,
          output.stride(0), values, value_stride, topk, sort_by_value,
          workspace, parts);
  topk_fp32_compact12_kernel<1024, 4096, OutIndexT, false, true>
      <<<rows, 1024, 0, stream>>>(
          input.data_ptr<float>(), indices, input.size(1), input.stride(0),
          output.stride(0), values, value_stride, nullptr, OutIndexT{0}, 0.0f,
          topk, values != nullptr, sort_by_value);
  if (abort_when_nan_found) launch_nan_guard(indices, rows, output.stride(0), stream);
}

template <typename OutIndexT>
void dispatch_two_stage_partitions(
    const at::Tensor& input, at::Tensor& output, float* values,
    int64_t value_stride, bool sort_by_value, bool abort_when_nan_found,
    int* workspace, int parts, musaStream_t stream) {
  const int topk = output.size(1);
  if (topk > kTwoStageMedTopK) {
    launch_two_stage_partitions<OutIndexT, TwoStageLargePolicy>(
        input, output, values, value_stride, sort_by_value, abort_when_nan_found,
        workspace, parts, stream);
  } else if (topk > kTwoStageSmallTopK) {
    launch_two_stage_partitions<OutIndexT, TwoStageMedPolicy>(
        input, output, values, value_stride, sort_by_value, abort_when_nan_found,
        workspace, parts, stream);
  } else {
    launch_two_stage_partitions<OutIndexT, TwoStageSmallPolicy>(
        input, output, values, value_stride, sort_by_value, abort_when_nan_found,
        workspace, parts, stream);
  }
}

void launch_topk_fp32_two_stage(
    const at::Tensor& input, at::Tensor& output, at::Tensor& scratch,
    const c10::optional<at::Tensor>& output_value, bool sort_by_value,
    bool abort_when_nan_found, int parts) {
  const bool return_value = output_value.has_value();
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                  input.dim() == 2 && input.scalar_type() == at::kFloat &&
                  input.stride(1) == 1 && input.size(1) % 4 == 0 &&
                  input.size(1) < (1 << 23),
              "invalid two-stage input");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == input.size(0) &&
                  output.size(1) > 0 && output.size(1) <= kTwoStageMaxTopK &&
                  output.stride(1) == 1 &&
                  (output.scalar_type() == at::kLong || output.scalar_type() == at::kInt) &&
                  output.device() == input.device(),
              "invalid two-stage indices");
  const int64_t topk = output.size(1);
  const int min_samples = topk > kTwoStageMedTopK
                              ? TwoStageLargePolicy::kSamples
                              : (topk > kTwoStageSmallTopK
                                     ? TwoStageMedPolicy::kSamples
                                     : TwoStageSmallPolicy::kSamples);
  TORCH_CHECK(input.size(1) >= min_samples, "invalid two-stage input");
  TORCH_CHECK(scratch.scalar_type() == at::kInt && scratch.is_contiguous() &&
                  scratch.device() == input.device() &&
                  scratch.numel() >= two_stage_scratch_ints(input.size(0), parts),
              "invalid two-stage scratch");
  TORCH_CHECK(!sort_by_value || (return_value && output.size(1) <= kTwoStageMedTopK),
              "invalid two-stage value flags");
  TORCH_CHECK(!return_value ||
                  (output_value->scalar_type() == at::kFloat &&
                   output_value->device() == input.device() &&
                   output_value->sizes() == output.sizes() && output_value->stride(1) == 1),
              "invalid two-stage values");

  at::musa::MUSAGuard guard(input.device());
  auto stream = at::musa::getCurrentMUSAStream(input.get_device()).stream();
  float* values = return_value ? output_value->data_ptr<float>() : nullptr;
  const int64_t value_stride = return_value ? output_value->stride(0) : 0;
  int* workspace = scratch.data_ptr<int>();
  if (output.scalar_type() == at::kLong) {
    dispatch_two_stage_partitions<int64_t>(input, output, values, value_stride, sort_by_value,
                                    abort_when_nan_found, workspace, parts, stream);
  } else {
    dispatch_two_stage_partitions<int32_t>(input, output, values, value_stride, sort_by_value,
                                    abort_when_nan_found, workspace, parts, stream);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

struct Fp32MultiScratch {
  int* partial_hist;
  int* partial_nan;
  int* thresholds;
  int* remain;
  int* output_count;
  int* has_nan;
};

Fp32MultiScratch split_fp32_multi_scratch(
    int* scratch, int rows, int parts) {
  const int64_t hist_size =
      static_cast<int64_t>(rows) * parts * kRadix;
  const int64_t partial_size = static_cast<int64_t>(rows) * parts;
  Fp32MultiScratch result;
  result.partial_hist = scratch;
  result.partial_nan = result.partial_hist + hist_size;
  result.thresholds = result.partial_nan + partial_size;
  result.remain = result.thresholds + 4 * rows;
  result.output_count = result.remain + rows;
  result.has_nan = result.output_count + rows;
  return result;
}

template <int kByte>
__device__ __forceinline__ bool fp32_prefix_matches(
    uint32_t key, const int* thresholds) {
  if constexpr (kByte < 3) {
    if (((key >> 24) & 0xFFu) !=
        static_cast<uint32_t>(thresholds[3]))
      return false;
  }
  if constexpr (kByte < 2) {
    if (((key >> 16) & 0xFFu) !=
        static_cast<uint32_t>(thresholds[2]))
      return false;
  }
  if constexpr (kByte < 1) {
    if (((key >> 8) & 0xFFu) !=
        static_cast<uint32_t>(thresholds[1]))
      return false;
  }
  return true;
}

template <int kByte>
__global__ void partial_fp32_histogram_kernel(
    const float* __restrict__ input, int* __restrict__ partial_hist,
    int* __restrict__ partial_nan, const int* __restrict__ thresholds,
    const int* __restrict__ has_nan, int64_t width, int parts) {
  __shared__ int histogram[kRadix];
  __shared__ int nan_count;
  const int tid = static_cast<int>(threadIdx.x);
  const int task = static_cast<int>(blockIdx.x);
  const int row_index = task / parts;
  if constexpr (kByte < 3) {
    if (has_nan[row_index]) return;
  }
  const int part = task - row_index * parts;
  const int64_t begin = width * part / parts;
  const int64_t end = width * (part + 1) / parts;
  const float* row = input + static_cast<int64_t>(row_index) * width;
  const int* row_thresholds = thresholds + 4 * row_index;
  for (int bin = tid; bin < kRadix; bin += blockDim.x)
    histogram[bin] = 0;
  if (tid == 0) nan_count = 0;
  __syncthreads();

  for (int64_t index = begin + tid; index < end; index += blockDim.x) {
    const float value = row[index];
    const uint32_t bits = raw_bits(value);
    if constexpr (kByte == 3) {
      if (is_nan_key(bits)) atomicAdd(&nan_count, 1);
    }
    const uint32_t key = ordered_key(value);
    if (fp32_prefix_matches<kByte>(key, row_thresholds))
      atomicAdd(&histogram[(key >> (8 * kByte)) & 0xFFu], 1);
  }
  __syncthreads();

  int* destination =
      partial_hist + static_cast<int64_t>(task) * kRadix;
  for (int bin = tid; bin < kRadix; bin += blockDim.x)
    destination[bin] = histogram[bin];
  if constexpr (kByte == 3) {
    if (tid == 0) partial_nan[task] = nan_count;
  }
}

template <int kByte, typename OutIndexT>
__global__ void reduce_fp32_histogram_kernel(
    const int* __restrict__ partial_hist,
    const int* __restrict__ partial_nan, OutIndexT* __restrict__ output,
    int* __restrict__ thresholds, int* __restrict__ remain,
    int* __restrict__ output_count, int* __restrict__ has_nan,
    int64_t output_stride, int parts, int topk) {
  __shared__ int histogram[kRadix];
  const int tid = static_cast<int>(threadIdx.x);
  const int row_index = static_cast<int>(blockIdx.x);
  if constexpr (kByte < 3) {
    if (has_nan[row_index]) return;
  }
  if (tid < kRadix) {
    int count = 0;
    for (int part = 0; part < parts; ++part)
      count += partial_hist[
          (static_cast<int64_t>(row_index) * parts + part) * kRadix + tid];
    histogram[tid] = count;
  }
  if constexpr (kByte == 3) {
    if (tid == 0) {
      int row_nan_count = 0;
      for (int part = 0; part < parts; ++part)
        row_nan_count += partial_nan[row_index * parts + part];
      has_nan[row_index] = row_nan_count != 0;
      output_count[row_index] = 0;
      remain[row_index] = topk;
      if (row_nan_count != 0)
        output[static_cast<int64_t>(row_index) * output_stride] =
            static_cast<OutIndexT>(kNanSentinel);
    }
  }
  __syncthreads();
  if (has_nan[row_index]) return;
  const int wanted = kByte == 3 ? topk : remain[row_index];
  warp_select(histogram, wanted, thresholds + 4 * row_index + kByte,
              remain + row_index);
}

template <typename OutIndexT>
__global__ void finish_fp32_multi_kernel(
    const float* __restrict__ input, OutIndexT* __restrict__ output,
    const int* __restrict__ thresholds, int* __restrict__ remain,
    int* __restrict__ output_count, const int* __restrict__ has_nan,
    int64_t width, int64_t output_stride, float* __restrict__ values,
    int64_t value_stride, int parts, int topk, bool return_value) {
  const int tid = static_cast<int>(threadIdx.x);
  const int task = static_cast<int>(blockIdx.x);
  const int row_index = task / parts;
  if (has_nan[row_index]) return;
  const int part = task - row_index * parts;
  const int64_t begin = width * part / parts;
  const int64_t end = width * (part + 1) / parts;
  const float* row = input + static_cast<int64_t>(row_index) * width;
  OutIndexT* indices =
      output + static_cast<int64_t>(row_index) * output_stride;
  float* row_values =
      values ? values + static_cast<int64_t>(row_index) * value_stride : nullptr;
  const int* row_thresholds = thresholds + 4 * row_index;
  const uint32_t pivot_key =
      (static_cast<uint32_t>(row_thresholds[3]) << 24) |
      (static_cast<uint32_t>(row_thresholds[2]) << 16) |
      (static_cast<uint32_t>(row_thresholds[1]) << 8) |
      static_cast<uint32_t>(row_thresholds[0]);

  for (int64_t index = begin + tid; index < end; index += blockDim.x) {
    const float value = row[index];
    const uint32_t key = ordered_key(value);
    if (key > pivot_key) {
      const int position = atomicAdd(output_count + row_index, 1);
      if (position < topk)
        store_topk_result(indices, row_values, return_value, position, index,
                          value);
    } else if (key == pivot_key) {
      const int previous = atomicAdd(remain + row_index, -1);
      if (previous > 0)
        store_topk_result(indices, row_values, return_value, topk - previous,
                          index, value);
    }
  }
}

template <typename OutIndexT>
void launch_topk_fp32_multi_typed(
    const float* input_ptr, OutIndexT* output_ptr,
    const Fp32MultiScratch& buffers, int rows, int parts,
    int worker_threads, int64_t width, int64_t output_stride, float* values,
    int64_t value_stride, int topk, bool return_value, musaStream_t stream) {
  const int tasks = rows * parts;
  partial_fp32_histogram_kernel<3><<<tasks, worker_threads, 0, stream>>>(
      input_ptr, buffers.partial_hist, buffers.partial_nan,
      buffers.thresholds, buffers.has_nan, width, parts);
  reduce_fp32_histogram_kernel<3, OutIndexT><<<rows, 256, 0, stream>>>(
      buffers.partial_hist, buffers.partial_nan, output_ptr,
      buffers.thresholds, buffers.remain, buffers.output_count,
      buffers.has_nan, output_stride, parts, topk);
  partial_fp32_histogram_kernel<2><<<tasks, worker_threads, 0, stream>>>(
      input_ptr, buffers.partial_hist, buffers.partial_nan,
      buffers.thresholds, buffers.has_nan, width, parts);
  reduce_fp32_histogram_kernel<2, OutIndexT><<<rows, 256, 0, stream>>>(
      buffers.partial_hist, buffers.partial_nan, output_ptr,
      buffers.thresholds, buffers.remain, buffers.output_count,
      buffers.has_nan, output_stride, parts, topk);
  partial_fp32_histogram_kernel<1><<<tasks, worker_threads, 0, stream>>>(
      input_ptr, buffers.partial_hist, buffers.partial_nan,
      buffers.thresholds, buffers.has_nan, width, parts);
  reduce_fp32_histogram_kernel<1, OutIndexT><<<rows, 256, 0, stream>>>(
      buffers.partial_hist, buffers.partial_nan, output_ptr,
      buffers.thresholds, buffers.remain, buffers.output_count,
      buffers.has_nan, output_stride, parts, topk);
  partial_fp32_histogram_kernel<0><<<tasks, worker_threads, 0, stream>>>(
      input_ptr, buffers.partial_hist, buffers.partial_nan,
      buffers.thresholds, buffers.has_nan, width, parts);
  reduce_fp32_histogram_kernel<0, OutIndexT><<<rows, 256, 0, stream>>>(
      buffers.partial_hist, buffers.partial_nan, output_ptr,
      buffers.thresholds, buffers.remain, buffers.output_count,
      buffers.has_nan, output_stride, parts, topk);
  finish_fp32_multi_kernel<OutIndexT><<<tasks, worker_threads, 0, stream>>>(
      input_ptr, output_ptr, buffers.thresholds, buffers.remain,
      buffers.output_count, buffers.has_nan, width, output_stride, values,
      value_stride, parts, topk, return_value);
}

void launch_topk_fp32_multi(
    const at::Tensor& input, at::Tensor& output, at::Tensor& scratch,
    int64_t parts64, int64_t worker_threads64,
    const c10::optional<at::Tensor>& output_value = c10::nullopt,
    bool abort_when_nan_found = true) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
              "input must be a MUSA tensor");
  TORCH_CHECK(output.device() == input.device() &&
                  scratch.device() == input.device(),
              "output/scratch device mismatch");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK((output.scalar_type() == at::kInt ||
               output.scalar_type() == at::kLong) &&
                  scratch.scalar_type() == at::kInt,
              "output must be int32/int64; scratch must be int32");
  TORCH_CHECK(input.is_contiguous() && output.stride(1) == 1 &&
                  scratch.is_contiguous(),
              "MUSA FP32 multi-CTA topk requires contiguous input/scratch and "
              "output last dim");
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2 && scratch.dim() == 1,
              "invalid tensor ranks");
  const int topk = static_cast<int>(output.size(1));
  TORCH_CHECK(!output_value.has_value() ||
                  (output_value->device() == input.device() &&
                   output_value->scalar_type() == at::kFloat &&
                   output_value->dim() == 2 &&
                   output_value->size(0) == input.size(0) &&
                   output_value->size(1) == topk &&
                   output_value->stride(1) == 1),
              "output_value must be FP32 [batch, topk]");
  TORCH_CHECK(output.size(0) == input.size(0) && topk > 0 &&
                  topk <= kMaxTopK && input.size(1) > topk &&
                  input.size(1) < (1 << 23),
              "invalid FP32 multi-CTA shape");
  TORCH_CHECK(parts64 > 1 && parts64 <= 64, "parts must be in [2, 64]");
  TORCH_CHECK(worker_threads64 == 128 || worker_threads64 == 256 ||
                  worker_threads64 == 512,
              "worker_threads must be 128, 256, or 512");

  const int rows = static_cast<int>(input.size(0));
  if (rows == 0) return;
  const int parts = static_cast<int>(parts64);
  const int worker_threads = static_cast<int>(worker_threads64);
  const int64_t required =
      static_cast<int64_t>(rows) * parts * kRadix +
      static_cast<int64_t>(rows) * parts + 7LL * rows;
  TORCH_CHECK(scratch.numel() >= required, "scratch tensor is too small");
  const int64_t width = input.size(1);
  const int64_t output_stride = output.stride(0);
  const int64_t value_stride =
      output_value.has_value() ? output_value->stride(0) : 0;
  float* value_ptr =
      output_value.has_value() ? output_value->data_ptr<float>() : nullptr;
  const Fp32MultiScratch buffers = split_fp32_multi_scratch(
      scratch.data_ptr<int>(), rows, parts);
  at::musa::MUSAGuard guard(input.device());
  const auto stream = at::musa::getCurrentMUSAStream(input.device().index());
  if (output.scalar_type() == at::kInt)
    launch_topk_fp32_multi_typed(
        input.data_ptr<float>(), output.data_ptr<int32_t>(), buffers, rows,
        parts, worker_threads, width, output_stride, value_ptr, value_stride,
        topk, output_value.has_value(), stream);
  else
    launch_topk_fp32_multi_typed(
        input.data_ptr<float>(), output.data_ptr<int64_t>(), buffers, rows,
        parts, worker_threads, width, output_stride, value_ptr, value_stride,
        topk, output_value.has_value(), stream);
  if (abort_when_nan_found) {
    if (output.scalar_type() == at::kInt)
      launch_nan_guard(output.data_ptr<int32_t>(), rows, output_stride, stream);
    else
      launch_nan_guard(output.data_ptr<int64_t>(), rows, output_stride, stream);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

template <typename OutIndexT>
__global__ void output_idx_offset_kernel(
    OutIndexT* __restrict__ output, int64_t output_stride,
    const int* __restrict__ end_ptr, const int* __restrict__ offset_ptr,
    int full_length, int topk) {
  const int row_index = static_cast<int>(blockIdx.x);
  int valid_count = end_ptr ? end_ptr[row_index] : full_length;
  if (valid_count > topk) valid_count = topk;
  OutIndexT* row = output + static_cast<int64_t>(row_index) * output_stride;
  const int64_t offset = offset_ptr[row_index];
  for (int position = static_cast<int>(threadIdx.x); position < valid_count;
       position += static_cast<int>(blockDim.x)) {
    const OutIndexT index = row[position];
    if (index != static_cast<OutIndexT>(kNanSentinel))
      row[position] = static_cast<OutIndexT>(
          static_cast<int64_t>(index) + offset);
  }
}

void launch_output_idx_offset(
    at::Tensor& output, const c10::optional<at::Tensor>& end,
    const at::Tensor& offset, int64_t full_length64) {
  TORCH_CHECK(output.device().type() == c10::DeviceType::PrivateUse1 &&
                  offset.device() == output.device() &&
                  (!end.has_value() || end->device() == output.device()),
              "output/end/offset must be MUSA tensors on the same device");
  TORCH_CHECK(output.dim() == 2 && output.stride(1) == 1 &&
                  (output.scalar_type() == at::kInt ||
                   output.scalar_type() == at::kLong),
              "output must be contiguous-last-dim int32/int64 [batch, topk]");
  TORCH_CHECK(offset.scalar_type() == at::kInt && offset.dim() == 1 &&
                  offset.size(0) == output.size(0) && offset.is_contiguous(),
              "offset must be contiguous int32 [batch]");
  TORCH_CHECK(!end.has_value() ||
                  (end->scalar_type() == at::kInt && end->dim() == 1 &&
                   end->size(0) == output.size(0) && end->is_contiguous()),
              "end must be contiguous int32 [batch]");
  const int rows = static_cast<int>(output.size(0));
  if (rows == 0) return;
  const int topk = static_cast<int>(output.size(1));
  TORCH_CHECK(full_length64 > 0 && full_length64 < (1 << 23),
              "full_length must be in (0, 2^23)");
  const int full_length = static_cast<int>(full_length64);
  const int* end_ptr = end.has_value() ? end->data_ptr<int>() : nullptr;
  const int* offset_ptr = offset.data_ptr<int>();
  at::musa::MUSAGuard guard(output.device());
  const auto stream = at::musa::getCurrentMUSAStream(output.device().index());
  if (output.scalar_type() == at::kInt)
    output_idx_offset_kernel<int32_t><<<rows, 256, 0, stream>>>(
        output.data_ptr<int32_t>(), output.stride(0), end_ptr, offset_ptr,
        full_length, topk);
  else
    output_idx_offset_kernel<int64_t><<<rows, 256, 0, stream>>>(
        output.data_ptr<int64_t>(), output.stride(0), end_ptr, offset_ptr,
        full_length, topk);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}


constexpr int kSortBucketBits = 8;
constexpr int kSortBuckets = 1 << kSortBucketBits;
constexpr int kSortBucketMask = kSortBuckets - 1;
constexpr int kMaxBucket = 32;

template <typename OutIndexT, typename ValueT, bool kCarryValue, int kTopkBuf>
__global__ void sort_output_offset_kernel(
    OutIndexT* __restrict__ output, int64_t output_stride,
    ValueT* __restrict__ values, int64_t value_stride,
    const int* __restrict__ end_ptr, const int* __restrict__ offset_ptr,
    int full_length, int topk) {
  __shared__ int keys[kTopkBuf];
  __shared__ int staged[kTopkBuf];
  __shared__ int sorted[kTopkBuf];
  __shared__ ValueT value_in[kCarryValue ? kTopkBuf : 1];
  __shared__ ValueT value_out[kCarryValue ? kTopkBuf : 1];
  __shared__ int histogram[kSortBuckets + 1];
  __shared__ int cursor[kSortBuckets];
  __shared__ int bucket_limit;
  const int threads = static_cast<int>(blockDim.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int row_index = static_cast<int>(blockIdx.x);
  int valid_count = end_ptr ? end_ptr[row_index] : full_length;
  if (valid_count > topk) valid_count = topk;
  if (valid_count <= 0) return;
  OutIndexT* row = output + static_cast<int64_t>(row_index) * output_stride;
  const int64_t offset =
      offset_ptr ? static_cast<int64_t>(offset_ptr[row_index]) : 0;
  ValueT* row_values =
      kCarryValue ? values + static_cast<int64_t>(row_index) * value_stride
                  : nullptr;
  if constexpr (kCarryValue) {
    for (int position = tid; position < valid_count; position += threads)
      value_in[position] = row_values[position];
  }
  __syncthreads();
  if (row[0] == static_cast<OutIndexT>(kNanSentinel)) {
    for (int position = 1 + tid; position < valid_count; position += threads)
      row[position] = static_cast<OutIndexT>(
          static_cast<int64_t>(row[position]) + offset);
    return;
  }
  if (valid_count == 1) {
    row[0] = static_cast<OutIndexT>(static_cast<int64_t>(row[0]) + offset);
    return;
  }
  if (full_length > kSortBuckets * 256) {
    for (int position = tid; position < valid_count; position += threads)
      keys[position] = static_cast<int>(row[position]);
    __syncthreads();
    for (int sequence = 2; sequence <= kTopkBuf; sequence <<= 1) {
      for (int distance = sequence >> 1; distance > 0; distance >>= 1) {
        for (int position = tid; position < kTopkBuf; position += threads) {
          const int partner = position ^ distance;
          if (partner <= position) continue;
          const bool ascending = (position & sequence) == 0;
          const int left = position < valid_count ? keys[position] : INT_MAX;
          const int right = partner < valid_count ? keys[partner] : INT_MAX;
          if (partner < valid_count &&
              ((ascending && left > right) || (!ascending && left < right))) {
            keys[position] = right;
            keys[partner] = left;
            if constexpr (kCarryValue) {
              const ValueT carried = value_in[position];
              value_in[position] = value_in[partner];
              value_in[partner] = carried;
            }
          }
        }
        __syncthreads();
      }
    }
    for (int position = tid; position < valid_count; position += threads) {
      row[position] = static_cast<OutIndexT>(
          static_cast<int64_t>(keys[position]) + offset);
      if constexpr (kCarryValue) row_values[position] = value_in[position];
    }
    return;
  }
  for (int bucket = tid; bucket <= kSortBuckets; bucket += threads)
    histogram[bucket] = 0;
  if (tid == 0) bucket_limit = 0;
  __syncthreads();
  for (int position = tid; position < valid_count; position += threads) {
    const int key = static_cast<int>(row[position]);
    keys[position] = key;
    atomicAdd(&histogram[key >> kSortBucketBits], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int running = 0;
    int largest = 0;
    for (int bucket = 0; bucket < kSortBuckets; ++bucket) {
      const int count = histogram[bucket];
      histogram[bucket] = running;
      cursor[bucket] = running;
      running += count;
      largest = count > largest ? count : largest;
    }
    bucket_limit = largest;
  }
  __syncthreads();
  if (bucket_limit > kMaxBucket) {
    for (int position = tid; position < kTopkBuf; position += threads)
      staged[position] = position < valid_count ? keys[position] : INT_MAX;
    __syncthreads();
    for (int position = tid; position < valid_count; position += threads) {
      const int self = staged[position];
      int rank = 0;
      for (int other = 0; other < kTopkBuf; ++other) {
        const int value = staged[other];
        rank += static_cast<int>(value < self) +
            static_cast<int>(value == self && other < position);
      }
      sorted[rank] = self;
      if constexpr (kCarryValue) value_out[rank] = value_in[position];
    }
    __syncthreads();
  } else {
    for (int position = tid; position < valid_count; position += threads) {
      const int key = keys[position];
      const int slot = atomicAdd(&cursor[key >> kSortBucketBits], 1);
      staged[slot] = key;
    }
    __syncthreads();
    for (int position = tid; position < valid_count; position += threads) {
      const int key = keys[position];
      const int bucket = key >> kSortBucketBits;
      const int low = key & kSortBucketMask;
      const int begin = histogram[bucket];
      const int end = cursor[bucket];
      int rank = 0;
      for (int slot = begin; slot < end; ++slot)
        rank += static_cast<int>((staged[slot] & kSortBucketMask) < low);
      sorted[begin + rank] = key;
      if constexpr (kCarryValue) value_out[begin + rank] = value_in[position];
    }
    __syncthreads();
  }
  for (int position = tid; position < valid_count; position += threads) {
    row[position] = static_cast<OutIndexT>(
        static_cast<int64_t>(sorted[position]) + offset);
    if constexpr (kCarryValue) row_values[position] = value_out[position];
  }
}

template <typename OutIndexT, int kTopkBuf>
static void launch_sort_output_offset_typed(
    at::Tensor& output, const c10::optional<at::Tensor>& output_value,
    const int* end_ptr, const int* offset_ptr, int full_length, int topk,
    int threads, musaStream_t stream) {
  OutIndexT* out_ptr = output.data_ptr<OutIndexT>();
  const int64_t out_stride = output.stride(0);
  const int rows = static_cast<int>(output.size(0));
  if (!output_value.has_value()) {
    sort_output_offset_kernel<OutIndexT, int, false, kTopkBuf>
        <<<rows, threads, 0, stream>>>(
            out_ptr, out_stride, static_cast<int*>(nullptr), 0, end_ptr,
            offset_ptr, full_length, topk);
  } else if (output_value->scalar_type() == at::kFloat) {
    sort_output_offset_kernel<OutIndexT, float, true, kTopkBuf>
        <<<rows, threads, 0, stream>>>(
            out_ptr, out_stride, output_value->data_ptr<float>(),
            output_value->stride(0), end_ptr, offset_ptr, full_length, topk);
  } else {
    sort_output_offset_kernel<OutIndexT, __mt_bfloat16, true, kTopkBuf>
        <<<rows, threads, 0, stream>>>(
            out_ptr, out_stride,
            reinterpret_cast<__mt_bfloat16*>(output_value->data_ptr()),
            output_value->stride(0), end_ptr, offset_ptr, full_length, topk);
  }
}

void launch_sort_output_fused(
    at::Tensor& output, const c10::optional<at::Tensor>& output_value,
    const c10::optional<at::Tensor>& end,
    const c10::optional<at::Tensor>& offset, int64_t full_length64) {
  TORCH_CHECK(output.device().type() == c10::DeviceType::PrivateUse1 &&
                  (!end.has_value() || end->device() == output.device()) &&
                  (!offset.has_value() || offset->device() == output.device()),
              "output/end/offset must be MUSA tensors on the same device");
  TORCH_CHECK(output.dim() == 2 && output.stride(1) == 1 &&
                  (output.scalar_type() == at::kInt ||
                   output.scalar_type() == at::kLong),
              "output must be contiguous-last-dim int32/int64 [batch, topk]");
  TORCH_CHECK(!output_value.has_value() ||
                  (output_value->dim() == 2 &&
                   output_value->sizes() == output.sizes() &&
                   output_value->stride(1) == 1 &&
                   (output_value->scalar_type() == at::kBFloat16 ||
                    output_value->scalar_type() == at::kFloat)),
              "output_value must be BF16/FP32 [batch, topk]");
  TORCH_CHECK(!end.has_value() ||
                  (end->scalar_type() == at::kInt && end->dim() == 1 &&
                   end->size(0) == output.size(0) && end->is_contiguous()),
              "end must be contiguous int32 [batch]");
  TORCH_CHECK(!offset.has_value() ||
                  (offset->scalar_type() == at::kInt && offset->dim() == 1 &&
                   offset->size(0) == output.size(0) && offset->is_contiguous()),
              "offset must be contiguous int32 [batch]");
  const int rows = static_cast<int>(output.size(0));
  if (rows == 0) return;
  const int topk = static_cast<int>(output.size(1));
  TORCH_CHECK(topk <= 1024, "fused epilogue supports topk <= 1024");
  TORCH_CHECK(full_length64 > 0 && full_length64 < (1 << 23),
              "full_length must be in (0, 2^23)");
  const int full_length = static_cast<int>(full_length64);
  const int* end_ptr = end.has_value() ? end->data_ptr<int>() : nullptr;
  const int* offset_ptr = offset.has_value() ? offset->data_ptr<int>() : nullptr;
  const int threads = topk >= 512 ? 512 : (topk >= 256 ? 256 : 128);
  at::musa::MUSAGuard guard(output.device());
  const auto stream = at::musa::getCurrentMUSAStream(output.device().index());
  if (output.scalar_type() == at::kInt) {
    if (topk <= 512)
      launch_sort_output_offset_typed<int32_t, 512>(
          output, output_value, end_ptr, offset_ptr, full_length, topk, threads,
          stream);
    else
      launch_sort_output_offset_typed<int32_t, 1024>(
          output, output_value, end_ptr, offset_ptr, full_length, topk, threads,
          stream);
  } else {
    if (topk <= 512)
      launch_sort_output_offset_typed<int64_t, 512>(
          output, output_value, end_ptr, offset_ptr, full_length, topk, threads,
          stream);
    else
      launch_sort_output_offset_typed<int64_t, 1024>(
          output, output_value, end_ptr, offset_ptr, full_length, topk, threads,
          stream);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

template <typename OutIndexT, typename ValueT, bool kSortByValue,
          bool kCarryValue>
__global__ void sort_output_kernel(
    OutIndexT* __restrict__ output, int64_t output_stride,
    ValueT* __restrict__ values, int64_t value_stride,
    const int* __restrict__ end_ptr, int full_length, int topk) {
  __shared__ OutIndexT shared_indices[kMaxTopK];
  __shared__ ValueT shared_values[kCarryValue ? kMaxTopK : 1];

  const int row_index = static_cast<int>(blockIdx.x);
  int valid_count = end_ptr ? end_ptr[row_index] : full_length;
  if (valid_count > topk) valid_count = topk;
  if (valid_count <= 1) return;
  OutIndexT* row = output + static_cast<int64_t>(row_index) * output_stride;
  if (row[0] == static_cast<OutIndexT>(kNanSentinel)) return;
  ValueT* row_values =
      kCarryValue ? values + static_cast<int64_t>(row_index) * value_stride
                  : nullptr;

  int sort_width = 1;
  while (sort_width < valid_count) sort_width <<= 1;
  for (int position = static_cast<int>(threadIdx.x); position < sort_width;
       position += static_cast<int>(blockDim.x)) {
    if (position < valid_count) {
      shared_indices[position] = row[position];
      if constexpr (kCarryValue)
        shared_values[position] = row_values[position];
    } else if constexpr (kSortByValue) {
      shared_indices[position] = static_cast<OutIndexT>(-1);
      shared_values[position] = -INFINITY;
    } else {
      shared_indices[position] = std::numeric_limits<OutIndexT>::max();
      if constexpr (kCarryValue) shared_values[position] = ValueT{};
    }
  }
  __syncthreads();

  for (int sequence = 2; sequence <= sort_width; sequence <<= 1) {
    for (int distance = sequence >> 1; distance > 0; distance >>= 1) {
      for (int position = static_cast<int>(threadIdx.x);
           position < sort_width;
           position += static_cast<int>(blockDim.x)) {
        const int partner = position ^ distance;
        if (partner <= position) continue;
        bool ascending = (position & sequence) == 0;
        if constexpr (kSortByValue) ascending = !ascending;

        bool greater;
        bool less;
        if constexpr (kSortByValue) {
          const ValueT left = shared_values[position];
          const ValueT right = shared_values[partner];
          const bool left_valid = shared_indices[position] >= 0;
          const bool right_valid = shared_indices[partner] >= 0;
          greater = left > right ||
              (left == right && left_valid > right_valid);
          less = left < right ||
              (left == right && left_valid < right_valid);
        } else {
          greater = shared_indices[position] > shared_indices[partner];
          less = shared_indices[position] < shared_indices[partner];
        }
        if ((ascending && greater) || (!ascending && less)) {
          const OutIndexT index = shared_indices[position];
          shared_indices[position] = shared_indices[partner];
          shared_indices[partner] = index;
          if constexpr (kCarryValue) {
            const ValueT value = shared_values[position];
            shared_values[position] = shared_values[partner];
            shared_values[partner] = value;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int position = static_cast<int>(threadIdx.x); position < valid_count;
       position += static_cast<int>(blockDim.x)) {
    row[position] = shared_indices[position];
    if constexpr (kCarryValue)
      row_values[position] = shared_values[position];
  }
}



__device__ __forceinline__ unsigned long long value_sort_pack(
    unsigned key, int index) {
  return (static_cast<unsigned long long>(key) << 24) |
         (static_cast<unsigned long long>(index) << 1) | 1ull;
}

__device__ __forceinline__ int value_sort_bucket(
    unsigned key, unsigned low, unsigned span) {
  if (span == 0u) return 0;
  const unsigned long long scaled =
      (static_cast<unsigned long long>(key - low) *
       static_cast<unsigned long long>(kValueSortBuckets - 1)) /
      span;
  return static_cast<int>(scaled);
}

template <typename OutIndexT, bool kParallelPrefix>
__device__ __forceinline__ void sort_selected_row_by_value_impl(
    OutIndexT* __restrict__ indices, float* __restrict__ row_values,
    int valid_count, int topk, int* __restrict__ scratch) {
  unsigned* s_key = reinterpret_cast<unsigned*>(scratch);
  int* s_idx = scratch + kValueSortFastMax;
  int* s_staged = scratch + 2 * kValueSortFastMax;
  int* s_hist = scratch + 3 * kValueSortFastMax;
  int* s_cursor = s_hist + kValueSortBuckets;
  int* s_prefix = s_cursor + kValueSortBuckets;
  unsigned* s_low = reinterpret_cast<unsigned*>(s_prefix + kValueSortBuckets);
  unsigned* s_high = s_low + 1;
  int* s_max_bucket = reinterpret_cast<int*>(s_low + 2);

  const int tid = static_cast<int>(threadIdx.x);
  const int block_threads = static_cast<int>(blockDim.x);
  if (tid == 0) {
    *s_low = 0xFFFFFFFFu;
    *s_high = 0u;
    *s_max_bucket = 0;
  }
  for (int b = tid; b < kValueSortBuckets; b += block_threads) s_hist[b] = 0;
  __syncthreads();

  for (int i = tid; i < valid_count; i += block_threads) {
    const unsigned key = ordered_key(row_values[i]);
    s_key[i] = key;
    s_idx[i] = static_cast<int>(indices[i]);
    atomicMin(s_low, key);
    atomicMax(s_high, key);
  }
  __syncthreads();

  const unsigned low = *s_low;
  const unsigned span = *s_high - low;
  for (int i = tid; i < valid_count; i += block_threads)
    atomicAdd(&s_hist[value_sort_bucket(s_key[i], low, span)], 1);
  __syncthreads();

  if constexpr (kParallelPrefix) {
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int count = tid < kValueSortBuckets ? s_hist[tid] : 0;
    int inclusive = count;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xFFFFFFFFu, inclusive, offset);
      if (lane >= offset) inclusive += other;
    }
    if (lane == 31) s_staged[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
      int warp_prefix = lane < (kValueSortBuckets / 32) ? s_staged[lane] : 0;
      const int warp_count = warp_prefix;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xFFFFFFFFu, warp_prefix, offset);
        if (lane >= offset) warp_prefix += other;
      }
      if (lane < (kValueSortBuckets / 32))
        s_staged[lane] = warp_prefix - warp_count;
    }
    __syncthreads();
    if (tid < kValueSortBuckets) {
      const int prefix = s_staged[warp] + inclusive - count;
      s_prefix[tid] = prefix;
      s_cursor[tid] = prefix;
      atomicMax(s_max_bucket, count);
      if (tid == kValueSortBuckets - 1)
        s_prefix[kValueSortBuckets] = prefix + count;
    }
  } else {
    if (tid == 0) {
      int running = 0;
      for (int b = 0; b < kValueSortBuckets; ++b) {
        s_prefix[b] = running;
        running += s_hist[b];
      }
      s_prefix[kValueSortBuckets] = running;
    }
    __syncthreads();
    for (int b = tid; b < kValueSortBuckets; b += block_threads) {
      s_cursor[b] = s_prefix[b];
      atomicMax(s_max_bucket, s_hist[b]);
    }
  }
  __syncthreads();

  if (*s_max_bucket <= kValueSortMaxBucket) {
    for (int i = tid; i < valid_count; i += block_threads) {
      const int b = value_sort_bucket(s_key[i], low, span);
      const int position = atomicAdd(&s_cursor[b], 1);
      s_staged[position] = i;
    }
    __syncthreads();
    for (int p = tid; p < valid_count; p += block_threads) {
      const int i = s_staged[p];
      const unsigned key = s_key[i];
      const unsigned long long packed = value_sort_pack(key, s_idx[i]);
      const int b = value_sort_bucket(key, low, span);
      const int begin = s_prefix[b];
      const int bucket_end = s_prefix[b] + s_hist[b];
      int greater_in_bucket = 0;
      for (int q = begin; q < bucket_end; ++q) {
        const int j = s_staged[q];
        if (value_sort_pack(s_key[j], s_idx[j]) > packed) ++greater_in_bucket;
      }
      const int position = (valid_count - bucket_end) + greater_in_bucket;
      indices[position] = static_cast<OutIndexT>(s_idx[i]);
      row_values[position] = value_from_ordered_key(key);
    }
  } else {
    for (int i = tid; i < valid_count; i += block_threads) {
      const unsigned long long packed = value_sort_pack(s_key[i], s_idx[i]);
      int greater = 0;
      for (int j = 0; j < valid_count; ++j)
        if (value_sort_pack(s_key[j], s_idx[j]) > packed) ++greater;
      indices[greater] = static_cast<OutIndexT>(s_idx[i]);
      row_values[greater] = value_from_ordered_key(s_key[i]);
    }
  }
}

template <typename OutIndexT>
__device__ __forceinline__ void sort_selected_row_by_value(
    OutIndexT* __restrict__ indices, float* __restrict__ row_values,
    int valid_count, int topk, int* __restrict__ scratch) {
  sort_selected_row_by_value_impl<OutIndexT, false>(
      indices, row_values, valid_count, topk, scratch);
}

template <typename OutIndexT>
__global__ void sort_value_output_bucket_kernel(
    OutIndexT* __restrict__ output, int64_t output_stride,
    float* __restrict__ values, int64_t value_stride,
    const int* __restrict__ end_ptr, int full_length, int topk) {
  __shared__ unsigned s_key[kValueSortFastMax];
  __shared__ OutIndexT s_idx[kValueSortFastMax];
  __shared__ int s_staged[kValueSortFastMax];
  __shared__ int s_hist[kValueSortBuckets];
  __shared__ int s_cursor[kValueSortBuckets];
  __shared__ int s_prefix[kValueSortBuckets + 1];
  __shared__ unsigned s_low, s_high;
  __shared__ int s_max_bucket;

  const int tid = static_cast<int>(threadIdx.x);
  const int row_index = static_cast<int>(blockIdx.x);
  int valid_count = end_ptr ? end_ptr[row_index] : full_length;
  if (valid_count > topk) valid_count = topk;
  if (valid_count <= 1) return;
  OutIndexT* row = output + static_cast<int64_t>(row_index) * output_stride;
  if (row[0] == static_cast<OutIndexT>(kNanSentinel)) return;
  float* row_values = values + static_cast<int64_t>(row_index) * value_stride;

  if (tid == 0) {
    s_low = 0xFFFFFFFFu;
    s_high = 0u;
    s_max_bucket = 0;
  }
  for (int b = tid; b < kValueSortBuckets; b += blockDim.x) s_hist[b] = 0;
  __syncthreads();

  for (int i = tid; i < valid_count; i += blockDim.x) {
    const unsigned key = ordered_key(row_values[i]);
    s_key[i] = key;
    s_idx[i] = row[i];
    atomicMin(&s_low, key);
    atomicMax(&s_high, key);
  }
  __syncthreads();

  const unsigned low = s_low;
  const unsigned span = s_high - s_low;
  for (int i = tid; i < valid_count; i += blockDim.x)
    atomicAdd(&s_hist[value_sort_bucket(s_key[i], low, span)], 1);
  __syncthreads();

  if (tid == 0) {
    int running = 0;
    for (int b = 0; b < kValueSortBuckets; ++b) {
      s_prefix[b] = running;
      running += s_hist[b];
    }
    s_prefix[kValueSortBuckets] = running;
  }
  __syncthreads();
  for (int b = tid; b < kValueSortBuckets; b += blockDim.x) {
    s_cursor[b] = s_prefix[b];
    atomicMax(&s_max_bucket, s_hist[b]);
  }
  __syncthreads();

  if (s_max_bucket <= kValueSortMaxBucket) {
    for (int i = tid; i < valid_count; i += blockDim.x) {
      const int b = value_sort_bucket(s_key[i], low, span);
      const int position = atomicAdd(&s_cursor[b], 1);
      s_staged[position] = i;
    }
    __syncthreads();
    for (int p = tid; p < valid_count; p += blockDim.x) {
      const int i = s_staged[p];
      const unsigned key = s_key[i];
      const unsigned long long packed = value_sort_pack(key, s_idx[i]);
      const int b = value_sort_bucket(key, low, span);
      const int begin = s_prefix[b];
      const int bucket_end = s_prefix[b] + s_hist[b];
      int greater_in_bucket = 0;
      for (int q = begin; q < bucket_end; ++q) {
        const int j = s_staged[q];
        if (value_sort_pack(s_key[j], s_idx[j]) > packed) ++greater_in_bucket;
      }
      const int position = (valid_count - bucket_end) + greater_in_bucket;
      row[position] = s_idx[i];
      row_values[position] = value_from_ordered_key(key);
    }
  } else {
    for (int i = tid; i < valid_count; i += blockDim.x) {
      const unsigned long long packed = value_sort_pack(s_key[i], s_idx[i]);
      int greater = 0;
      for (int j = 0; j < valid_count; ++j)
        if (value_sort_pack(s_key[j], s_idx[j]) > packed) ++greater;
      row[greater] = s_idx[i];
      row_values[greater] = value_from_ordered_key(s_key[i]);
    }
  }
}

template <typename OutIndexT, int kThreads, int kItemsPerThread>
__global__ void radix_sort_packed_value_output_kernel(
    OutIndexT* __restrict__ output, int64_t output_stride,
    float* __restrict__ values, int64_t value_stride,
    const int* __restrict__ end_ptr, int full_length, int topk) {
  using BlockRadixSort =
      cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread>;
  __shared__ typename BlockRadixSort::TempStorage sort_storage;

  const int row_index = static_cast<int>(blockIdx.x);
  int valid_count = end_ptr ? end_ptr[row_index] : full_length;
  if (valid_count > topk) valid_count = topk;
  if (valid_count <= 1) return;
  OutIndexT* row = output + static_cast<int64_t>(row_index) * output_stride;
  if (row[0] == static_cast<OutIndexT>(kNanSentinel)) return;
  float* row_values = values + static_cast<int64_t>(row_index) * value_stride;

  uint64_t items[kItemsPerThread];
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int position = static_cast<int>(threadIdx.x) * kItemsPerThread + item;
    if (position < valid_count) {
      const uint64_t value_key = ordered_key(row_values[position]);
      const uint64_t index = static_cast<uint64_t>(row[position]);
      items[item] = (value_key << 24) | (index << 1) | 1u;
    } else {
      items[item] = 0;
    }
  }
  BlockRadixSort(sort_storage).SortDescending(items);
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int position = static_cast<int>(threadIdx.x) * kItemsPerThread + item;
    if (position < valid_count) {
      const uint64_t packed = items[item];
      row[position] = static_cast<OutIndexT>((packed >> 1) & 0x7FFFFFu);
      row_values[position] =
          value_from_ordered_key(static_cast<uint32_t>(packed >> 24));
    }
  }
}

template <typename OutIndexT>
void launch_sort_output_typed(
    at::Tensor& output, const c10::optional<at::Tensor>& output_value,
    const int* end_ptr, int full_length, bool sort_by_value,
    musaStream_t stream) {
  const int rows = static_cast<int>(output.size(0));
  const int topk = static_cast<int>(output.size(1));
  const int64_t output_stride = output.stride(0);
  const int sort_threads = topk > 1024 ? 1024 : (topk > 512 ? 512 : 256);
  if (sort_by_value) {
    if (topk > 2048)
      radix_sort_packed_value_output_kernel<OutIndexT, 1024, 4>
          <<<rows, 1024, 0, stream>>>(
              output.data_ptr<OutIndexT>(), output_stride,
              output_value->data_ptr<float>(), output_value->stride(0),
              end_ptr, full_length, topk);
    else if (topk > 1024)
      radix_sort_packed_value_output_kernel<OutIndexT, 1024, 2>
          <<<rows, 1024, 0, stream>>>(
              output.data_ptr<OutIndexT>(), output_stride,
              output_value->data_ptr<float>(), output_value->stride(0),
              end_ptr, full_length, topk);
    else
      sort_value_output_bucket_kernel<OutIndexT>
          <<<rows, 256, 0, stream>>>(
              output.data_ptr<OutIndexT>(), output_stride,
              output_value->data_ptr<float>(), output_value->stride(0),
              end_ptr, full_length, topk);
  } else if (!output_value.has_value()) {
    sort_output_kernel<OutIndexT, int, false, false>
        <<<rows, sort_threads, 0, stream>>>(
            output.data_ptr<OutIndexT>(), output_stride, nullptr, 0, end_ptr,
            full_length, topk);
  } else if (output_value->scalar_type() == at::kFloat) {
    sort_output_kernel<OutIndexT, float, false, true>
        <<<rows, sort_threads, 0, stream>>>(
            output.data_ptr<OutIndexT>(), output_stride,
            output_value->data_ptr<float>(), output_value->stride(0), end_ptr,
            full_length, topk);
  } else {
    sort_output_kernel<OutIndexT, __mt_bfloat16, false, true>
        <<<rows, sort_threads, 0, stream>>>(
            output.data_ptr<OutIndexT>(), output_stride,
            reinterpret_cast<__mt_bfloat16*>(output_value->data_ptr()),
            output_value->stride(0), end_ptr, full_length, topk);
  }
}

void launch_sort_output(
    at::Tensor& output, const c10::optional<at::Tensor>& output_value,
    const c10::optional<at::Tensor>& end, int64_t full_length64,
    bool sort_by_value) {
  TORCH_CHECK(output.device().type() == c10::DeviceType::PrivateUse1 &&
                  (!output_value.has_value() ||
                   output_value->device() == output.device()) &&
                  (!end.has_value() || end->device() == output.device()),
              "output/value/end must be MUSA tensors on the same device");
  TORCH_CHECK(output.dim() == 2 && output.stride(1) == 1 &&
                  (output.scalar_type() == at::kInt ||
                   output.scalar_type() == at::kLong),
              "output must be contiguous-last-dim int32/int64 [batch, topk]");
  TORCH_CHECK(!output_value.has_value() ||
                  (output_value->dim() == 2 &&
                   output_value->sizes() == output.sizes() &&
                   output_value->stride(1) == 1 &&
                   (output_value->scalar_type() == at::kBFloat16 ||
                    output_value->scalar_type() == at::kFloat)),
              "output_value must be BF16/FP32 [batch, topk]");
  TORCH_CHECK(!sort_by_value ||
                  (output_value.has_value() &&
                   output_value->scalar_type() == at::kFloat),
              "value sorting requires FP32 output_value");
  TORCH_CHECK(!end.has_value() ||
                  (end->scalar_type() == at::kInt && end->dim() == 1 &&
                   end->size(0) == output.size(0) && end->is_contiguous()),
              "end must be contiguous int32 [batch]");
  TORCH_CHECK(full_length64 > 0 && full_length64 < (1 << 23),
              "full_length must be in (0, 2^23)");
  const int rows = static_cast<int>(output.size(0));
  if (rows == 0) return;
  const int full_length = static_cast<int>(full_length64);
  const int* end_ptr = end.has_value() ? end->data_ptr<int>() : nullptr;
  at::musa::MUSAGuard guard(output.device());
  const auto stream = at::musa::getCurrentMUSAStream(output.device().index());
  if (output.scalar_type() == at::kInt)
    launch_sort_output_typed<int32_t>(
        output, output_value, end_ptr, full_length, sort_by_value, stream);
  else
    launch_sort_output_typed<int64_t>(
        output, output_value, end_ptr, full_length, sort_by_value, stream);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

bool launch_topk_fp32(
    const at::Tensor& input, at::Tensor& output,
    const c10::optional<at::Tensor>& output_value = c10::nullopt,
    const c10::optional<at::Tensor>& end = c10::nullopt,
    int64_t idx_fill64 = 2147483647,
    double value_fill64 = -INFINITY,
    bool abort_when_nan_found = true, bool sort_by_value = false) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
              "input must be a MUSA tensor");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(output.device() == input.device() &&
                  (output.scalar_type() == at::kInt ||
                   output.scalar_type() == at::kLong),
              "output must be an int32/int64 MUSA tensor");
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2 && input.stride(1) == 1 &&
                  output.stride(1) == 1,
              "invalid FP32 native tensor layout");
  const int topk = static_cast<int>(output.size(1));
  TORCH_CHECK(topk > 0 && topk <= kMaxTopK && input.size(1) > 0 &&
                  input.size(1) < (1 << 23),
              "invalid FP32 native topk/width");
  TORCH_CHECK(!output_value.has_value() ||
                  (output_value->device() == input.device() &&
                   output_value->scalar_type() == at::kFloat &&
                   output_value->dim() == 2 &&
                   output_value->size(0) == input.size(0) &&
                   output_value->size(1) == topk &&
                   output_value->stride(1) == 1),
              "output_value must be FP32 [batch, topk]");
  TORCH_CHECK(!end.has_value() ||
                  (end->device() == input.device() && end->scalar_type() == at::kInt &&
                   end->dim() == 1 && end->size(0) == input.size(0) &&
                   end->is_contiguous()),
              "end must be contiguous int32 [batch]");
  const int rows = static_cast<int>(input.size(0));
  if (rows == 0) return false;
  const int64_t width = input.size(1);
  const int64_t input_stride = input.stride(0);
  const int64_t output_stride = output.stride(0);
  const int64_t value_stride = output_value.has_value()
      ? output_value->stride(0)
      : 0;
  float* values = output_value.has_value()
      ? output_value->data_ptr<float>()
      : nullptr;
  const int* end_ptr = end.has_value() ? end->data_ptr<int>() : nullptr;
  at::musa::MUSAGuard guard(input.device());
  const auto stream = at::musa::getCurrentMUSAStream(input.device().index());
  const float* input_ptr = input.data_ptr<float>();
  bool sorted_in_kernel = false;
  if (output.scalar_type() == at::kInt)
    sorted_in_kernel = launch_topk_fp32_typed(
        input_ptr, output.data_ptr<int32_t>(), rows, width, input_stride,
        output_stride, values, value_stride, end_ptr,
        static_cast<int32_t>(idx_fill64), static_cast<float>(value_fill64),
        topk, output_value.has_value(), sort_by_value, stream);
  else
    sorted_in_kernel = launch_topk_fp32_typed(
        input_ptr, output.data_ptr<int64_t>(), rows, width, input_stride,
        output_stride, values, value_stride, end_ptr,
        static_cast<int64_t>(idx_fill64), static_cast<float>(value_fill64),
        topk, output_value.has_value(), sort_by_value, stream);
  if (abort_when_nan_found) {
    if (output.scalar_type() == at::kInt)
      launch_nan_guard(output.data_ptr<int32_t>(), rows, output_stride, stream);
    else
      launch_nan_guard(output.data_ptr<int64_t>(), rows, output_stride, stream);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
  return sorted_in_kernel;
}

void launch_topk_bf16(
    const at::Tensor& input, at::Tensor& output,
    const c10::optional<at::Tensor>& output_value = c10::nullopt,
    const c10::optional<at::Tensor>& end = c10::nullopt,
    int64_t idx_fill64 = 2147483647,
    double value_fill64 = -INFINITY,
    bool abort_when_nan_found = true) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
              "input must be a MUSA tensor");
  TORCH_CHECK(output.device() == input.device(), "output device mismatch");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
  TORCH_CHECK(output.scalar_type() == at::kInt ||
                  output.scalar_type() == at::kLong,
              "output must be int32 or int64");
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2, "expected 2-D tensors");
  TORCH_CHECK(input.stride(1) == 1 && output.stride(1) == 1,
              "MUSA C++ topk requires contiguous last dimensions");
  const int topk = static_cast<int>(output.size(1));
  TORCH_CHECK(!output_value.has_value() ||
                  (output_value->device() == input.device() &&
                   output_value->scalar_type() == at::kBFloat16 &&
                   output_value->dim() == 2 &&
                   output_value->size(0) == input.size(0) &&
                   output_value->size(1) == topk &&
                   output_value->stride(1) == 1),
              "output_value must be BF16 [batch, topk] with contiguous last dim");
  TORCH_CHECK(!end.has_value() ||
                  (end->device() == input.device() && end->scalar_type() == at::kInt &&
                   end->dim() == 1 && end->size(0) == input.size(0) &&
                   end->is_contiguous()),
              "end must be contiguous int32 [batch]");
  TORCH_CHECK(output.size(0) == input.size(0) && topk > 0 &&
                  topk <= kMaxTopK,
              "output must have shape [batch, topk], topk in [1, 4096]");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) < (1 << 23),
              "width must be in (0, 2^23)");

  const int rows = static_cast<int>(input.size(0));
  if (rows == 0) return;
  const int64_t width = input.size(1);
  const int64_t input_stride = input.stride(0);
  const int64_t output_stride = output.stride(0);
  const int64_t value_stride = output_value.has_value()
      ? output_value->stride(0)
      : 0;
  auto* value_ptr = output_value.has_value()
      ? reinterpret_cast<__mt_bfloat16*>(output_value->data_ptr())
      : nullptr;
  const int* end_ptr = end.has_value() ? end->data_ptr<int>() : nullptr;
  Bf16Bits fill_bits;
  fill_bits.value = static_cast<__mt_bfloat16>(static_cast<float>(value_fill64));
  const auto* input_ptr =
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr());
  at::musa::MUSAGuard guard(input.device());
  const auto stream = at::musa::getCurrentMUSAStream(input.device().index());
  if (output.scalar_type() == at::kInt)
    launch_topk_bf16_typed(
        input_ptr, output.data_ptr<int32_t>(), rows, width, input_stride,
        output_stride, value_ptr, value_stride, end_ptr,
        static_cast<int32_t>(idx_fill64), fill_bits.value, topk,
        output_value.has_value(), abort_when_nan_found, stream);
  else
    launch_topk_bf16_typed(
        input_ptr, output.data_ptr<int64_t>(), rows, width, input_stride,
        output_stride, value_ptr, value_stride, end_ptr,
        static_cast<int64_t>(idx_fill64), fill_bits.value, topk,
        output_value.has_value(), abort_when_nan_found, stream);
  if (abort_when_nan_found) {
    if (output.scalar_type() == at::kInt)
      launch_nan_guard(output.data_ptr<int32_t>(), rows, output_stride, stream);
    else
      launch_nan_guard(output.data_ptr<int64_t>(), rows, output_stride, stream);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

template <typename OutIndexT>
void launch_topk_bf16_multi_typed(
    const __mt_bfloat16* input_ptr, OutIndexT* output_ptr,
    const MultiScratch& buffers, int rows, int parts, int worker_threads,
    int64_t width, int64_t output_stride, __mt_bfloat16* values,
    int64_t value_stride, int topk, bool return_value, musaStream_t stream) {
  const int tasks = rows * parts;
  partial_high_histogram_kernel<<<tasks, worker_threads, 0, stream>>>(
      input_ptr, buffers.high_hist, buffers.partial_nan, width, parts);
  reduce_high_histogram_kernel<OutIndexT><<<rows, 256, 0, stream>>>(
      buffers.high_hist, buffers.partial_nan, output_ptr,
      buffers.threshold_hi, buffers.remain_hi, buffers.output_count,
      buffers.has_nan, output_stride, parts, topk);
  partial_low_histogram_kernel<OutIndexT>
      <<<tasks, worker_threads, 0, stream>>>(
          input_ptr, output_ptr, buffers.low_hist, buffers.threshold_hi,
          buffers.output_count, buffers.has_nan, buffers.candidate_counts,
          buffers.candidate_indices, width, output_stride, values, value_stride,
          parts, topk, return_value);
  reduce_low_histogram_kernel<<<rows, 256, 0, stream>>>(
      buffers.low_hist, buffers.threshold_lo, buffers.remain_lo,
      buffers.remain_hi, buffers.has_nan, parts);
  finish_topk_kernel<OutIndexT><<<tasks, worker_threads, 0, stream>>>(
      input_ptr, output_ptr, buffers.threshold_hi, buffers.threshold_lo,
      buffers.remain_lo, buffers.output_count, buffers.has_nan,
      buffers.candidate_counts, buffers.candidate_indices, width,
      output_stride, values, value_stride, parts, topk, return_value);
}

void launch_topk_bf16_multi(const at::Tensor& input, at::Tensor& output,
                            at::Tensor& scratch, int64_t parts64,
                            int64_t worker_threads64,
                            const c10::optional<at::Tensor>& output_value =
                                c10::nullopt,
                            bool abort_when_nan_found = true) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
              "input must be a MUSA tensor");
  TORCH_CHECK(output.device() == input.device() && scratch.device() == input.device(),
              "output/scratch device mismatch");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
  TORCH_CHECK((output.scalar_type() == at::kInt ||
               output.scalar_type() == at::kLong) &&
                  scratch.scalar_type() == at::kInt,
              "output must be int32 or int64; scratch must be int32");
  TORCH_CHECK(input.is_contiguous() && output.stride(1) == 1 &&
                  scratch.is_contiguous(),
              "MUSA multi-CTA topk requires contiguous input/scratch and "
              "output last dim");
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2 && scratch.dim() == 1,
              "invalid tensor ranks");
  const int topk = static_cast<int>(output.size(1));
  TORCH_CHECK(!output_value.has_value() ||
                  (output_value->device() == input.device() &&
                   output_value->scalar_type() == at::kBFloat16 &&
                   output_value->dim() == 2 &&
                   output_value->size(0) == input.size(0) &&
                   output_value->size(1) == topk &&
                   output_value->stride(1) == 1),
              "output_value must be BF16 [batch, topk] with contiguous last dim");
  TORCH_CHECK(output.size(0) == input.size(0) && topk > 0 &&
                  topk <= kMaxTopK,
              "output must have shape [batch, topk], topk in [1, 4096]");
  TORCH_CHECK(input.size(1) > topk && input.size(1) < (1 << 23),
              "width must be in (topk, 2^23)");
  TORCH_CHECK(parts64 > 1 && parts64 <= 64, "parts must be in [2, 64]");
  TORCH_CHECK(worker_threads64 == 128 || worker_threads64 == 256 ||
                  worker_threads64 == 512,
              "worker_threads must be 128, 256, or 512");

  const int rows = static_cast<int>(input.size(0));
  const int parts = static_cast<int>(parts64);
  const int worker_threads = static_cast<int>(worker_threads64);
  if (rows == 0) return;
  const int64_t required =
      2LL * rows * parts * kRadix + 2LL * rows * parts + 6LL * rows +
      1LL * rows * input.size(1);
  TORCH_CHECK(scratch.numel() >= required, "scratch tensor is too small");
  const int64_t width = input.size(1);
  const int64_t output_stride = output.stride(0);
  const int64_t value_stride = output_value.has_value()
      ? output_value->stride(0)
      : 0;
  auto* value_ptr = output_value.has_value()
      ? reinterpret_cast<__mt_bfloat16*>(output_value->data_ptr())
      : nullptr;
  const auto* input_ptr =
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr());
  auto* scratch_ptr = static_cast<int*>(scratch.data_ptr());
  const MultiScratch buffers = split_scratch(scratch_ptr, rows, parts, width);
  at::musa::MUSAGuard guard(input.device());
  const auto stream = at::musa::getCurrentMUSAStream(input.device().index());
  if (output.scalar_type() == at::kInt)
    launch_topk_bf16_multi_typed(
        input_ptr, output.data_ptr<int32_t>(), buffers, rows, parts,
        worker_threads, width, output_stride, value_ptr, value_stride, topk,
        output_value.has_value(), stream);
  else
    launch_topk_bf16_multi_typed(
        input_ptr, output.data_ptr<int64_t>(), buffers, rows, parts,
        worker_threads, width, output_stride, value_ptr, value_stride, topk,
        output_value.has_value(), stream);
  if (abort_when_nan_found) {
    if (output.scalar_type() == at::kInt)
      launch_nan_guard(output.data_ptr<int32_t>(), rows, output_stride, stream);
    else
      launch_nan_guard(output.data_ptr<int64_t>(), rows, output_stride, stream);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

constexpr int kPersistOutCap = 2048;
constexpr int kPersistMaxSpins = 1 << 26;

struct PersistBf16 {
  using Elem = __mt_bfloat16;
  static constexpr int kVec = 8;  // 16 B
  struct __align__(16) Vec {
    Elem v[8];
  };
  static __device__ __forceinline__ unsigned key(Elem x) {
    return static_cast<unsigned>(ordered_key(x));
  }
  static __device__ __forceinline__ bool is_nan(Elem x) {
    return is_nan_key(raw_bits(x));
  }
};

__device__ __forceinline__ void persist_grid_barrier(volatile int* bar,
                                                     int grid) {
  volatile int* cnt = bar;
  volatile int* gen = bar + 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    const int observed = *gen;
    __threadfence();  // release this CTA's histogram contributions
    if (atomicAdd(const_cast<int*>(cnt), 1) == grid - 1) {
      *cnt = 0;
      __threadfence();
      *gen = observed + 1;
    } else {
      int spins = 0;
      while (*gen == observed && ++spins < kPersistMaxSpins) {
      }
    }
  }
  __syncthreads();
}

template <int kThreads, int kShift, int kBits, int kFiltShift>
__device__ __forceinline__ void persist_level_hist(
    const __mt_bfloat16* __restrict__ row, int start, int end,
    int* __restrict__ ghist, int* __restrict__ g_nan, unsigned prefix,
    int* s_lvl) {
  constexpr int kV = PersistBf16::kVec;
  constexpr unsigned kMask = (1u << kBits) - 1u;
  const int tid = static_cast<int>(threadIdx.x);
  for (int b = tid; b < (1 << kBits); b += kThreads) s_lvl[b] = 0;
  __syncthreads();
  int nan_count = 0;
  const int nfull = (end - start) / kV;
  const typename PersistBf16::Vec* vp =
      reinterpret_cast<const typename PersistBf16::Vec*>(row + start);
  for (int c = tid; c < nfull; c += kThreads) {
    const typename PersistBf16::Vec vec = vp[c];
#pragma unroll
    for (int j = 0; j < kV; ++j) {
      const PersistBf16::Elem value = vec.v[j];
      const unsigned k = PersistBf16::key(value);
      if (kFiltShift == 0) nan_count += PersistBf16::is_nan(value);
      if (kFiltShift != 0 && (k >> kFiltShift) != prefix) continue;
      atomicAdd(&s_lvl[(k >> kShift) & kMask], 1);
    }
  }
  for (int i = start + nfull * kV + tid; i < end; i += kThreads) {
    const PersistBf16::Elem value = row[i];
    const unsigned k = PersistBf16::key(value);
    if (kFiltShift == 0) nan_count += PersistBf16::is_nan(value);
    if (kFiltShift != 0 && (k >> kFiltShift) != prefix) continue;
    atomicAdd(&s_lvl[(k >> kShift) & kMask], 1);
  }
  __syncthreads();
  if (kFiltShift == 0 && nan_count) atomicAdd(g_nan, nan_count);
  for (int b = tid; b < (1 << kBits); b += kThreads) {
    const int count = s_lvl[b];
    if (count) atomicAdd(ghist + b, count);
  }
}

template <int kThreads>
__device__ __forceinline__ void persist_scan(volatile const int* hist, int wanted,
                                             int* s_piv, int* s_rem,
                                             int* s_warp) {
  constexpr int kWarps = kThreads >> 5;
  const int t = static_cast<int>(threadIdx.x);
  const int cnt = (t < kRadix) ? hist[t] : 0;
  int x = cnt;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xFFFFFFFFu, x, off);
    if ((t & 31) >= off) x += y;
  }
  if ((t & 31) == 31) s_warp[t >> 5] = x;
  __syncthreads();
  if (t < kWarps) {
    int y = s_warp[t];
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int z = __shfl_up_sync(0xFFFFFFFFu, y, off);
      if (t >= off) y += z;
    }
    s_warp[t] = y;
  }
  __syncthreads();
  const int prefix = ((t >> 5) ? s_warp[(t >> 5) - 1] : 0) + x;
  const int total = s_warp[kWarps - 1];
  const int target = total - wanted;
  if (target > 0) {
    if (t < kRadix && target > prefix - cnt && target <= prefix) {
      *s_piv = t;
      *s_rem = prefix - target;
    }
  } else if (t == 0) {
    *s_piv = 0;
    *s_rem = wanted - (total - hist[0]);
  }
  __syncthreads();
}

template <int kThreads>
__device__ __forceinline__ void persist_emit(
    const __mt_bfloat16* __restrict__ row, int start, int end, int* s_buf,
    int* s_tie, int* s_nh, int* s_nt, unsigned thr) {
  constexpr int kV = PersistBf16::kVec;
  const int tid = static_cast<int>(threadIdx.x);
  if (tid == 0) {
    *s_nh = 0;
    *s_nt = 0;
  }
  __syncthreads();
  const int nfull = (end - start) / kV;
  const typename PersistBf16::Vec* vp =
      reinterpret_cast<const typename PersistBf16::Vec*>(row + start);
  for (int c = tid; c < nfull; c += kThreads) {
    const typename PersistBf16::Vec vec = vp[c];
    const int base = start + c * kV;
#pragma unroll
    for (int j = 0; j < kV; ++j) {
      const unsigned k = PersistBf16::key(vec.v[j]);
      if (k > thr) {
        const int slot = atomicAdd(s_nh, 1);
        if (slot < kPersistOutCap) s_buf[slot] = base + j;
      } else if (k == thr) {
        const int slot = atomicAdd(s_nt, 1);
        if (slot < kPersistOutCap) s_tie[slot] = base + j;
      }
    }
  }
  for (int i = start + nfull * kV + tid; i < end; i += kThreads) {
    const unsigned k = PersistBf16::key(row[i]);
    if (k > thr) {
      const int slot = atomicAdd(s_nh, 1);
      if (slot < kPersistOutCap) s_buf[slot] = i;
    } else if (k == thr) {
      const int slot = atomicAdd(s_nt, 1);
      if (slot < kPersistOutCap) s_tie[slot] = i;
    }
  }
  __syncthreads();
}

template <int kThreads, typename OutIndexT>
__global__ __launch_bounds__(kThreads, 1) void topk_bf16_persist_kernel(
    const __mt_bfloat16* __restrict__ input, OutIndexT* __restrict__ out,
    __mt_bfloat16* __restrict__ out_value, int* __restrict__ scratch,
    int* __restrict__ bar, int rows, int width, int topk, int64_t out_stride,
    int64_t value_stride, int abort_when_nan) {
  __shared__ int s_lvl[kRadix];
  __shared__ int s_buf[kPersistOutCap];
  __shared__ int s_tie[kPersistOutCap];
  __shared__ int s_warp[kThreads >> 5];
  __shared__ int s_piv, s_rem, s_nh, s_nt, s_base_h, s_base_t;

  const int tid = static_cast<int>(threadIdx.x);
  const int grid = static_cast<int>(gridDim.x);
  const int bx = static_cast<int>(blockIdx.x);
  const int row_id = bx % rows;
  const int parts = grid / rows;
  const int part = bx / rows;
  const int chunk = (width + parts - 1) / parts;
  const int start = part * chunk;
  const int end = (start + chunk < width) ? start + chunk : width;
  const __mt_bfloat16* row = input + static_cast<int64_t>(row_id) * width;
  OutIndexT* out_row = out + static_cast<int64_t>(row_id) * out_stride;
  __mt_bfloat16* val_row =
      out_value ? out_value + static_cast<int64_t>(row_id) * value_stride
                : nullptr;
  int* hist0 = scratch + static_cast<int64_t>(row_id) * kRadix;
  int* hist1 = scratch + static_cast<int64_t>(rows + row_id) * kRadix;
  int* pos = scratch + static_cast<int64_t>(rows) * 2 * kRadix + 2 * row_id;
  int* row_nan = scratch + static_cast<int64_t>(rows) * 2 * kRadix + 2 * rows +
                 row_id;
  volatile int* barv = bar;
  const bool write_value = out_value != nullptr;

  if (part == 0 && tid == 0) {
    pos[0] = 0;
    pos[1] = 0;
    row_nan[0] = 0;
  }

  persist_level_hist<kThreads, 8, 8, 0>(row, start, end, hist0, row_nan, 0u,
                                        s_lvl);
  persist_grid_barrier(barv, grid);
  persist_scan<kThreads>(hist0, topk, &s_piv, &s_rem, s_warp);
  const unsigned hi = static_cast<unsigned>(s_piv);
  int remain = s_rem;

  persist_level_hist<kThreads, 0, 8, 8>(row, start, end, hist1, nullptr, hi,
                                        s_lvl);
  persist_grid_barrier(barv, grid);
  persist_scan<kThreads>(hist1, remain, &s_piv, &s_rem, s_warp);
  const unsigned thr = (hi << 8) | static_cast<unsigned>(s_piv);
  remain = s_rem;
  const int count_above = topk - remain;

  const int nan_total = row_nan[0];

  persist_emit<kThreads>(row, start, end, s_buf, s_tie, &s_nh, &s_nt, thr);
  const int nh = (s_nh < kPersistOutCap) ? s_nh : kPersistOutCap;
  const int nt = (s_nt < kPersistOutCap) ? s_nt : kPersistOutCap;
  if (tid == 0) {
    s_base_h = nh ? atomicAdd(pos, nh) : 0;
    s_base_t = nt ? atomicAdd(pos + 1, nt) : 0;
  }
  __syncthreads();
  const int base_h = s_base_h;
  const int tbase = s_base_t;
  for (int i = tid; i < nh; i += kThreads) {
    const int p = base_h + i;
    if (p < topk) {
      const int index = s_buf[i];
      out_row[p] = static_cast<OutIndexT>(index);
      if (write_value) val_row[p] = row[index];
    }
  }
  for (int i = tid; i < nt; i += kThreads) {
    const int ticket = tbase + i;
    if (ticket < remain) {
      const int index = s_tie[i];
      const int p = count_above + ticket;
      out_row[p] = static_cast<OutIndexT>(index);
      if (write_value) val_row[p] = row[index];
    }
  }
  __syncthreads();

  if (nan_total > 0 && tid == 0) {
    out_row[0] = static_cast<OutIndexT>(kNanSentinel);
  }

  for (int b = part * kThreads + tid; b < kRadix; b += parts * kThreads) {
    hist0[b] = 0;
    hist1[b] = 0;
  }
}

pybind11::dict fp32_kernel_attrs() {
  musaFuncAttributes a{};
  (void)musaFuncGetAttributes(
      &a, reinterpret_cast<const void*>(
              &topk_fp32_compact12_kernel<1024, 4096, int64_t>));
  pybind11::dict d;
  d["numRegs"] = a.numRegs;
  d["localBytes"] = static_cast<int64_t>(a.localSizeBytes);
  d["sharedBytes"] = static_cast<int64_t>(a.sharedSizeBytes);
  return d;
}

int64_t persist_scratch_ints(int64_t rows) {
  return rows * (2 * kRadix) + 2 * rows + rows + 8;
}

void launch_topk_bf16_persist(const at::Tensor& input, at::Tensor& output,
                              at::Tensor& scratch,
                              const c10::optional<at::Tensor>& output_value,
                              int64_t parts64, int64_t threads64,
                              bool abort_when_nan_found) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
              "input must be a MUSA tensor");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "persistent path requires bfloat16 input");
  TORCH_CHECK(output.scalar_type() == at::kInt ||
                  output.scalar_type() == at::kLong,
              "persistent path requires int32 or int64 output");
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2 && scratch.dim() == 1,
              "invalid tensor ranks");
  TORCH_CHECK(input.is_contiguous() && scratch.is_contiguous() &&
                  output.stride(1) == 1,
              "persistent path requires contiguous input/scratch and a "
              "contiguous output last dim");
  const int rows = static_cast<int>(input.size(0));
  const int width = static_cast<int>(input.size(1));
  const int topk = static_cast<int>(output.size(1));
  const int parts = static_cast<int>(parts64);
  const int threads = static_cast<int>(threads64);
  TORCH_CHECK(rows > 0 && parts > 0 && rows * parts <= 56,
              "persistent grid must stay resident (parts * rows <= 56)");
  TORCH_CHECK(threads == 256 || threads == 1024,
              "worker threads must be 256 or 1024");
  TORCH_CHECK(topk > 0 && topk <= kPersistOutCap,
              "persistent path supports topk <= 2048");
  TORCH_CHECK(width > topk && width % 8 == 0,
              "width must be a multiple of 8 and greater than topk");
  const int64_t chunk = (width + parts - 1) / parts;
  TORCH_CHECK(chunk % 8 == 0, "row slice must stay 16 B aligned");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(input.data_ptr()) % 16) == 0,
              "input must be 16 B aligned");
  TORCH_CHECK(output.size(0) == input.size(0) && output.stride(0) >= topk,
              "bad output shape");
  TORCH_CHECK(!output_value.has_value() ||
                  (output_value->device() == input.device() &&
                   output_value->scalar_type() == at::kBFloat16 &&
                   output_value->dim() == 2 &&
                   output_value->size(0) == input.size(0) &&
                   output_value->size(1) == topk &&
                   output_value->stride(1) == 1),
              "output_value must be BF16 [batch, topk] with contiguous last dim");
  TORCH_CHECK(scratch.numel() >= persist_scratch_ints(rows),
              "persistent scratch tensor is too small");
  TORCH_CHECK(!abort_when_nan_found,
              "the persistent path does not carry a device abort");
  const int64_t grid64 = static_cast<int64_t>(rows) * parts;
  TORCH_CHECK(grid64 > 0 && grid64 <= 56, "grid must be resident");
  const int64_t out_stride = output.stride(0);
  const int64_t value_stride =
      output_value.has_value() ? output_value->stride(0) : 0;
  auto* value_ptr = output_value.has_value()
                        ? reinterpret_cast<__mt_bfloat16*>(
                              output_value->data_ptr())
                        : nullptr;
  const auto* input_ptr =
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr());
  auto* scratch_ptr = static_cast<int*>(scratch.data_ptr());
  const int64_t hist_words = static_cast<int64_t>(rows) * (2 * kRadix);
  int* bar = scratch_ptr + hist_words + 2 * rows + rows;
  at::musa::MUSAGuard guard(input.device());
  const auto stream = at::musa::getCurrentMUSAStream(input.device().index());
  const int abort_flag = abort_when_nan_found ? 1 : 0;
  const bool out32 = output.scalar_type() == at::kInt;
  const int grid = static_cast<int>(grid64);
  if (threads == 1024) {
    if (out32)
      topk_bf16_persist_kernel<1024, int32_t><<<grid, 1024, 0, stream>>>(
          input_ptr, output.data_ptr<int32_t>(), value_ptr, scratch_ptr, bar,
          rows, width, topk, out_stride, value_stride, abort_flag);
    else
      topk_bf16_persist_kernel<1024, int64_t><<<grid, 1024, 0, stream>>>(
          input_ptr, output.data_ptr<int64_t>(), value_ptr, scratch_ptr, bar,
          rows, width, topk, out_stride, value_stride, abort_flag);
  } else {
    if (out32)
      topk_bf16_persist_kernel<256, int32_t><<<grid, 256, 0, stream>>>(
          input_ptr, output.data_ptr<int32_t>(), value_ptr, scratch_ptr, bar,
          rows, width, topk, out_stride, value_stride, abort_flag);
    else
      topk_bf16_persist_kernel<256, int64_t><<<grid, 256, 0, stream>>>(
          input_ptr, output.data_ptr<int64_t>(), value_ptr, scratch_ptr, bar,
          rows, width, topk, out_stride, value_stride, abort_flag);
  }
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

#ifndef DEEP_SELECT_DISABLE_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("topk_bf16", &launch_topk_bf16, "DeepSelect BF16 MUSA topk");
  module.def("topk_fp32", &launch_topk_fp32, "DeepSelect FP32 MUSA topk");
  module.def("topk_fp32_multi", &launch_topk_fp32_multi,
             "DeepSelect FP32 MUSA multi-CTA topk");
  module.def("topk_fp32_two_stage", &launch_topk_fp32_two_stage,
             "Experimental exact two-stage FP32 MUSA topk");
  module.def("two_stage_scratch_ints", &two_stage_scratch_ints,
             "Experimental two-stage scratch size in int32 words");
  module.def("output_idx_offset", &launch_output_idx_offset,
             "DeepSelect MUSA output index offset epilogue");
  module.def("sort_output", &launch_sort_output,
             "DeepSelect MUSA native sort epilogue");
  module.def("sort_output_fused", &launch_sort_output_fused,
             "DeepSelect MUSA fused sort+offset epilogue");
  module.def("topk_bf16_multi", &launch_topk_bf16_multi,
             "DeepSelect BF16 MUSA multi-CTA topk");
  module.def("topk_bf16_persist", &launch_topk_bf16_persist,
             "DeepSelect BF16 MUSA persistent-grid topk");
  module.def("fp32_kernel_attrs", &fp32_kernel_attrs,
             "resources of the fp32 compact12 selector");
  module.def("persist_scratch_ints", &persist_scratch_ints,
             "DeepSelect BF16 persistent-grid scratch size in int32 words");
}
#endif
