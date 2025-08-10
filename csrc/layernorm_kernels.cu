#include "type_convert.cuh"
#include "dispatch_utils.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include "quantization/vectorization_utils.cuh"

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace vllm {

// ---------- helpers ----------
template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
#ifdef __HIP_PLATFORM_AMD__
  const unsigned long long m = 0xffffffffffffffffull;  // HIP needs 64-bit mask
#else
  const unsigned           m = 0xffffffffu;            // CUDA 32-bit mask
#endif
  // Always reduce over 32 lanes to match downstream logic.
  constexpr int kWidth = 32;
  v += __shfl_down_sync(m, v, 16, kWidth);
  v += __shfl_down_sync(m, v, 8,  kWidth);
  v += __shfl_down_sync(m, v, 4,  kWidth);
  v += __shfl_down_sync(m, v, 2,  kWidth);
  v += __shfl_down_sync(m, v, 1,  kWidth);
  return v;
}

template <typename T>
__device__ __forceinline__ bool same_phase(const T* a, const T* b, int widthB) {
  auto ai = reinterpret_cast<uintptr_t>(a);
  auto bi = reinterpret_cast<uintptr_t>(b);
  return ((ai ^ bi) & (widthB - 1)) == 0;
}

// Safe 16B copy to shared: prefix to align, vector main, scalar tail.
template <typename T>
__device__ __forceinline__ void copy_row_to_shared_aligned(
    const T* __restrict__ src, T* __restrict__ dst, int n_elems, int tid) {
  const uintptr_t sa = reinterpret_cast<uintptr_t>(src);
  const uintptr_t da = reinterpret_cast<uintptr_t>(dst);
  const bool same16 = (((sa ^ da) & (16 - 1)) == 0);

  if (!same16) {
    for (int i = tid; i < n_elems; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
    return;
  }
  const int ebytes = sizeof(T);
  const int perVec = 16 / ebytes;

  int prefix = 0;
  const int mis = sa & (16 - 1);
  if (mis) prefix = (16 - mis) / ebytes;
  if (prefix > n_elems) prefix = n_elems;

  // scalar prefix
  for (int i = tid; i < prefix; i += blockDim.x) dst[i] = src[i];

  // vector main
  const int remain     = n_elems - prefix;
  const int main_elems = (remain / perVec) * perVec;
  if (main_elems > 0) {
    const uint4* __restrict__ vsrc =
        reinterpret_cast<const uint4*>(src + prefix);

#if defined(__HIP_PLATFORM_AMD__)
    // ROCm: vector load from global, scalar 32-bit stores to shared
    uint32_t* __restrict__ s32 =
        reinterpret_cast<uint32_t*>(dst + prefix);
    const int nvec = main_elems / perVec;       // 16B packets
    constexpr int WORDS_PER_PKT = 16 / sizeof(uint32_t); // = 4
    for (int v = tid; v < nvec; v += blockDim.x) {
      uint4 p = vsrc[v];
      const int base = v * WORDS_PER_PKT;
      s32[base + 0] = p.x;
      s32[base + 1] = p.y;
      s32[base + 2] = p.z;
      s32[base + 3] = p.w;
    }
#else
    // CUDA: vector load + vector store (fastest)
    uint4* __restrict__ vdst =
        reinterpret_cast<uint4*>(dst + prefix);
    const int nvec = main_elems / perVec;
    for (int v = tid; v < nvec; v += blockDim.x) {
      uint4 p = vsrc[v];
      vdst[v] = p;
    }
#endif
  }

  // scalar tail
  const int tail = prefix + main_elems;
  for (int i = tid + tail; i < n_elems; i += blockDim.x) dst[i] = src[i];
  __syncthreads();
}

// ---------------- vec/scalar ops (generic, used for all dtypes) ----------------
template <int V, typename T>
struct VecMulNormWeight {
  const vec_n_t<T, V>* __restrict__ wv;  // vector view of weight (aligned with in/out)
  float inv_rms;
  int   stride_vec;
  mutable int64_t vec_idx;

  __device__ __forceinline__ void operator()(vec_n_t<T, V>& dst,
                                             const vec_n_t<T, V>& src) const {
    const vec_n_t<T, V> w = wv[vec_idx];
#pragma unroll
    for (int j = 0; j < V; ++j) {
      T xn = static_cast<T>(static_cast<float>(src.val[j]) * inv_rms);
      dst.val[j] = xn * w.val[j];
    }
    vec_idx += stride_vec;
  }
};

template <typename T>
struct ScalarMulNormWeight {
  const T* __restrict__ w_base;   // already offset by +prefix
  T*       __restrict__ out_base; // out_row + prefix
  float inv_rms;
  __device__ __forceinline__ void operator()(T& dst, const T src) const {
    const int i = static_cast<int>(&dst - out_base);
    T xn = static_cast<T>(static_cast<float>(src) * inv_rms);
    dst = xn * w_base[i];
  }
};

// ---------- kernel ----------
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,          // [..., hidden_size]
    const scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    const scalar_t* __restrict__ weight, // [hidden_size]
    const float epsilon, const int /*num_tokens*/, const int hidden_size,
    int smem_elems) {

  const scalar_t* __restrict__ in_row  = input + blockIdx.x * input_stride;
  scalar_t*       __restrict__ out_row = out   + blockIdx.x * hidden_size;

  // Optional cached-row (half) when host provisioned shmem
  extern __shared__ unsigned char smem_raw[];
  scalar_t* s_in = reinterpret_cast<scalar_t*>(smem_raw);

#ifdef __HIP_PLATFORM_AMD__
  constexpr bool kAllowCache = false;
#else
  constexpr bool kAllowCache = true;
#endif
  const bool use_cached =
      kAllowCache && (sizeof(scalar_t) == 2) && (smem_elems > 0);

#if !defined(__HIP_PLATFORM_AMD__)
  if (use_cached) {
    copy_row_to_shared_aligned(in_row, s_in, hidden_size, threadIdx.x);
  }
#endif

  // -------- Pass 1: sum of squares --------
  using acc_t = float;
  acc_t sumsq = acc_t(0);
  if (use_cached) {
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      acc_t x = static_cast<acc_t>(s_in[i]);
      sumsq += x * x;
    }
  } else {
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      acc_t x = static_cast<acc_t>(in_row[i]);
      sumsq += x * x;
    }
  }

  // warp + block reduction in acc_t
  acc_t wsum = warp_sum<acc_t>(sumsq);
  __shared__ acc_t warp_sums_sh[32];
  if ((threadIdx.x & 31) == 0) warp_sums_sh[threadIdx.x >> 5] = wsum;
  __syncthreads();

  acc_t total = acc_t(0);
  if (threadIdx.x < 32) {
    acc_t v = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_sums_sh[threadIdx.x] : acc_t(0);
    total = warp_sum<acc_t>(v);
    if (threadIdx.x == 0) warp_sums_sh[0] = total;
  }
  __syncthreads();

  // compute inv_rms in float to match baseline epsilon semantics
  const float inv_rms =
      rsqrtf(static_cast<float>(warp_sums_sh[0] / static_cast<acc_t>(hidden_size)) + epsilon);

  // -------- Fast path: HS == blockDim.x (e.g., 1024) --------
  if (hidden_size == blockDim.x) {
    int i = threadIdx.x;
    float x = static_cast<float>(use_cached ? s_in[i] : in_row[i]);
    scalar_t xn = static_cast<scalar_t>(x * inv_rms);
    out_row[i] = xn * weight[i];
    return;
  }

  // -------- Pass 2: Vectorize when phases align --------
  constexpr int V = (sizeof(scalar_t) == 2) ? 8 : 4;  // 16B packets
  constexpr int WIDTH = V * sizeof(scalar_t);

  const bool can_vec =
      (hidden_size % V == 0) &&
      same_phase(in_row, out_row, WIDTH) &&
      same_phase(in_row, weight,  WIDTH);

  if (can_vec) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(in_row);
    const int mis = addr & (WIDTH - 1);
    const int prefix = mis ? (WIDTH - mis) / (int)sizeof(scalar_t) : 0;

    // scalar prefix
    for (int i = threadIdx.x; i < prefix; i += blockDim.x) {
      float x = static_cast<float>(use_cached ? s_in[i] : in_row[i]);
      out_row[i] = static_cast<scalar_t>(x * inv_rms) * weight[i];
    }

    // vector main
    const int remain   = hidden_size - prefix;
    const int main_len = (remain / V) * V;
    if (main_len > 0) {
      using VecT = vec_n_t<scalar_t, V>;
      const VecT* __restrict__ wv =
          reinterpret_cast<const VecT*>(weight + prefix);

      VecMulNormWeight<V, scalar_t> vec_op{
          /*wv=*/ wv,
          /*inv_rms=*/ inv_rms,
          /*stride_vec=*/ (int)blockDim.x,
          /*vec_idx=*/ (int64_t)threadIdx.x
      };
      ScalarMulNormWeight<scalar_t> sca_op{
          /*w_base=*/ weight + prefix,
          /*out_base=*/ out_row + prefix,
          /*inv_rms=*/ inv_rms
      };

      const scalar_t* vin = use_cached ? (s_in + prefix) : (in_row + prefix);
      vectorize_with_alignment<V>(
          vin, out_row + prefix, main_len,
          threadIdx.x, blockDim.x, vec_op, sca_op);
    }

    // scalar tail
    const int tail = prefix + ((remain / V) * V);
    for (int i = threadIdx.x + tail; i < hidden_size; i += blockDim.x) {
      float x = static_cast<float>(use_cached ? s_in[i] : in_row[i]);
      out_row[i] = static_cast<scalar_t>(x * inv_rms) * weight[i];
    }
    return;
  }

  // -------- Fallback scalar --------
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(use_cached ? s_in[i] : in_row[i]);
    scalar_t xn = static_cast<scalar_t>(x * inv_rms);
    out_row[i] = xn * weight[i];
  }
}


/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[strided_id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * input_stride + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

}  // namespace vllm

static inline int ln_block_threads_unified(int H) {
  int threads = (H >= 1024) ? 256
             : (H >= 512)  ? 512
                            : std::min(1024, ((H + 31) / 32) * 32);
  return std::min(1024, std::max(128, ((threads + 31) / 32) * 32));
}

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(weight.is_contiguous());

  const int hidden_size   = input.size(-1);
  const int num_tokens    = input.numel() / hidden_size;
  const int64_t in_stride = input.stride(-2);

  dim3 grid(num_tokens);
  dim3 block(ln_block_threads_unified(hidden_size));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Optional cached-row for FP16 (recommended). Kernel still works if this is 0.
  size_t shmem_bytes = 0;
  int smem_elems = 0;
  if (input.scalar_type() == at::kHalf && hidden_size <= 4096) {
    shmem_bytes = static_cast<size_t>(hidden_size) * sizeof(at::Half);
    smem_elems  = hidden_size;  // flag to kernel that shmem was provisioned
  }

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    vllm::rms_norm_kernel<scalar_t>
        <<<grid, block, shmem_bytes, stream>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            in_stride,
            weight.data_ptr<scalar_t>(),
            static_cast<float>(epsilon),
            num_tokens,
            hidden_size,
            smem_elems);
  });
}


#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                    \
  VLLM_DISPATCH_FLOATING_TYPES(                                             \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {               \
        vllm::fused_add_rms_norm_kernel<scalar_t, width>                    \
            <<<grid, block, 0, stream>>>(                                   \
                input.data_ptr<scalar_t>(), input_stride,                   \
                residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), \
                epsilon, num_tokens, hidden_size);                          \
      });

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  constexpr int vector_width = 8;
  constexpr int req_alignment_bytes =
      vector_width * 2;  // vector_width * sizeof(bfloat16 or float16) (float32
                         // falls back to non-vectorized version anyway)
  bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                          res_ptr % req_alignment_bytes == 0 &&
                          wt_ptr % req_alignment_bytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input_stride % vector_width == 0;
  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}
