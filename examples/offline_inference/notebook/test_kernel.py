# triton_conv1d_v2.py

import torch
import triton
import triton.language as tl

# @triton.autotune(
#     configs=[
#         triton.Config({'BS': 32}, num_stages=3, num_warps=4),
#         triton.Config({'BS': 64}, num_stages=3, num_warps=4),
#         triton.Config({'BS': 128}, num_stages=3, num_warps=4),
#     ],
#     key=['seq_len', 'kernel_size'],
# )
# @triton.jit
# def triton_conv1d_kernel_v2(
#     x_ptr, w_ptr, out_ptr,
#     batch, seq_len, in_channels, out_channels, kernel_size, groups,
#     stride, padding, dilation,
#     out_seq_len,
#     stride_xb, stride_xc, stride_xl,
#     stride_wc, stride_wg, stride_wk,
#     stride_ob, stride_oc, stride_ol,
#     BS: tl.constexpr,
# ):
#     pid_b = tl.program_id(0)
#     pid_c = tl.program_id(1)
#     pid_l = tl.program_id(2)

#     # 一维 block: 处理 BS 个连续的输出位置
#     b_offs = pid_b * BS + tl.arange(0, BS)  # (BS,)
#     l_offs = pid_l * BS + tl.arange(0, BS)  # (BS,)

#     # 逐元素 mask
#     mask_b = b_offs < batch
#     mask_l = (l_offs >= 0) & (l_offs < out_seq_len)
#     mask = mask_b & mask_l  # ← 一维！(BS,)

#     c_offs = pid_c
#     if c_offs >= out_channels:
#         return

#     group_id = c_offs // (out_channels // groups)
#     group_in_channels = in_channels // groups

#     acc = tl.zeros((BS,), dtype=tl.float32)

#     # Input channel loop
#     for c_in in range(group_in_channels):
#         x_c = group_id * group_in_channels + c_in
#         w_ptr_c = w_ptr + c_offs * (group_in_channels * kernel_size) + c_in * kernel_size

#         for k in range(kernel_size):
#             # 计算输入位置
#             l_in = l_offs * stride + k * dilation - padding  # (BS,)
#             mask_x = mask & (l_in >= 0) & (l_in < seq_len)   # (BS,)

#             # Load input: (BS,)
#             x_ptrs = x_ptr + b_offs * stride_xb + x_c * stride_xc + l_in * stride_xl
#             x_val = tl.load(x_ptrs, mask=mask_x, other=0.0)

#             # Load weight: scalar
#             w_val = tl.load(w_ptr_c + k)

#             acc += x_val * w_val  # (BS,) += (BS,) * scalar → (BS,)

#     # Store output
#     out_ptrs = out_ptr + b_offs * stride_ob + c_offs * stride_oc + l_offs * stride_ol
#     out_val = acc.to(x_ptr.dtype.element_ty)
#     tl.store(out_ptrs, out_val, mask=mask)


# # triton_conv1d_wrapper.py

# def triton_conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
#     """
#     x: (B, C_in, L)
#     weight: (C_out, C_in//groups, K)
#     return: (B, C_out, L_out)
#     """
#     B, C_in, L = x.shape
#     C_out, _, K = weight.shape

#     L_out = (L + 2 * padding - dilation * (K - 1) - 1) // stride + 1

#     out = torch.zeros((B, C_out, L_out), device=x.device, dtype=x.dtype)

#     # Strides
#     sx_b, sx_c, sx_l = x.stride()
#     sw_c, sw_g, sw_k = weight.stride()
#     so_b, so_c, so_l = out.stride()

#     def grid(meta):
#         return (
#             triton.cdiv(B, meta['BS']),
#             C_out,
#             triton.cdiv(L_out, meta['BS'])
#         )

#     triton_conv1d_kernel_v2[grid](
#         x, weight, out,
#         B, L, C_in, C_out, K, groups,
#         stride, padding, dilation, L_out,
#         sx_b, sx_c, sx_l,
#         sw_c, sw_g, sw_k,
#         so_b, so_c, so_l,
#         # BS=64,
#     )

#     if bias is not None:
#         out += bias.view(1, -1, 1)

#     return out



# # test_conv1d.py

# import torch
# import triton
# import time
# # from triton_conv1d_wrapper import triton_conv1d

# # 参数组合
# configs = [
#     dict(B=4, L=128, C_in=64, C_out=64, K=3, groups=1, stride=1, padding=1),
#     dict(B=4, L=512, C_in=128, C_out=128, K=5, groups=1, stride=1, padding=2),
#     dict(B=2, L=1024, C_in=256, C_out=256, K=7, groups=1, stride=1, padding=3),
#     dict(B=4, L=256, C_in=128, C_out=128, K=3, groups=2, stride=1, padding=1),
#     dict(B=1, L=2048, C_in=512, C_out=256, K=1, groups=1, stride=1, padding=0),
#     dict(B=8, L=64, C_in=64, C_out=128, K=15, groups=1, stride=2, padding=7),
# ]

# def benchmark_torch_and_triton():
#     print(f"{'Config':<40} {'Torch (ms)':<12} {'Triton (ms)':<12} {'Max Diff':<12} {'Correct'}")
#     print("-" * 80)

#     for cfg in configs:
#         B, L, C_in, C_out, K = cfg['B'], cfg['L'], cfg['C_in'], cfg['C_out'], cfg['K']
#         groups = cfg['groups']
#         stride, padding = cfg['stride'], cfg['padding']

#         # 计算输出长度
#         L_out = (L + 2 * padding - (K - 1) - 1) // stride + 1

#         x = torch.randn(B, C_in, L, device='cuda', dtype=torch.float32)
#         w = torch.randn(C_out, C_in // groups, K, device='cuda', dtype=torch.float32)
#         # bias = torch.randn(C_out, device='cuda')

#         # PyTorch
#         torch.cuda.synchronize()
#         start = time.time()
#         for _ in range(10):
#             with torch.no_grad():
#                 out_torch = torch.nn.functional.conv1d(
#                     x, w, None, stride=stride, padding=padding, dilation=1, groups=groups
#                 )
#         torch.cuda.synchronize()
#         torch_time = (time.time() - start) / 10 * 1000

#         # Triton
#         torch.cuda.synchronize()
#         start = time.time()
#         for _ in range(10):
#             with torch.no_grad():
#                 out_triton = triton_conv1d(x, w, bias=None, stride=stride, padding=padding, dilation=1, groups=groups)
#         torch.cuda.synchronize()
#         triton_time = (time.time() - start) / 10 * 1000

#         # 误差
#         diff = (out_torch - out_triton).abs().max().item()
#         correct = diff < 1e-2  # 由于数值精度，适当放宽

#         config_str = f"B{B}_L{L}_C{C_in}x{C_out}_K{K}_G{groups}"
#         print(f"{config_str:<40} {torch_time:<12.3f} {triton_time:<12.3f} {diff:<12.2e} {correct}")

# if __name__ == "__main__":
#     benchmark_torch_and_triton()


# import torch
# import triton
# import triton.language as tl
# from typing import Optional

# @triton.jit
# def conv1d_kernel(
#     input_ptr, weight_ptr, bias_ptr, output_ptr,
#     batch_size, seq_len, in_channels, out_channels, kernel_size,
#     stride, padding, dilation,
#     input_batch_stride, input_channel_stride, input_seq_stride,
#     weight_out_stride, weight_in_stride, weight_kernel_stride,
#     output_batch_stride, output_channel_stride, output_seq_stride,
#     groups,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     """
#     Triton kernel for grouped 1D convolution
    
#     Args:
#         groups: Number of groups for grouped convolution
#     """
#     pid = tl.program_id(0)
    
#     # Calculate output position
#     total_out_elements = batch_size * out_channels * ((seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)
    
#     if pid >= total_out_elements:
#         return
    
#     # Decompose pid
#     out_w = ((seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)
#     b = pid // (out_channels * out_w)
#     c_out = (pid // out_w) % out_channels
#     s_out = pid % out_w
    
#     # Calculate input positions
#     s_in_start = s_out * stride - padding
    
#     # Grouped convolution handling
#     group_size = in_channels // groups
#     group_id = c_out // (out_channels // groups)
#     in_c_start = group_id * group_size
#     in_c_end = in_c_start + group_size
    
#     # Initialize accumulator
#     acc = 0.0
    
#     # Perform convolution
#     for k in range(kernel_size):
#         s_in = s_in_start + k * dilation
        
#         if 0 <= s_in < seq_len:
#             for c_in in range(in_c_start, in_c_end):
#                 # Load input value
#                 input_idx = (
#                     b * input_batch_stride +
#                     c_in * input_channel_stride +
#                     s_in * input_seq_stride
#                 )
#                 x = tl.load(input_ptr + input_idx)
                
#                 # Load weight
#                 weight_idx = (
#                     c_out * weight_out_stride +
#                     c_in * weight_in_stride +
#                     k * weight_kernel_stride
#                 )
#                 w = tl.load(weight_ptr + weight_idx)
                
#                 acc += x * w
    
#     # Add bias if provided
#     if bias_ptr is not None:
#         bias_val = tl.load(bias_ptr + c_out)
#         acc += bias_val
    
#     # Store output
#     output_idx = (
#         b * output_batch_stride +
#         c_out * output_channel_stride +
#         s_out * output_seq_stride
#     )
#     tl.store(output_ptr + output_idx, acc)

# def triton_conv1d(
#     input: torch.Tensor,
#     weight: torch.Tensor,
#     bias: Optional[torch.Tensor] = None,
#     stride: int = 1,
#     padding: int = 0,
#     dilation: int = 1,
#     groups: int = 1
# ) -> torch.Tensor:
#     """
#     Triton implementation of 1D convolution
    
#     Args:
#         input: Input tensor of shape (batch_size, in_channels, seq_len)
#         weight: Weight tensor of shape (out_channels, in_channels // groups, kernel_size)
#         bias: Optional bias tensor of shape (out_channels,)
#         stride: Stride of the convolution
#         padding: Padding added to both sides of the input
#         dilation: Spacing between kernel elements
#         groups: Number of groups for grouped convolution
        
#     Returns:
#         Output tensor of shape (batch_size, out_channels, output_length)
#     """
#     assert input.device.type == "cuda", "Input must be on CUDA device"
#     assert weight.device.type == "cuda", "Weight must be on CUDA device"
#     if bias is not None:
#         assert bias.device.type == "cuda", "Bias must be on CUDA device"
    
#     batch_size, in_channels, seq_len = input.shape
#     out_channels, _, kernel_size = weight.shape
    
#     # Calculate output dimensions
#     output_length = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
#     # Initialize output tensor
#     output = torch.empty(
#         (batch_size, out_channels, output_length),
#         device=input.device,
#         dtype=input.dtype
#     )
    
#     # Launch Triton kernel
#     grid = lambda meta: (batch_size * out_channels * output_length,)
    
#     conv1d_kernel[grid](
#         input, weight, bias, output,
#         batch_size, seq_len, in_channels, out_channels, kernel_size,
#         stride, padding, dilation,
#         input.stride(0), input.stride(1), input.stride(2),
#         weight.stride(0), weight.stride(1), weight.stride(2),
#         output.stride(0), output.stride(1), output.stride(2),
#         groups,
#         BLOCK_SIZE=triton.next_power_of_2(output_length),
#     )
    
#     return output


# import time
# import numpy as np
# import pandas as pd
# # import matplotlib.pyplot as plt
# from typing import Dict, List, Tuple

# def benchmark_conv1d(
#     batch_size: int,
#     seq_len: int,
#     in_channels: int,
#     out_channels: int,
#     kernel_size: int,
#     groups: int = 1,
#     stride: int = 1,
#     padding: int = 0,
#     dilation: int = 1,
#     num_runs: int = 10,
#     warmup_runs: int = 3
# ) -> Dict[str, float]:
#     """Benchmark PyTorch vs Triton conv1d implementation"""
    
#     # Create input tensors
#     input_tensor = torch.randn(batch_size, in_channels, seq_len, device='cuda', dtype=torch.float32)
#     weight_tensor = torch.randn(out_channels, in_channels // groups, kernel_size, device='cuda', dtype=torch.float32)
#     bias_tensor = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    
#     # PyTorch implementation
#     torch_conv = torch.nn.Conv1d(
#         in_channels, out_channels, kernel_size,
#         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True
#     ).cuda()
    
#     with torch.no_grad():
#         torch_conv.weight.copy_(weight_tensor)
#         torch_conv.bias.copy_(bias_tensor)
    
#     # Warmup
#     for _ in range(warmup_runs):
#         _ = torch_conv(input_tensor)
#         _ = triton_conv1d(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups)
    
#     # Benchmark PyTorch
#     torch.cuda.synchronize()
#     start_time = time.perf_counter()
#     for _ in range(num_runs):
#         output_torch = torch_conv(input_tensor)
#     torch.cuda.synchronize()
#     torch_time = (time.perf_counter() - start_time) / num_runs
    
#     # Benchmark Triton
#     torch.cuda.synchronize()
#     start_time = time.perf_counter()
#     for _ in range(num_runs):
#         output_triton = triton_conv1d(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups)
#     torch.cuda.synchronize()
#     triton_time = (time.perf_counter() - start_time) / num_runs
    
#     # Verify correctness
#     with torch.no_grad():
#         max_diff = torch.abs(output_torch - output_triton).max().item()
#         mean_diff = torch.abs(output_torch - output_triton).mean().item()
    
#     return {
#         'torch_time_ms': torch_time * 1000,
#         'triton_time_ms': triton_time * 1000,
#         'speedup': torch_time / triton_time,
#         'max_diff': max_diff,
#         'mean_diff': mean_diff
#     }

# def run_comprehensive_benchmark():
#     """Run comprehensive benchmark with different configurations"""
    
#     configs = [
#         # (batch_size, seq_len, in_channels, out_channels, kernel_size, groups)
#         (1, 128, 64, 128, 3, 1),
#         (1, 512, 128, 256, 5, 1),
#         (4, 1024, 256, 512, 7, 1),
#         (8, 2048, 512, 1024, 3, 1),
#         (16, 4096, 256, 256, 5, 1),
#         (32, 8192, 512, 512, 7, 1),
#         # Grouped convolution tests
#         (4, 1024, 256, 512, 3, 2),
#         (4, 1024, 256, 512, 3, 4),
#         (4, 1024, 256, 512, 3, 8),
#         (8, 2048, 512, 1024, 5, 4),
#         # Different kernel sizes
#         (2, 1024, 256, 512, 1, 1),
#         (2, 1024, 256, 512, 3, 1),
#         (2, 1024, 256, 512, 5, 1),
#         (2, 1024, 256, 512, 7, 1),
#         (2, 1024, 256, 512, 11, 1),
#     ]
    
#     results = []
    
#     for config in configs:
#         batch_size, seq_len, in_channels, out_channels, kernel_size, groups = config
        
#         print(f"Testing config: batch={batch_size}, seq_len={seq_len}, "
#               f"in_ch={in_channels}, out_ch={out_channels}, "
#               f"kernel={kernel_size}, groups={groups}")
        
#         try:
#             result = benchmark_conv1d(*config)
#             result.update({
#                 'batch_size': batch_size,
#                 'seq_len': seq_len,
#                 'in_channels': in_channels,
#                 'out_channels': out_channels,
#                 'kernel_size': kernel_size,
#                 'groups': groups
#             })
#             results.append(result)
#             print(f"  PyTorch: {result['torch_time_ms']:.3f}ms, "
#                   f"Triton: {result['triton_time_ms']:.3f}ms, "
#                   f"Speedup: {result['speedup']:.2f}x, "
#                   f"Max diff: {result['max_diff']:.2e}")
#         except Exception as e:
#             print(f"  Error: {e}")
    
#     return pd.DataFrame(results)

# # def plot_benchmark_results(df: pd.DataFrame):
# #     """Plot benchmark results"""
    
# #     # fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
# #     # 1. Speedup vs Sequence Length
# #     ax = axes[0, 0]
# #     for kernel_size in sorted(df['kernel_size'].unique()):
# #         data = df[df['kernel_size'] == kernel_size]
# #         ax.plot(data['seq_len'], data['speedup'], 
# #                 marker='o', label=f'kernel={kernel_size}')
# #     ax.set_xlabel('Sequence Length')
# #     ax.set_ylabel('Speedup (PyTorch/Triton)')
# #     ax.set_title('Speedup vs Sequence Length')
# #     ax.legend()
# #     ax.grid(True)
    
# #     # 2. Speedup vs Embedding Size
# #     ax = axes[0, 1]
# #     for groups in sorted(df['groups'].unique()):
# #         data = df[df['groups'] == groups]
# #         ax.scatter(data['in_channels'], data['speedup'], 
# #                   label=f'groups={groups}', alpha=0.7)
# #     ax.set_xlabel('Input Channels')
# #     ax.set_ylabel('Speedup (PyTorch/Triton)')
# #     ax.set_title('Speedup vs Input Channels')
# #     ax.legend()
# #     ax.grid(True)
    
# #     # 3. Speedup vs Groups
# #     ax = axes[1, 0]
# #     group_data = df.groupby('groups')['speedup'].agg(['mean', 'std'])
# #     ax.errorbar(group_data.index, group_data['mean'], 
# #                 yerr=group_data['std'], marker='o', capsize=5)
# #     ax.set_xlabel('Number of Groups')
# #     ax.set_ylabel('Mean Speedup')
# #     ax.set_title('Speedup vs Number of Groups')
# #     ax.grid(True)
    
# #     # 4. Kernel Size Impact
# #     ax = axes[1, 1]
# #     kernel_data = df.groupby('kernel_size')['speedup'].agg(['mean', 'std'])
# #     ax.bar(kernel_data.index, kernel_data['mean'], 
# #            yerr=kernel_data['std'], capsize=5)
# #     ax.set_xlabel('Kernel Size')
# #     ax.set_ylabel('Mean Speedup')
# #     ax.set_title('Speedup vs Kernel Size')
# #     ax.grid(True)
    
#     # plt.tight_layout()
#     # plt.savefig('conv1d_benchmark_results.png', dpi=300, bbox_inches='tight')
#     # plt.show()

# if __name__ == "__main__":
#     # Run comprehensive benchmark
#     print("Running comprehensive conv1d benchmark...")
#     results_df = run_comprehensive_benchmark()
    
#     # Save results
#     results_df.to_csv('conv1d_benchmark_results.csv', index=False)
#     print("\nBenchmark results saved to 'conv1d_benchmark_results.csv'")
    
#     # Plot results
#     # plot_benchmark_results(results_df)
    
#     # Print summary
#     print("\n" + "="*60)
#     print("BENCHMARK SUMMARY")
#     print("="*60)
#     print(f"Total configurations tested: {len(results_df)}")
#     print(f"Average speedup: {results_df['speedup'].mean():.2f}x")
#     print(f"Best speedup: {results_df['speedup'].max():.2f}x")
#     print(f"Worst speedup: {results_df['speedup'].min():.2f}x")
#     print(f"Average max difference: {results_df['max_diff'].mean():.2e}")
#     print(f"Average mean difference: {results_df['mean_diff'].mean():.2e}")



@triton.autotune(
    configs=[
        triton.Config({'BS': 32}, num_stages=3, num_warps=4),
        triton.Config({'BS': 64}, num_stages=3, num_warps=4),
        triton.Config({'BS': 128}, num_stages=3, num_warps=4),
    ],
    key=['seq_len', 'kernel_size'],
)
@triton.jit
def triton_conv1d_kernel_v2(
    x_ptr, w_ptr, out_ptr,
    batch, seq_len, in_channels, out_channels, kernel_size, groups,
    stride, padding, dilation,
    out_seq_len,
    stride_xb, stride_xc, stride_xl,
    stride_wc, stride_wg, stride_wk,
    stride_ob, stride_oc, stride_ol,
    BS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)

    b_offs = pid_b * BS + tl.arange(0, BS)
    l_offs = pid_l * BS + tl.arange(0, BS)

    mask_b = b_offs < batch
    mask_l = (l_offs >= 0) & (l_offs < out_seq_len)
    mask = mask_b & mask_l  # (BS,)

    c_offs = pid_c
    if c_offs >= out_channels:
        return

    # --- Fix: Correct group and local channel indexing ---
    group_size_out = out_channels // groups
    group_size_in = in_channels // groups

    group_id = c_offs // group_size_out
    local_c_out = c_offs % group_size_out  # local output channel in group

    acc = tl.zeros((BS,), dtype=tl.float32)

    # Input channel loop
    for c_in in range(group_size_in):
        x_c = group_id * group_size_in + c_in  # global input channel

        # Base pointer for weight[local_c_out, c_in, :]
        w_base = w_ptr + \
            group_id * (group_size_out * group_size_in * kernel_size) + \
            local_c_out * (group_size_in * kernel_size) + \
            c_in * kernel_size

        for k in range(kernel_size):
            l_in = l_offs * stride + k * dilation - padding
            mask_x = mask & (l_in >= 0) & (l_in < seq_len)

            # Input pointer
            x_ptrs = x_ptr + b_offs * stride_xb + x_c * stride_xc + l_in * stride_xl
            x_val = tl.load(x_ptrs, mask=mask_x, other=0.0)

            # Weight: scalar
            w_val = tl.load(w_base + k)

            acc += x_val.to(tl.float32) * w_val

    # Store output
    out_ptrs = out_ptr + b_offs * stride_ob + c_offs * stride_oc + l_offs * stride_ol
    out_val = acc.to(x_ptr.dtype.element_ty)
    tl.store(out_ptrs, out_val, mask=mask)




def triton_conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    x = x.contiguous()
    weight = weight.contiguous()

    B, C_in, L = x.shape
    C_out, _, K = weight.shape
    L_out = (L + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    out = torch.zeros((B, C_out, L_out), device=x.device, dtype=x.dtype)

    sx_b, sx_c, sx_l = x.stride()
    sw_c, sw_g, sw_k = weight.stride()  # 注意：sw_g 可能用不到
    so_b, so_c, so_l = out.stride()

    def grid(meta):
        return (
            triton.cdiv(B, meta['BS']),
            C_out,
            triton.cdiv(L_out, meta['BS'])
        )

    triton_conv1d_kernel_v2[grid](
        x, weight, out,
        B, L, C_in, C_out, K, groups,
        stride, padding, dilation, L_out,
        sx_b, sx_c, sx_l,
        sw_c, sw_g, sw_k,
        so_b, so_c, so_l,
    )

    if bias is not None:
        out += bias.view(1, -1, 1)

    return out



# 小规模测试
x = torch.randn(2, 8, 16).cuda()
w = torch.randn(8, 4, 3).cuda()  # groups=2
out_torch = torch.nn.functional.conv1d(x, w, padding=1, groups=2)
print("torch output: {}\n".format(out_torch))
out_triton = triton_conv1d(x, w, padding=1, groups=2)
print("triton output: {}\n".format(out_triton))
print((out_torch - out_triton).abs().max())  # 应该 < 1e-5



@triton.autotune(
    configs=[
        triton.Config({'BS_b': 2, 'BS_l': 32, 'CS': 32}, num_stages=3, num_warps=4),
        triton.Config({'BS_b': 2, 'BS_l': 64, 'CS': 64}, num_stages=3, num_warps=4),
        triton.Config({'BS_b': 2, 'BS_l': 128, 'CS': 128}, num_stages=3, num_warps=4),
    ],
    key=['seq_len', 'kernel_size'],
)
@triton.jit
def triton_conv1d_kernel_v3(
    x_ptr, w_ptr, out_ptr,
    batch, seq_len, in_channels, out_channels, kernel_size, groups,
    stride, padding, dilation,
    out_seq_len,
    stride_xb, stride_xc, stride_xl,
    stride_wc, stride_wg, stride_wk,
    stride_ob, stride_oc, stride_ol,
    BS_b: tl.constexpr,  # batch block size
    BS_l: tl.constexpr,  # sequence block size
    CS: tl.constexpr,    # output channel block size
):
    pid_b = tl.program_id(0)
    pid_bl = tl.program_id(1)  # combined b and l
    pid_c = tl.program_id(2)   # output channel block

    # --- Compute batch and sequence offsets ---
    b_offs = pid_b * BS_b + tl.arange(0, BS_b)  # (BS_b,)
    l_offs = (pid_bl * BS_l + tl.arange(0, BS_l))  # (BS_l,)

    mask_b = b_offs < batch
    mask_l = (l_offs >= 0) & (l_offs < out_seq_len)
    mask_bl = mask_b[:, None] & mask_l[None, :]  # (BS_b, BS_l)

    # --- Output channel block ---
    c_offs = pid_c * CS + tl.arange(0, CS)
    mask_c = c_offs < out_channels
    if tl.sum(mask_c) == 0:
        return

    # --- Group setup ---
    group_size_out = out_channels // groups
    group_size_in = in_channels // groups

    group_id = c_offs // group_size_out
    local_c_out = c_offs % group_size_out

    # --- Acc for each (b,l,c) combination ---
    acc = tl.zeros((BS_b, BS_l, CS), dtype=tl.float32)

    # Input channel loop
    for c_in in range(group_size_in):
        x_c = group_id * group_size_in + c_in  # broadcasted

        # Base weight pointer for each output channel
        w_base = w_ptr + \
            group_id * (group_size_out * group_size_in * kernel_size) + \
            local_c_out * (group_size_in * kernel_size) + \
            c_in * kernel_size  # (CS,)

        for k in range(kernel_size):
            l_in = l_offs * stride + k * dilation - padding  # (BS_l,)
            mask_x = mask_bl[:, :] & ((l_in >= 0) & (l_in < seq_len))[None, :]  # (BS_b, BS_l)

            # Load x: (BS_b, BS_l)
            x_ptrs = x_ptr + b_offs[:, None] * stride_xb + x_c[None, None] * stride_xc + l_in[None, :] * stride_xl
            x_val = tl.load(x_ptrs, mask=mask_x, other=0.0)  # (BS_b, BS_l)

            # Load w: (CS,)
            w_val = tl.load(w_base + k)  # (CS,)

            # Outer product: (BS_b, BS_l) @ (CS,) -> (BS_b, BS_l, CS)
            acc += x_val[:, :, None] * w_val[None, None, :]

    # Store output
    out_ptrs = out_ptr + \
        b_offs[:, None, None] * stride_ob + \
        c_offs[None, None, :] * stride_oc + \
        l_offs[None, :, None] * stride_ol

    out_val = acc.to(x_ptr.dtype.element_ty)
    tl.store(out_ptrs, out_val, mask=mask_bl[:, :, None] & mask_c[None, None, :])