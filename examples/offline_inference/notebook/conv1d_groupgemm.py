# import torch
# import triton
# import triton.language as tl

# # ==============================
# # Step 1: 复用你已有的 grouped_matmul_kernel
# # ==============================

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_SM': 84}),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_SM': 84}),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
#     ],
#     key=['group_size'],
# )
# @triton.jit
# def grouped_matmul_kernel(
#     group_a_ptrs,
#     group_b_ptrs,
#     group_c_ptrs,
#     group_gemm_sizes,
#     g_lds,
#     group_size,
#     NUM_SM: tl.constexpr,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
# ):
#     tile_idx = tl.program_id(0)
#     last_problem_end = 0
#     for g in range(group_size):
#         gm = tl.load(group_gemm_sizes + g * 3)
#         gn = tl.load(group_gemm_sizes + g * 3 + 1)
#         gk = tl.load(group_gemm_sizes + g * 3 + 2)
#         num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
#         num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
#         num_tiles = num_m_tiles * num_n_tiles

#         while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
#             k = gk
#             lda = tl.load(g_lds + g * 3)
#             ldb = tl.load(g_lds + g * 3 + 1)
#             ldc = tl.load(g_lds + g * 3 + 2)
#             a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
#             b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
#             c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

#             tile_idx_in_gemm = tile_idx - last_problem_end
#             tile_m_idx = tile_idx_in_gemm // num_n_tiles
#             tile_n_idx = tile_idx_in_gemm % num_n_tiles

#             offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#             offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#             offs_k = tl.arange(0, BLOCK_SIZE_K)

#             a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
#             b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]

#             accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#             for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
#                 tl.multiple_of(a_ptrs, [16, 16])
#                 tl.multiple_of(b_ptrs, [16, 16])
#                 a = tl.load(a_ptrs)
#                 b = tl.load(b_ptrs)
#                 accumulator += tl.dot(a, b)
#                 a_ptrs += BLOCK_SIZE_K
#                 b_ptrs += BLOCK_SIZE_K * ldb

#             c = accumulator.to(tl.float16)

#             offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#             offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#             c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

#             tl.store(c_ptrs, c)

#             tile_idx += NUM_SM

#         last_problem_end += num_tiles


# def group_gemm_fn(group_A, group_B):
#     device = torch.device('cuda')
#     assert len(group_A) == len(group_B)
#     group_size = len(group_A)

#     # ✅ 检查 contiguous 和对齐
#     for i, A in enumerate(group_A):
#         assert A.is_contiguous(), f"A[{i}] not contiguous"
#         assert (A.data_ptr() % 16 == 0), f"A[{i}] ptr={A.data_ptr():x} not 16-byte aligned"
#     for i, B in enumerate(group_B):
#         assert B.is_contiguous(), f"B[{i}] not contiguous"
#         assert (B.data_ptr() % 16 == 0), f"B[{i}] ptr={B.data_ptr():x} not 16-byte aligned"

#     A_addrs = []
#     B_addrs = []
#     C_addrs = []
#     g_sizes = []
#     g_lds = []
#     group_C = []

#     for i in range(group_size):
#         A = group_A[i]
#         B = group_B[i]
#         assert A.shape[1] == B.shape[0], f"Shape mismatch: {A.shape} @ {B.shape}"
#         M, K = A.shape
#         K, N = B.shape
#         C = torch.empty((M, N), device=device, dtype=A.dtype)
#         group_C.append(C)
#         A_addrs.append(A.data_ptr())
#         B_addrs.append(B.data_ptr())
#         C_addrs.append(C.data_ptr())
#         g_sizes += [M, N, K]
#         g_lds += [A.stride(0), B.stride(0), C.stride(0)]

#     d_a_ptrs = torch.tensor(A_addrs, device=device)
#     d_b_ptrs = torch.tensor(B_addrs, device=device)
#     d_c_ptrs = torch.tensor(C_addrs, device=device)
#     d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
#     d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)

#     grid = lambda META: (META['NUM_SM'],)
#     grouped_matmul_kernel[grid](
#         d_a_ptrs, d_b_ptrs, d_c_ptrs,
#         d_g_sizes, d_g_lds,
#         group_size,
#     )
#     return group_C


# def im2col_1d(x, kernel_size, stride=1, padding=0, dilation=1):
#     """
#     Perform im2col on 1D tensor.
    
#     Args:
#         x: [B, C, L] tensor
#         kernel_size: int
#         stride, padding, dilation: int
    
#     Returns:
#         x_col: [B, L_out, C * kernel_size] tensor
#     """
#     B, C, L = x.shape
#     k = kernel_size

#     # 计算有效长度
#     Lp = L + 2 * padding
#     # 使用 F.pad 进行填充
#     x = torch.nn.functional.pad(x, (padding, padding), mode='constant', value=0)

#     # 计算输出长度
#     L_out = (Lp - dilation * (k - 1) - 1) // stride + 1
#     if L_out <= 0:
#         raise ValueError(f"Output length <= 0: L={L}, k={k}, padding={padding}, stride={stride}, dilation={dilation}")

#     # 使用 as_strided 构造滑动窗口
#     # 每个窗口起始位置: i, i+stride, i+2*stride, ...
#     # 每个窗口内取 k 个点，步长为 dilation

#     # 输出形状: [B, L_out, C, k]
#     output_shape = (B, L_out, C, k)

#     # 步长计算
#     x_stride_b, x_stride_c, x_stride_l = x.stride()
#     # 每个窗口内，元素之间的步长（考虑 dilation）
#     # 窗口维度的步长
#     new_stride = (
#         x_stride_b,           # batch 维度
#         stride * x_stride_l,  # 输出序列维度：每次跳 stride
#         x_stride_c,           # 通道维度
#         dilation * x_stride_l # 卷积核内维度：每次跳 dilation
#     )

#     # 使用 as_strided 构造视图
#     x_strided = x.as_strided(output_shape, new_stride)

#     # 展平 -> [B, L_out, C * k]
#     x_col = x_strided.reshape(B, L_out, C * k)

#     return x_col


# def conv1d_group_gemm(x, weight, bias=None, stride=1, padding=0, dilation=1):
#     device = x.device
#     B, C, L_in = x.shape
#     _, _, K = weight.shape
#     assert weight.shape == (C, C, K)
#     assert x.dtype == weight.dtype

#     # im2col_1d 返回的是 view，可能不连续
#     x_col = im2col_1d(x, kernel_size=K, stride=stride, padding=padding, dilation=dilation)
#     # x_col: [B, L_out, C*K]

#     w_flat = weight.reshape(C, C * K)
#     W_T = w_flat.t().contiguous()  # [C*K, C]

#     # ✅ 关键：强制 contiguous
#     group_A = [x_col[b].contiguous() for b in range(B)]  # [L_out, C*K] 连续
#     group_B = [W_T] * B

#     # 执行 Grouped GEMM
#     group_C = group_gemm_fn(group_A, group_B)  # list of [L_out, C]

#     # 合并
#     y = torch.stack([c.transpose(0, 1) for c in group_C], dim=0)  # [B, C, L_out]

#     if bias is not None:
#         y = y + bias.view(1, C, 1)

#     return y
# # def conv1d_group_gemm(x, weight, bias=None, stride=1, padding=0, dilation=1):
# #     """
# #     Conv1d using Grouped GEMM (via im2col)
    
# #     Args:
# #         x: [B, C, L] float16/fp32, on CUDA
# #         weight: [C, C, K] (since in_channels == out_channels)
# #         bias: [C] (optional)
# #         stride, padding, dilation: int
    
# #     Returns:
# #         y: [B, C, L_out] 
# #     """
# #     device = x.device
# #     B, C, L_in = x.shape
# #     _, _, K = weight.shape  # C, C, K
# #     assert weight.shape == (C, C, K), f"Expected weight {C}x{C}x{K}, got {weight.shape}"
# #     assert x.dtype == weight.dtype, "x and weight must have same dtype"

# #     # 计算输出长度
# #     L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
# #     if L_out <= 0:
# #         raise ValueError(f"Output length <= 0: L_in={L_in}, K={K}, padding={padding}, stride={stride}")

# #     # Step 1: 对每个 batch 做 im2col
# #     # 使用 torch.nn.Unfold 模拟 im2col
# #     unfold = torch.nn.Unfold(
# #         kernel_size=(K,), 
# #         dilation=dilation, 
# #         padding=padding, 
# #         stride=stride
# #     )
# #     # x: [B, C, L] -> unfold -> [B, C*K, L_out]
# #     x_unfold = unfold(x.unsqueeze(2))  # 添加 dummy H=1 维度
# #     # x_unfold: [B, C*K, L_out] -> 转置 -> [B, L_out, C*K]
# #     x_unfold = x_unfold.transpose(1, 2)  # [B, L_out, C*K]

# #     # Step 2: 展平卷积核
# #     # weight: [C, C, K] -> [C, C*K]
# #     w_flat = weight.reshape(C, C * K)  # [C, C*K]

# #     # Step 3: 构造 Grouped GEMM 输入
# #     group_A = []  # 每个是 [L_out, C*K]
# #     group_B = []  # 每个是 [C*K, C] -> 转置后 [C, C*K]，但我们要做 A @ W^T，所以传 W^T

# #     W_T = w_flat.t().contiguous()  # [C*K, C]

# #     for b in range(B):
# #         A = x_unfold[b]  # [L_out, C*K]
# #         group_A.append(A)
# #         group_B.append(W_T)  # 权重复用

# #     # Step 4: 调用 grouped GEMM
# #     # 注意：group_gemm_fn 计算的是 C = A @ B
# #     # 我们这里 A: [L_out, C*K], B: [C*K, C] → C: [L_out, C]
# #     group_C = group_gemm_fn(group_A, group_B)  # list of [L_out, C]

# #     # Step 5: 合并结果
# #     y = torch.stack(group_C, dim=0)  # [B, L_out, C]
# #     y = y.transpose(1, 2).contiguous()  # [B, C, L_out]

# #     # Step 6: 加 bias
# #     if bias is not None:
# #         y = y + bias.view(1, C, 1)

# #     return y


# # 参数设置
# B, C, L_in, K = 4, 4096, 100, 4

# # 创建数据
# x = torch.randn(B, C, L_in, device='cuda', dtype=torch.float16)
# weight = torch.randn(C, C, K, device='cuda', dtype=torch.float16)
# bias = torch.randn(C, device='cuda', dtype=torch.float16)

# # 执行
# y_custom = conv1d_group_gemm(x, weight, bias, padding=0, stride=1)

# print(y_custom.shape)  # torch.Size([4, 4096, 97])

# # ✅ 对比 PyTorch 原生结果（注意：PyTorch Conv1d 是 [B,C,L]，weight [out,in//g,K]）
# # 由于 groups=1，in==out，可以直接用
# conv = torch.nn.Conv1d(C, C, kernel_size=K, stride=1, padding=0, bias=True).to('cuda').to(torch.float16)
# conv.weight.data = weight
# conv.bias.data = bias

# with torch.no_grad():
#     y_torch = conv(x)
# print(y_torch.shape)  # torch.Size([4, 4096, 97])

# # 数值对比
# print("Max diff:", (y_custom - y_torch).abs().max().item())  # 应该很小，< 1e-2 (fp16)



import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv1d_tiled_kernel(
    x_ptr,           # [B, C, L_in]
    w_ptr,           # [C, C, K]
    b_ptr,           # [C]
    y_ptr,           # [B, C, L_out]
    B, C, L_in, L_out, K,
    stride_xc,       # x.stride(1) = L_in
    stride_wc,       # w.stride(1) = K*C
    stride_wk,       # w.stride(2) = 1
    stride_yc,       # y.stride(1) = L_out
    padding,
    stride_,
    # Block sizes (constexpr)
    BLOCK_B: tl.constexpr,  # always 1
    BLOCK_C: tl.constexpr,  # e.g., 128
    BLOCK_L: tl.constexpr,  # e.g., 16
    BLOCK_K: tl.constexpr,  # e.g., 4
):
    # 3D grid: (B, num_L_blocks, num_C_blocks)
    b = tl.program_id(0)
    l_block_idx = tl.program_id(1)
    c_block_idx = tl.program_id(2)

    if b >= B:
        return

    # --- 计算当前 block 负责的 l_out 和 c 范围 ---
    l_start = l_block_idx * BLOCK_L
    c_start = c_block_idx * BLOCK_C

    l_end = min(l_start + BLOCK_L, L_out)
    c_end = min(c_start + BLOCK_C, C)

    offs_l = l_start + tl.arange(0, BLOCK_L)  # [BLOCK_L]
    offs_c = c_start + tl.arange(0, BLOCK_C)  # [BLOCK_C]

    # 有效掩码
    l_mask = (offs_l < L_out)  # [BLOCK_L]
    c_mask = (offs_c < C)      # [BLOCK_C]

    # --- 初始化累加器 ---
    # acc[l, c] for current block
    acc = tl.zeros((BLOCK_L, BLOCK_C), dtype=tl.float32)

    # --- 遍历 kernel size K ---
    for k in range(0, K):
        # 计算输入时间步
        l_in = offs_l * stride_ - padding + k
        # 判断是否在 [0, L_in) 范围内
        in_bounds = (l_in >= 0) & (l_in < L_in)  # [BLOCK_L]

        # 加载 x[b, :, l_in] -> shape [BLOCK_L, C]
        # 我们只加载当前 c_block
        x_ptrs = x_ptr + b * C * L_in + offs_c[None, :] * stride_xc + l_in[:, None]
        # 掩码：l_in 有效 且 c 有效
        x_mask = in_bounds[:, None] & c_mask[None, :]
        x_val = tl.load(x_ptrs, mask=x_mask, other=0.0, cache_modifier=".cg")  # [BLOCK_L, BLOCK_C]

        # 加载 w[c_out, :, k] -> w[c_out, c_in, k]
        # w_ptr shape: [C, C, K] -> stride: (C*K, K, 1)
        w_ptrs = w_ptr + offs_c[:, None] * stride_wc + offs_c[None, :] * stride_wk + k
        # 掩码：c_in 和 c_out 都有效
        w_mask = c_mask[:, None] & c_mask[None, :]
        w_val = tl.load(w_ptrs, mask=w_mask, other=0.0, cache_modifier=".cg")  # [BLOCK_C, BLOCK_C]

        # --- GEMM: acc[l, c_out] += x[l, c_in] * w[c_out, c_in, k] ---
        # 使用 tl.dot: x [BLOCK_L, BLOCK_C] @ w.T [BLOCK_C, BLOCK_C] -> [BLOCK_L, BLOCK_C]
        w_val_t = w_val
        acc += tl.dot(x_val, w_val_t.to(tl.float32))

    # --- 存储输出 ---
    y_ptrs = y_ptr + b * C * L_out + offs_c[None, :] * stride_yc + offs_l[:, None]
    y_mask = l_mask[:, None] & c_mask[None, :]
    tl.store(y_ptrs, acc.to(tl.float16), mask=y_mask)

    # bias 在外面加（可 fused，但这里保持简洁）



def fused_conv1d_tiled(x, weight, bias=None, padding=0, stride_=1):
    B, C, L_in = x.shape
    C_out, C_in, K = weight.shape
    assert C == C_in == C_out, "Only support C_in == C_out == C"
    L_out = (L_in + 2 * padding - K) // stride_ + 1
    assert L_out > 0

    y = torch.zeros(B, C, L_out, device=x.device, dtype=x.dtype)

    # 块大小选择
    BLOCK_C = 128  # C=4096 -> 4096/128 = 32 blocks
    BLOCK_L = triton.next_power_of_2(L_out) if L_out < 64 else 32
    BLOCK_L = min(BLOCK_L, 32)  # 每个 block 处理 16~32 个 l_out
    BLOCK_K = min(triton.next_power_of_2(K), 8)

    # 网格
    num_l_blocks = triton.cdiv(L_out, BLOCK_L)
    num_c_blocks = triton.cdiv(C, BLOCK_C)
    grid = (B, num_l_blocks, num_c_blocks)

    # 启动 kernel
    fused_conv1d_tiled_kernel[grid](
        x, weight, bias, y,
        B, C, L_in, L_out, K,
        x.stride(1),     # stride_xc = L_in
        weight.stride(1), # stride_wc = K*C
        weight.stride(2), # stride_wk = 1
        y.stride(1),     # stride_yc = L_out
        padding, stride_,
        BLOCK_B=1,
        BLOCK_C=BLOCK_C,
        BLOCK_L=BLOCK_L,
        BLOCK_K=BLOCK_K,
    )

    # 加 bias
    if bias is not None:
        y += bias.view(1, C, 1)

    return y


# 参数
B, C, L_in, K = 2, 4096, 1024, 4
padding, stride_ = 1, 1

x = torch.randn(B, C, L_in, device='cuda', dtype=torch.float16)
weight = torch.randn(C, C, K, device='cuda', dtype=torch.float16)
bias = torch.randn(C, device='cuda', dtype=torch.float16)

# Triton
# %timeit -n 10 -r 5 
fused_conv1d_tiled(x, weight, bias, padding=padding, stride_=stride_).cuda()

# PyTorch
conv_torch = torch.nn.Conv1d(C, C, K, padding=padding, stride=stride_, bias=True).cuda().to(torch.float16)
conv_torch.weight.data = weight
conv_torch.bias.data = bias
with torch.no_grad():
    # %timeit -n 10 -r 5 
    conv_torch(x).cuda()