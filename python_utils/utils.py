import numpy as np
from scipy.interpolate import interp1d
from skimage.transform import AffineTransform, warp
from matplotlib.patches import Ellipse
from matplotlib.widgets import EllipseSelector
from scipy.ndimage import gaussian_filter
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# from cucim.skimage import transform

def interp_transformations(tform_array, lambda_vals, method='spline'):
    """
    对输入的 tform_array 插值，返回插值后的 transformation 矩阵数组。

    参数:
      tform_array: 3D 或 4D 的 NumPy 数组。
                   - 若为 3D，则尺寸为 (C, 3, 3)，表示只有 1 个视图、C 帧。
                   - 若为 4D，则尺寸为 (B, C, 3, 3)，表示 B 个视图、C 帧。
      lambda_vals: 归一化深度值序列，可以是 list 或 numpy.array，长度为 num_depth。
      method: 插值方法，默认 'spline'，在 Python 中对应 SciPy 的 'cubic'。
              也可以指定 'linear'、'quadratic' 等其他插值方法。

    返回:
      - 若输入为 4D 数组 (B, C, 3, 3)，返回 4D 数组 (B, num_depth, 3, 3)。
      - 若输入为 3D 数组 (C, 3, 3)，返回 3D 数组 (num_depth, 3, 3)。
    """
    # 将 'spline' 映射为 SciPy 中的 'cubic'
    if method.lower() == 'spline':
        interp_method = 'cubic'
    else:
        interp_method = method.lower()

    # 判断输入维度，只允许 3D 或 4D
    if tform_array.ndim not in [3, 4]:
        raise ValueError(f"tform_array 必须是 3D 或 4D 数组，当前维度 = {tform_array.ndim}")

    # 若只有 3D，则视为单视图，先在最前面扩展出“视图”维度 (B=1)
    single_view = (tform_array.ndim == 3)
    if single_view:
        # 扩展为 (1, C, 3, 3)，便于统一后续处理
        tform_array = tform_array[np.newaxis, ...]

    # 读取形状：B 表示视图数，C 表示帧数
    B, C, _, _ = tform_array.shape
    num_depth = len(lambda_vals)

    # 原始插值采样点，假设帧数从 0 均匀到 1
    xi = np.linspace(0, 1, C)

    # 预先分配结果：插值后为 (B, num_depth, 3, 3)
    tform_interp = np.zeros((B, num_depth, 3, 3), dtype=tform_array.dtype)

    for i in range(B):
        # 取出第 i 个视图的所有帧，形状 (C, 3, 3)
        # 按列优先 (Fortran 顺序 'F') 将每帧 3x3 拉平为长度 9 的向量
        # 得到形状 (C, 9)
        elements = tform_array[i].reshape(C, 9, order='F')

        # 用于存放插值结果，形状 (num_depth, 9)
        elements_interp = np.zeros((num_depth, 9), dtype=tform_array.dtype)

        # 对矩阵展开后每个元素做 1D 插值
        for k in range(9):
            interp_fn = interp1d(
                xi,
                elements[:, k],
                kind=interp_method,
                fill_value="extrapolate"  # 保证超出原 xi 范围时也能计算
            )
            elements_interp[:, k] = interp_fn(lambda_vals)

        # 将插值后的每个向量 (num_depth, 9) 再 reshape 回 (num_depth, 3, 3)
        # 同样要用列优先 'F' 恢复矩阵原始布局
        tform_interp[i] = elements_interp.reshape(num_depth, 3, 3, order='F')

    # 如果原始输入是 3D，那么只返回 (num_depth, 3, 3)，去掉开头的视图维度
    if single_view:
        tform_interp = tform_interp[0]

    return tform_interp

def interpolate_PSFs(PSFs, lambda_vals, pca_k):
    """
    使用 PCA + 插值方法对 PSFs 做插值
    
    参数:
    PSFs: 4D ndarray，形状 (patchHeight, patchWidth, num_frame, num_view)
    lambda_vals: 需要插值到的深度序列，长度为 num_depth
    pca_k: 取前 pca_k 个 PCA 分量参与重建
    
    返回:
    PSF_interp: 4D ndarray，形状 (patchHeight, patchWidth, num_depth, num_view)
    V_interp:   3D ndarray，形状 (num_view, num_depth, pca_k)
    """
    # 读取维度信息，对应 MATLAB 的 [patchHeight, patchWidth, num_frame, num_view] = size(PSFs)
    patchHeight, patchWidth, num_frame, num_view = PSFs.shape
    num_depth = len(lambda_vals)
    
    # 初始化输出
    PSF_interp = np.zeros((patchHeight, patchWidth, num_depth, num_view), dtype=PSFs.dtype)
    V_interp = np.zeros((num_view, num_depth, pca_k), dtype=PSFs.dtype)
    
    # 构造用于插值的原始位置 [0, 1]，与 MATLAB 的 linspace(0, 1, num_frame) 对应
    x_original = np.linspace(0, 1, num_frame)
    
    # 逐视角处理
    for i in range(num_view):
        # -- 1) 进行 SVD 分解 --
        #   MATLAB: A = reshape(PSFs(:,:,:,i), [], num_frame)
        #   注意: MATLAB reshape 默认列优先；Python 需要 order='F' 才能对齐。
        A = PSFs[:, :, :, i].reshape((-1, num_frame), order='F')
        
        #   MATLAB: [U, S, V] = svd(A, 'econ')
        #   Python: 返回 U, s(奇异值), Vt = svd(...); 其中 V = Vt.T
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V_ = Vt.T  # V_ 对应 MATLAB 中的 V
        
        # 取前 pca_k 个分量
        U_pca = U[:, :pca_k]              # (M, pca_k)
        S_pca = np.diag(s[:pca_k])        # (pca_k, pca_k)
        V_pca = V_[:, :pca_k]            # (num_frame, pca_k)
        
        # -- 2) 对 V_pca 做插值 --
        #   MATLAB: V_interp(i,:,j) = interp1(linspace(0,1,num_frame), V_pca(:,j), lambda, 'spline', 'extrap')
        #   Python: interp1d(..., kind='cubic', fill_value='extrapolate') 即可等效 'spline'
        for j in range(pca_k):
            f_interp = interp1d(
                x_original, 
                V_pca[:, j],
                kind='cubic',               # 对应 'spline'
                fill_value='extrapolate'
            )
            V_interp[i, :, j] = f_interp(lambda_vals)
        
        # -- 3) 用插值后的 V_interp 重建 PSF --
        #   MATLAB: X_recon = real(U_pca * S_pca * squeeze(V_interp(i,:,:))')
        #   这里 V_interp(i,:,:) 形状 -> (num_depth, pca_k)
        #   所以转置后变 (pca_k, num_depth)
        X_recon = U_pca @ S_pca @ V_interp[i, :, :].T  # (M, num_depth)
        X_recon = np.real(X_recon)
        
        # 负值截断为 0
        X_recon[X_recon < 0] = 0
        
        # 重排回 (patchHeight, patchWidth, num_depth)
        #   MATLAB: reshape(X_recon, patchHeight, patchWidth, num_depth)
        X_recon_reshaped = X_recon.reshape((patchHeight, patchWidth, num_depth), order='F')
        
        # 存到结果中
        PSF_interp[:, :, :, i] = X_recon_reshaped
        
        # 归一化
        total_sum = np.sum(PSF_interp[:, :, :, i])
        if total_sum > 0:
            PSF_interp[:, :, :, i] /= total_sum

    return PSF_interp, V_interp


def shift_and_sum_gpu(measurements, backward_list):
    """
    使用 PyTorch 在 GPU 上实现 shift-and-sum 重建，并通过 Welford 算法
    在线计算重建结果的均值和方差。

    参数:
    ----------
    measurements : np.ndarray
        形状 (Ny, Nx, nSamples, nViews) 的 4D 数组 (CPU上的 NumPy 数组)。
    backward_list : np.ndarray
        形状 (nViews, nDepths, 3, 3) 的 4D 数组，每个 (3,3) 矩阵为仿射变换，
        若是正向变换，需要取其逆。

    返回:
    ----------
    mean_recon : np.ndarray
        shift-and-sum 重建结果在所有样本上的平均值 (形状：Ny, Nx, nDepths)，CPU 上。
    var_recon : np.ndarray
        shift-and-sum 重建结果在所有样本上的方差 (形状：Ny, Nx, nDepths)，CPU 上。
    """

    # 判断是否有可用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 将输入转为 torch 张量，保证为 float32
    measurements_t = torch.tensor(measurements, dtype=torch.float32, device=device)
    backward_list_t = torch.tensor(backward_list, dtype=torch.float32, device=device)

    # 获取各维尺寸
    Ny, Nx, nSamples, nViews = measurements_t.shape
    _, nDepths, _, _ = backward_list_t.shape  # backward_list.shape == (nViews, nDepths, 3, 3)

    # 构造归一化坐标转换矩阵（基于 align_corners=True）
    # 注意：网格采样中的坐标范围是 [-1,1]
    T_mat = torch.tensor([[2/(Nx-1),       0, -1],
                          [      0, 2/(Ny-1), -1],
                          [      0,       0,  1]], dtype=torch.float32, device=device)
    
    T_inv = torch.tensor([[(Nx-1)/2,       0, (Nx-1)/2],
                          [      0, (Ny-1)/2, (Ny-1)/2],
                          [      0,       0,        1]], dtype=torch.float32, device=device)
    
    # 初始化 Welford 算法累积变量
    mean_ssResult = None  # 累计均值，形状 (Ny, Nx, nDepths)
    M2 = None             # 累计平方差
    count = 0

    # 主循环：遍历每个样本，进行重建并更新在线统计量
    for s in range(nSamples):
        # 取出第 s 个样本: shape (Ny, Nx, nViews)
        sample_data = measurements_t[:, :, s, :]
        # ssResult_sample 用于存储本样本的 3D 重建结果，形状 (Ny, Nx, nDepths)
        ssResult_sample = torch.zeros((Ny, Nx, nDepths), dtype=torch.float32, device=device)

        for j in range(nDepths):
            # 对于每个深度 j，从所有视角获得重建图像进行累加
            warped_views = []  # 用于存储 nViews 张 warped 图像，每张形状 (Ny, Nx)

            for i in range(nViews):
                # 当前视角的图像，形状 (Ny, Nx)
                im_mov = sample_data[:, :, i]
                # 扩展为 4D 张量：(1, 1, Ny, Nx)
                im_mov_4d = im_mov.unsqueeze(0).unsqueeze(0)

                # 取出该视角在深度 j 上的变换矩阵 A (3x3)
                A = backward_list_t[i, j, :, :]
                # 如果传入的是正向矩阵，需要取其逆；与原 CPU 代码中 AffineTransform(matrix=A).inverse 对应
                A_inv = torch.inverse(A)

                # 将 A_inv 从像素坐标转换到归一化坐标系下
                # A_norm = T_mat * A_inv * T_inv
                A_norm = T_mat @ A_inv @ T_inv
                # 拿到 affine_grid 需要的 2x3 矩阵（丢弃最后一行）
                theta = A_norm[:2, :]
                # 增加一个 batch 维度：shape (1, 2, 3)
                theta = theta.unsqueeze(0)

                # 生成采样网格，注意 grid 的大小与输入图像相同
                grid = F.affine_grid(theta, im_mov_4d.size(), align_corners=True)
                # 使用 grid_sample 进行插值采样（mode 可选 bilinear 或 nearest）
                warped = F.grid_sample(im_mov_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                # 去除 batch 及 channel 维度，得到 shape (Ny, Nx)
                warped_views.append(warped.squeeze(0).squeeze(0))
            
            # 将所有视角重建结果叠加后取平均：shape (Ny, Nx)
            warped_stack = torch.stack(warped_views, dim=0)
            ss_depth = torch.mean(warped_stack, dim=0)
            # 将当前深度的结果赋值到 ssResult_sample 中
            ssResult_sample[:, :, j] = ss_depth

        # Welford 算法在线更新
        count += 1
        if count == 1:
            mean_ssResult = ssResult_sample
            M2 = torch.zeros_like(ssResult_sample)
        else:
            delta = ssResult_sample - mean_ssResult
            mean_ssResult = mean_ssResult + delta / count
            delta2 = ssResult_sample - mean_ssResult
            M2 = M2 + delta * delta2

        print(f"Processing sample {s+1}/{nSamples} ...", end='\r')

    # 计算样本方差（无偏估计）
    var_recon = M2 / (count - 1) if count > 1 else torch.zeros_like(M2)

    # 返回结果（转回 CPU 的 numpy 数组，形状均为 (Ny, Nx, nDepths)）
    return mean_ssResult.cpu().numpy(), var_recon.cpu().numpy()

def shift_and_sum(measurements, backward_list):
    """
    使用 scikit-image 进行 shift-and-sum 重建，并通过 Welford 算法
    在线计算重建结果的均值和方差（CPU 版本）。

    参数:
    ----------
    measurements : np.ndarray
        形状 (Ny, Nx, nSamples, nViews) 的 4D 数组 (CPU上的 NumPy 数组)。
    backward_list : np.ndarray
        形状 (nViews, nDepths, 3, 3) 的 4D 数组，每个 (3,3) 矩阵为仿射变换，
        若是正向变换，需要取其逆

    返回:
    ----------
    mean_recon : np.ndarray
        shift-and-sum 重建结果在所有样本上的平均值 (形状：Ny, Nx, nDepths)，CPU 上。
    var_recon : np.ndarray
        shift-and-sum 重建结果在所有样本上的方差 (形状：Ny, Nx, nDepths)，CPU 上。
    """

    Ny, Nx, nSamples, nViews = measurements.shape
    _, nDepths, _, _ = backward_list.shape  # backward_list.shape == (nViews, nDepths, 3, 3)

    # 用于在线更新均值和方差
    mean_ssResult = np.zeros((Ny, Nx, nDepths), dtype=measurements.dtype)
    M2 = np.zeros((Ny, Nx, nDepths), dtype=measurements.dtype)
    count = 0

    # ========== 主循环：遍历每个样本，计算 shift-and-sum，并更新均值方差 ==========
    for s in range(nSamples):
        # 取第 s 个样本: shape=(Ny, Nx, nViews)
        sample_data = measurements[:, :, s, :]

        # 存储本次样本的 3D 重建结果 (Ny, Nx, nDepths)
        ssResult_sample = np.zeros((Ny, Nx, nDepths), dtype=measurements.dtype)

        # 对每个深度 j 做 shift-and-sum
        for j in range(nDepths):
            # 临时累加图像阵列：形状 (Ny, Nx, nViews)
            im_sum = np.zeros((Ny, Nx, nViews), dtype=measurements.dtype)

            for i in range(nViews):
                # 取出当前视图
                im_mov = sample_data[:, :, i]  # shape=(Ny, Nx)

                # 取出正向变换矩阵 (3x3) (若已是逆矩阵，则直接使用)
                A = backward_list[i, j, :, :]
                # 用 scikit-image AffineTransform + warp
                transform = AffineTransform(matrix=A).inverse
                # 注意：warp 需要的是 "inverse_map" 参数，这里放 transform 即可
                warped = warp(
                    im_mov,
                    inverse_map=transform,
                    output_shape=(Ny, Nx),
                    mode='constant',
                    cval=0.0,
                    preserve_range=True
                )
                im_sum[:, :, i] = warped

            # 对 nViews 张图像取平均
            ssResult_sample[:, :, j] = np.mean(im_sum, axis=2)

        # ========== 用 Welford 算法在线更新全局均值/方差 ==========
        count += 1
        if count == 1:
            mean_ssResult = ssResult_sample
            M2 = np.zeros_like(ssResult_sample)
        else:
            delta = ssResult_sample - mean_ssResult
            mean_ssResult += delta / count
            delta2 = ssResult_sample - mean_ssResult
            M2 += delta * delta2

        # 可能需要进度提示
        print(f"Processing sample {s+1}/{nSamples} ...", end='\r')

    # 计算样本方差 (无偏估计)
    var_recon = M2 / (count - 1) if count > 1 else np.zeros_like(M2)

    return mean_ssResult, var_recon

# def shift_and_sum_gpu(measurements, backward_list):
#     """
#     使用 cuCIM 在 GPU 上完成 shift-and-sum 重建，并通过 Welford 算法
#     在线计算重建结果的均值和方差。

#     参数:
#     ----------
#     measurements : np.ndarray or cp.ndarray
#         形状 (Ny, Nx, nSamples, nViews) 的 4D 数组 (CPU 或 GPU 上均可)。
#     backward_list : np.ndarray or cp.ndarray
#         形状 (nViews, nDepths, 3, 3) 的 4D 数组，每个 (3,3) 正向仿射矩阵。

#     返回:
#     ----------
#     mean_recon : cp.ndarray
#         shift-and-sum 重建结果在所有样本上的平均值 (形状：Ny, Nx, nDepths)，GPU 上。
#     var_recon : cp.ndarray
#         shift-and-sum 重建结果在所有样本上的方差 (形状：Ny, Nx, nDepths)，GPU 上。
#     """

#     # 如果 measurements 在 CPU 上，用 cp.asarray() 转到 GPU
#     measurements_gpu = cp.asarray(measurements)

#     Ny, Nx, nSamples, nViews = measurements_gpu.shape
#     _, nDepths, _, _ = backward_list.shape  # backward_list 的形状 (nViews, nDepths, 3, 3)

#     # 用于在线更新均值和方差
#     mean_ssResult = cp.zeros((Ny, Nx, nDepths), dtype=measurements_gpu.dtype)
#     M2 = cp.zeros((Ny, Nx, nDepths), dtype=measurements_gpu.dtype)
#     count = 0

#     # ========== 主循环：遍历每个样本，计算 shift-and-sum，并更新均值方差 ==========
#     for s in range(nSamples):
#         # 取第 s 个样本: shape=(Ny, Nx, nViews)
#         # 这里等效 MATLAB: sample_data = squeeze(measurements_gpu(:,:,s,:));
#         sample_data = measurements_gpu[:, :, s, :]

#         # 重建结果 (Ny, Nx, nDepths)
#         ssResult_sample = cp.zeros((Ny, Nx, nDepths), dtype=measurements_gpu.dtype)

#         # 对每个深度平面 j 做 shift-and-sum
#         for j in range(nDepths):
#             # 临时累积图像：形状 (Ny, Nx, nViews)
#             im_sum = cp.zeros((Ny, Nx, nViews), dtype=measurements_gpu.dtype)

#             for i in range(nViews):
#                 # 取出当前视图
#                 im_mov = sample_data[:, :, i]  # shape=(Ny, Nx)

#                 # 取出正向变换矩阵 (3x3)
#                 A = backward_list[i, j, :, :]
#                 # 若 A 是正向变换矩阵，需要做逆变换
#                 tform = transform.AffineTransform(matrix=A).inverse
#                 # 使用 cuCIM 进行仿射变换
#                 warped = transform.warp(
#                     im_mov,
#                     tform,
#                     output_shape=(Ny, Nx),
#                     mode='constant',
#                     cval=0.0,
#                     preserve_range=True
#                 )

#                 # 存入 im_sum
#                 im_sum[:, :, i] = warped

#             # 对 nViews 张图像取平均
#             ssResult_sample[:, :, j] = cp.mean(im_sum, axis=2)

#         # ========== 用 Welford 算法在线更新全局均值/方差 ==========
#         count += 1
#         if count == 1:
#             mean_ssResult = ssResult_sample
#             M2 = cp.zeros_like(ssResult_sample)
#         else:
#             delta = ssResult_sample - mean_ssResult
#             mean_ssResult += delta / count
#             delta2 = ssResult_sample - mean_ssResult
#             M2 += delta * delta2

#         # 可能需显示进度条，此处省略
#         # print(f"Processing sample {s+1}/{nSamples} ...", end='\r')

#     # 样本方差 (无偏估计)
#     if count > 1:
#         var_recon = M2 / (count - 1)
#     else:
#         var_recon = cp.zeros_like(M2)

#     mean_recon = mean_ssResult

#     return mean_recon, var_recon

def pad(a, axes=(0, 1)):
    """
    对输入数组 a 的指定若干维度（axes）做 pad 操作，
    在每个指定维度两侧填充 (dimension // 2) 个 0，
    其余维度不变。

    参数:
      a : np.ndarray
          输入数组。
      axes : tuple or list
          要进行 pad 的维度编号 (axis)。可以包含一个或多个维度编号。

    返回:
      padded : np.ndarray
          pad 之后的数组。在所有指定的 axis 上，每个维度长度将变为
          原长度 + 2 * (原长度 // 2)，其余维度不做改动。
    """
    # pad_width 是一个与 a.ndim 等长的列表，每个元素都是 (0, 0)。
    pad_width = [(0, 0)] * a.ndim

    # 对每个指定的 axis 执行填充
    for ax in axes:
        size_ax = a.shape[ax]
        half_ax = size_ax // 2
        pad_width[ax] = (half_ax, half_ax)

    padded = np.pad(a, pad_width, mode='constant', constant_values=0)
    return padded

def unpad(padded, original_shape, axes=(0, 1)):
    """
    对已经 pad 过的数组 padded（指定轴上的 pad 量为原长度 // 2），
    从对应维度上提取出中央区域，从而取消 pad 的影响。

    参数:
      padded : np.ndarray
          pad 后的数组，其在 axes 中指定的每个维度上，被额外填充了原长度 // 2 的 0。
      original_shape : tuple or list
          原始数组的形状 (至少要与 padded 的维度数量相同)。
      axes : tuple or list
          指定需要 unpad 的维度编号。
    
    返回:
      unpadded : np.ndarray
          取消 pad 后的数组，其形状与 original_shape 相同。
    """
    # 我们按照 original_shape 中对应 axis 的大小，计算要去除的 slice。
    # 对于未在 axes 中的维度，我们保留所有元素 (slice(None))
    slices = [slice(None)] * padded.ndim  # 先对所有维度设为全选

    for ax in axes:
        orig_size = original_shape[ax]
        pad_amt = orig_size // 2
        start_index = pad_amt
        end_index = pad_amt + orig_size
        slices[ax] = slice(start_index, end_index)

    unpadded = padded[tuple(slices)]
    return unpadded

def psf_pad(a, target_shape, axes=(0, 1)):
    """
    对输入数组 a 在指定的若干维度 (axes) 上进行 PSF pad 操作，
    将其 pad 到在这些维度上的目标尺寸 target_shape，
    保证数据居中，其它未指定的维度不变。

    参数:
      a : np.ndarray
          输入数组。
      target_shape : list or tuple
          指定在 axes 中对应的目标尺寸。例如若 axes=(0,2) 且 target_shape=(H_new, D_new)，
          则表示 a 的第 0 维要 pad 到 H_new、第 2 维要 pad 到 D_new。
      axes : list or tuple
          指定要进行 pad 的维度编号 (axis)，其长度需要与 target_shape 相同。

    返回:
      padded : np.ndarray
          pad 后的数组：在 axes 中每个维度的长度与 target_shape 中相对应的目标尺寸相同；
          其它维度保持原状。
    """
    if len(axes) != len(target_shape):
        raise ValueError("axes 与 target_shape 的长度必须一致。")

    # 构造 pad_width，默认对所有维度不做 pad
    pad_width = [(0, 0)] * a.ndim

    # 遍历指定 axes，计算需要填充的数量
    for ax, tgt_dim in zip(axes, target_shape):
        orig_dim = a.shape[ax]
        if tgt_dim < orig_dim:
            raise ValueError(f"目标尺寸必须不小于原始尺寸: 原始={orig_dim}, 目标={tgt_dim}")

        left_pad = (tgt_dim - orig_dim) // 2
        right_pad = tgt_dim - orig_dim - left_pad
        pad_width[ax] = (left_pad, right_pad)

    padded = np.pad(a, pad_width, mode='constant', constant_values=0)
    return padded


def ellipse_selector_with_sliders(image,center=None,size=None):
    """使用滑动条控制椭圆参数"""
    # 创建假彩色图像
    image = np.stack([image[:,:,1], image[:,:,-1], np.zeros_like(image[:,:,1])], axis=2)

    # 获取图像尺寸
    height, width = image.shape[:2]

    if center is None:
        center = [height//2, width//2]
    if size is None:
        size = [3*height//4, 3*width//4]
    
    # 创建滑动条控件
    center_x = widgets.IntSlider(value=center[0], min=100, max=width-100, 
                                description='Center X:', continuous_update=False)
    center_y = widgets.IntSlider(value=center[1], min=100, max=height-100, 
                                description='Center Y:', continuous_update=False)
    width_slider = widgets.IntSlider(value=size[0], min=100, max=width, 
                                    description='Width:', continuous_update=False)
    height_slider = widgets.IntSlider(value=size[1], min=100, max=height, 
                                    description='Height:', continuous_update=False)
    angle_slider = widgets.IntSlider(value=0, min=0, max=180, 
                                    description='Angle:', continuous_update=False)
    sigma_slider = widgets.IntSlider(value=30, min=1, max=100, 
                                    description='Gaussian Filter σ:', continuous_update=False)
    
    # 创建输出区域
    output_display = widgets.Output()
    
    # 存储最终结果
    result = {'mask': None}
    
    # 更新图像和椭圆显示
    def update_display(*args):
        # 创建掩码
        mask = create_ellipse_mask(
            (center_x.value, center_y.value),
            width_slider.value/2,
            height_slider.value/2,
            np.deg2rad(angle_slider.value),
            (height, width)
        )

        # 应用高斯滤波
        filtered_mask = gaussian_filter(mask, sigma=sigma_slider.value)
        result['mask'] = filtered_mask
        
        with output_display:
            # 清除之前的输出
            output_display.clear_output(wait=True)
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # 显示原图和椭圆
            ax1.imshow(image)
            ellipse = Ellipse((center_x.value, center_y.value), 
                            width_slider.value, height_slider.value, 
                            angle=angle_slider.value, 
                            fill=False, edgecolor='yellow', linewidth=2)
            ax1.add_patch(ellipse)
            ax1.set_title('Params')
            
            # 显示滤波后的掩码
            mask_img = ax2.imshow(filtered_mask, cmap='gray')
            ax2.set_title(f'Filtered mask (sigma={sigma_slider.value})')
            fig.colorbar(mask_img, ax=ax2)
            
            plt.tight_layout()
            plt.show()

    # 创建确认按钮
    confirm_button = widgets.Button(description='Confirm Selection')
    
    def on_confirm(b):
        update_display()
        print("Selection confirmed! You can now access the mask.")
    
    confirm_button.on_click(on_confirm)
    
    # 绑定事件处理函数
    center_x.observe(update_display, 'value')
    center_y.observe(update_display, 'value')
    width_slider.observe(update_display, 'value')
    height_slider.observe(update_display, 'value')
    angle_slider.observe(update_display, 'value')
    sigma_slider.observe(update_display, 'value')
    
    # 创建布局
    controls = widgets.VBox([
        widgets.HBox([center_x, center_y]),
        widgets.HBox([width_slider, height_slider]),
        angle_slider,
        sigma_slider,
        confirm_button
    ])
    
    # 布局整个UI
    ui = widgets.VBox([
        controls,
        output_display
    ])
    
    # 显示UI
    display(ui)
    
    # 初始显示
    update_display()
    
    return result['mask']

def create_ellipse_mask(center, a, b, angle, shape):
    """根据椭圆参数创建掩码"""
    mask = np.zeros(shape, dtype=np.float64)
    height, width = shape
    
    # 创建坐标网格
    y, x = np.ogrid[:height, :width]
    
    # 计算椭圆方程
    # 步骤1: 中心化坐标
    xc = x - center[0]
    yc = y - center[1]
    
    # 步骤2: 旋转坐标
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    xct = xc * cos_angle + yc * sin_angle
    yct = -xc * sin_angle + yc * cos_angle
    
    # 步骤3: 椭圆方程
    rad_eq = (xct**2 / a**2) + (yct**2 / b**2)
    
    # 步骤4: 生成掩码
    mask[rad_eq <= 1.0] = 1.0
    
    return mask