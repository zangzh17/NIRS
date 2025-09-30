import numpy as np
import cupy as cp
import pylops
import time

def deconvlucy(image, psf, num_iter=10, dampar=None, weight=None, readout=None, verbose=True):
    """
    使用Lucy-Richardson方法对图像进行解卷积，基于MATLAB的deconvlucy实现
    
    参数:
    -----------
    image : cp.ndarray
        需要解卷积的模糊图像 (N维)
    psf : cp.ndarray
        点扩散函数(卷积核)
    num_iter : int, 可选
        迭代次数 (默认: 10)
    dampar : float 或 cp.ndarray, 可选
        抑制平滑区域噪声的阻尼参数 (默认: 0)
    weight : cp.ndarray, 可选
        根据flat-field校正分配给每个像素的权重值 (默认: 全1)
    readout : float 或 cp.ndarray, 可选
        加性噪声(背景噪声，相机读出噪声) (默认: 0)
    verbose : bool, 可选
        是否打印进度信息 (默认: True)
        
    返回:
    --------
    deconvolved : cp.ndarray
        解卷积后的图像
    """
    # 确保输入是CuPy数组
    if not isinstance(image, cp.ndarray):
        image = cp.asarray(image)
    
    if not isinstance(psf, cp.ndarray):
        psf = cp.asarray(psf)
    
    # 初始化默认参数
    if dampar is None:
        dampar = 0.0
    
    if readout is None:
        readout = 0.0
    
    # 需要时将标量转换为数组
    if cp.isscalar(dampar):
        dampar = cp.full_like(image, dampar).ravel()
        
    if cp.isscalar(readout):
        readout = cp.full_like(image, readout).ravel()
    
    # 如果未提供权重，则创建权重(平场校正)
    if weight is None:
        weight = cp.ones_like(image).ravel()
    elif cp.isscalar(weight):
        weight = cp.full_like(image, weight).ravel()
    else:
        weight = weight.ravel()
    
    # 使用PyLops创建卷积运算符
    # 计算PSF偏移以进行适当的居中(假设PSF中心在中间)
    offset = tuple(s // 2 for s in psf.shape)
    
    # 创建卷积轴(-1表示最后一个维度，-2表示倒数第二个维度，以此类推)
    ndim = len(image.shape)
    axes = tuple(range(-ndim, 0))
    
    # 创建N维卷积运算符
    Cop = pylops.signalprocessing.ConvolveND(
        image.shape, h=psf, offset=offset, axes=axes, method='fft'
    )
    
    # 初始化迭代变量(遵循MATLAB实现)
    # J[0] = 原始图像
    # J[1] = 当前估计
    # J[2] = 上一个估计
    # J[3] = 用于加速的差向量
    J = [None] * 4
    J[0] = image
    J[1] = cp.full(image.shape, 0.5, dtype=image.dtype).ravel()  # 初始估计
    J[2] = cp.zeros_like(J[1]).ravel()  # 上一次估计
    J[3] = cp.zeros((image.size, 2)).ravel()  # 用于加速
    
    # 计算加权图像(带正值约束)
    # 这里应用任何需要的读出校正
    wI = cp.maximum(weight * (readout + J[0]), 0)
    
    # 计算归一化比例因子
    # 相当于将权重与PSF转置进行卷积
    scale = Cop.rmatvec(weight) + cp.sqrt(cp.finfo(float).eps)
    
    # 计算阻尼参数平方/2(为了计算效率)
    DAMPAR22 = (dampar**2) / 2
    
    # 主Lucy-Richardson迭代
    lambda_accel = 0  # 加速因子
    start_time = time.time()
    
    for k in range(num_iter):
        if verbose:
            # 计算估计剩余时间
            iter_time = time.time() - start_time
            eta = (iter_time / (k+1)) * (num_iter - k - 1) if k > 0 else 0
            print(f"RL Iter {k+1}/{num_iter} - Remaining: {eta:.1f}s", end='\r')
            # 确保GPU操作在报告时间之前完成
            cp.cuda.Stream.null.synchronize()
        
        # 1. 计算加速因子(第二次迭代之后)
        # 使用Biggs & Andrews 1997中的方法
        if k > 1:
            # 使用向量内积计算最优lambda
            num = cp.sum(J[3][:, 0] * J[3][:, 1])
            denom = cp.sum(J[3][:, 1]**2) + cp.finfo(float).eps
            lambda_accel = num / denom
            lambda_accel = cp.clip(lambda_accel, 0, 1)  # 确保稳定性
        
        # 2. 用加速方法预测下一次迭代
        Y = cp.maximum(J[1] + lambda_accel * (J[1] - J[2]), 0)  # 带正值约束
        
        # 3. 前向模糊(应用PSF卷积)
        ReBlurred = Cop.matvec(Y)
        
        # 4. 添加读出噪声并确保正值
        ReBlurred = ReBlurred + readout
        ReBlurred = cp.maximum(ReBlurred, cp.finfo(float).eps)  # 避免除以零
        
        # 5. 计算测量值与估计图像的比率
        ratio = wI / ReBlurred
        
        # 6. 如需要应用阻尼(抑制噪声放大)
        if cp.all(DAMPAR22 == 0):
            # 无阻尼 - 标准Lucy-Richardson
            ImRatio = ratio
        else:
            # 在平滑区域应用阻尼以抑制噪声
            gm = 10  # 阻尼因子
            # 计算相对偏差度量
            g = (wI * cp.log(ratio + cp.finfo(float).eps) + ReBlurred - wI) / DAMPAR22
            g = cp.minimum(g, 1)  # 限制最大偏差
            
            # 应用阻尼函数
            G = (g**(gm-1)) * (gm - (gm-1)*g)
            ImRatio = 1 + G * (ratio - 1)
        
        # 7. 通过应用PSF的转置计算校正
        correction = Cop.rmatvec(ImRatio)
        
        # 8. 更新估计
        J[2] = J[1].copy()  # 存储上一个估计
        J[1] = cp.maximum(Y * correction / scale, 0)  # 带正值约束的新估计
        
        # 9. 存储差异用于加速
        J[3] = cp.column_stack([
            cp.ravel(J[1] - Y),
            J[3][:, 0]
        ])
    
    if verbose:
        total_time = time.time() - start_time
        print(f"\nRL finished. Total time: {total_time:.2f}s")
    
    # 返回最终解卷积图像
    return J[1].reshape(Cop.dims).get()