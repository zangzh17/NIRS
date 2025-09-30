# 文件路径: python_utils/measurement_visualization.py

import numpy as np
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, IntSlider, FloatRangeSlider


def visualize_views(measurements):
    """
    交互式可视化 measurements 数据，支持对样本维度进行统计计算，
    并通过滑动条选择图像显示的裁剪范围（对比度调整）。
    
    参数:
        measurements: numpy.ndarray
            形状为 (640, 640, 100, 5)，其中：
              - 第三个维度 (axis=2) 表示 100 个样本，
              - 第四个维度 (axis=3) 代表 5 个子视角，需要同时显示。
    
    该函数提供两个交互控件:
      1. Statistic: 选择计算方式 (Mean: 求均值; Std: 先求方差再开方即标准差)
      2. Clip Range: 通过滑动条设定图像的显示范围 (vmin, vmax) 用于调整对比度
    """
    def update_visualization(statistic='Mean', clip_range=(0.0, 1.0)):
        # clip_range 解构为最小值和最大值
        clip_min, clip_max = clip_range

        if statistic == 'Mean':
            # 在 samples 维度 (axis=2) 上求均值，得到 shape 为 (640, 640, 5)
            combined = rescale_intensity(np.mean(measurements, axis=2),out_range=(0, 1.0))
            title_suffix = 'Mean Image Across Samples'
        elif statistic == 'Std':
            # 先求方差再开方，得到标准差图像
            combined = rescale_intensity(np.sqrt(np.var(measurements, axis=2)),out_range=(0, 1.0))
            title_suffix = 'Standard Deviation Image Across Samples'
        else:
            raise ValueError("Statistic 必须为 'Mean' 或 'Std'。")
        
        # 创建一行 5 个子图，每个子图对应一个子视角
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        for idx, ax in enumerate(axs):
            # 将图像显示调节对比度，通过 vmin 和 vmax
            ax.imshow(combined[..., idx], cmap='gray', vmin=clip_min, vmax=clip_max)
            ax.set_title(f'Sub-Aperture {idx+1}')
            ax.axis('off')
        fig.suptitle(title_suffix, fontsize=16)
        plt.show()

    # 定义交互式控件：
    statistic_picker = widgets.Dropdown(
        options=['Mean', 'Std'],
        value='Mean',
        description='Statistic'
    )
    
    clip_range_picker = widgets.FloatRangeSlider(
        value=[0.0, 1.0],
        min=0.0,
        max=1.0,
        step=0.01,
        description='Clip Range',
        continuous_update=False
    )
    
    interact(update_visualization,
             statistic=statistic_picker,
             clip_range=clip_range_picker)

def interactive_recon_viewer(avgRecon1, avgRecon2=None):
    """
    对于输入的一个或两个相同尺寸的三维数据 (Ny, Nx, nDepths)：
      - 分别归一化到 [0,1]
      - 通过滑动条调节 depth 和 clip_range 进行可视化
      - 如果提供了两个数据，通过下拉菜单选择可视化模式（仅显示Data1、仅显示Data2或并排显示二者）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from ipywidgets import interact, IntSlider, FloatRangeSlider, Dropdown, Layout

    # ============ 步骤1：归一化 ============
    # 数据1归一化
    global_min1 = float(np.nanmin(avgRecon1))
    global_max1 = float(np.nanmax(avgRecon1))
    denom1 = (global_max1 - global_min1) if global_max1 > global_min1 else 1e-9
    data1 = (avgRecon1 - global_min1) / denom1

    # 数据2归一化（如果提供）
    if avgRecon2 is not None:
        global_min2 = float(np.nanmin(avgRecon2))
        global_max2 = float(np.nanmax(avgRecon2))
        denom2 = (global_max2 - global_min2) if global_max2 > global_min2 else 1e-9
        data2 = (avgRecon2 - global_min2) / denom2
    else:
        data2 = None

    # 假设输入数据维度一致，取数据1的尺寸 (Ny, Nx, nDepths)
    Ny, Nx, nDepths = data1.shape

    # ============ 步骤2：创建控件 ============
    # Depth滑动条：选择切片深度
    depth_slider = IntSlider(
        value=0,
        min=0,
        max=nDepths - 1,
        step=1,
        description='Depth:',
        continuous_update=False
    )

    # Clip Range滑动条：选择剪裁范围
    clip_slider = FloatRangeSlider(
        value=[0.0, 1.0],
        min=0.0,
        max=1.0,
        step=0.01,
        description='Clip Range:',
        continuous_update=False,
        layout=Layout(width='75%')
    )

    # 下拉菜单：选择显示模式
    # 如果只提供一个数据，则仅显示Data1；否则可选Data1、Data2、Both
    if data2 is not None:
        mode_dropdown = Dropdown(
            options=[('Data1', 'data1'), ('Data2', 'data2'), ('Both', 'both')],
            value='data1',
            description='Data:'
        )
    else:
        mode_dropdown = Dropdown(
            options=[('Data1', 'data1')],
            value='data1',
            description='Data:'
        )

    # ============ 步骤3：定义绘图函数 ============
    def update_plot(depth, clip_range, mode):
        """
        depth      : 当前深度索引
        clip_range : (vmin, vmax)，用于截取数据后映射到 [0,1]
        mode       : 显示模式，'data1'仅显示数据1，'data2'仅显示数据2，'both'并排显示二者
        """
        vmin, vmax = clip_range
        # 为防止除零
        denom = (vmax - vmin) if vmax > vmin else 1e-9

        if mode == 'data1':
            # 取数据1对应切片，并根据clip_range截取后映射
            slice1 = data1[:, :, depth]
            slice1_norm = (np.clip(slice1, vmin, vmax) - vmin) / denom

            plt.figure(figsize=(5, 4))
            plt.imshow(slice1_norm, cmap='gray', origin='upper', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f"Data1: Depth = {depth}, Clip = {vmin:.3f}~{vmax:.3f}")
            plt.show()

        elif mode == 'data2':
            # 当仅显示数据2时，确保数据2存在
            if data2 is None:
                return
            slice2 = data2[:, :, depth]
            slice2_norm = (np.clip(slice2, vmin, vmax) - vmin) / denom

            plt.figure(figsize=(5, 4))
            plt.imshow(slice2_norm, cmap='gray', origin='upper', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f"Data2: Depth = {depth}, Clip = {vmin:.3f}~{vmax:.3f}")
            plt.show()

        elif mode == 'both':
            # 并排显示数据1和数据2
            if data2 is None:
                return
            slice1 = data1[:, :, depth]
            slice2 = data2[:, :, depth]
            slice1_norm = (np.clip(slice1, vmin, vmax) - vmin) / denom
            slice2_norm = (np.clip(slice2, vmin, vmax) - vmin) / denom

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(slice1_norm, cmap='gray', origin='upper', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f"Data1: Depth = {depth}, Clip = {vmin:.3f}~{vmax:.3f}")

            plt.subplot(1, 2, 2)
            plt.imshow(slice2_norm, cmap='gray', origin='upper', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f"Data2: Depth = {depth}, Clip = {vmin:.3f}~{vmax:.3f}")
            plt.show()

    # ============ 步骤4：使用 interact 联动 ============
    interact(
        update_plot,
        depth=depth_slider,
        clip_range=clip_slider,
        mode=mode_dropdown
    )

def plot_psf_slices(PSF_syn, x_index=None, y_index=None):
    """
    绘制 3D PSF_syn 在指定 x_index 上的 YZ 切片，以及在指定 y_index 上的 XZ 切片。

    参数:
      PSF_syn : np.ndarray
          形状为 (Y, X, Z) 的三维数组。
      x_index : int or None
          用于绘制 YZ slice 的 x 索引。若为 None，则默认取中间值。
      y_index : int or None
          用于绘制 XZ slice 的 y 索引。若为 None，则默认取中间值。
    """
    # 获取 Z, Y, X 三个维度的大小
    Y, X, Z = PSF_syn.shape

    # 若用户未指定 x_index 或 y_index，则取中间切片
    if x_index is None:
        x_index = X // 2
    if y_index is None:
        y_index = Y // 2

    # ========== YZ slice ==========
    # 固定某个 x 索引，只保留 (Z, Y) 两个维度
    yz_slice = PSF_syn[:, x_index, :]  

    # ========== XZ slice ==========
    # 固定某个 y 索引，只保留 (Z, X) 两个维度
    xz_slice = PSF_syn[y_index, :, :]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 绘制 YZ slice
    ax1 = axes[0]
    im1 = ax1.imshow(yz_slice, cmap='viridis', origin='lower', aspect='auto')
    ax1.set_title(f'YZ slice (x = {x_index})')
    ax1.set_xlabel('Z axis')
    ax1.set_ylabel('Y axis')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 绘制 XZ slice
    ax2 = axes[1]
    im2 = ax2.imshow(xz_slice, cmap='viridis', origin='lower', aspect='auto')
    ax2.set_title(f'XZ slice (y = {y_index})')
    ax2.set_xlabel('Z axis')
    ax2.set_ylabel('X axis')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()