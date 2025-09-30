#!/usr/bin/env python3
"""
简化的DeepCAD降噪接口
用于MATLAB调用，只需要输入tif路径、模型名称和项目路径
"""

import sys
import os

# 添加DeepCAD项目路径到Python路径
if deepcad_project_path not in sys.path:
    sys.path.insert(0, deepcad_project_path)

# 切换到DeepCAD项目目录
original_dir = os.getcwd()
os.chdir(deepcad_project_path)

from deepcad.test_collection import testing_class

# 创建包含输入tif文件的临时数据集目录

# 简化的测试参数，使用默认值
test_dict = {
    # dataset dependent parameters
    'patch_x': 150,                     # the width of 3D patches
    'patch_y': 150,                     # the height of 3D patches  
    'patch_t': 150,                     # the time dimension (frames) of 3D patches
    'overlap_factor': 0.4,              # overlap factor
    'scale_factor': 1,                  # the factor for image intensity scaling
    'test_datasize': 100000,               # the number of frames to be tested
    'datasets_path': input_tif_path,      # folder containing input tif file
    'pth_dir': 'pth',     # pth file root path
    'denoise_model': denoise_model,     # model name
    'output_dir': output_dir,  # result file root path
    # network related parameters
    'fmap': 16,                         # number of feature maps
    'GPU': '0',                         # GPU index
    'num_workers': 0,                   # Windows compatible
    'visualize_images_per_epoch': False, # no visualization
    'save_test_images_per_epoch': True  # save results
}

# 运行降噪
tc = testing_class(test_dict)
tc.run()
os.chdir(original_dir)