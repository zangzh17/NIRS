#!/usr/bin/env python3
"""
简化的DeepCAD训练接口
用于MATLAB调用，只需要输入训练数据路径、模型名称和项目路径
"""
import sys
import os

# 添加DeepCAD项目路径到Python路径
if deepcad_project_path not in sys.path:
    sys.path.insert(0, deepcad_project_path)

# 切换到DeepCAD项目目录
original_dir = os.getcwd()
os.chdir(deepcad_project_path)

try:
    from deepcad.train_collection import training_class
    from deepcad.utils import get_first_filename
    
    # 检查训练数据路径是否存在
    if not os.path.exists(training_data_path):
        raise ValueError(f"Training path invalid: {training_data_path}")
    
    # 检查训练数据路径中是否有tif文件
    tif_files = [f for f in os.listdir(training_data_path) if f.lower().endswith(('.tif', '.tiff'))]
    if not tif_files:
        raise ValueError(f"Training TIF images invalid: {training_data_path}")
    
    # 处理路径格式兼容性（Windows使用反斜杠，DeepCAD内部期望正斜杠）
    datasets_path_normalized = training_data_path.replace('\\', '/')

    # 构建训练参数字典
    train_dict = {
        # dataset dependent parameters
        'patch_x': int(patch_xy),
        'patch_y': int(patch_xy),
        'patch_t': int(patch_t),
        'overlap_factor': overlap_factor,
        'scale_factor': scale_factor,
        'select_img_num': select_img_num,
        'train_datasets_size': train_datasets_size,
        'datasets_path': datasets_path_normalized,
        'pth_dir': './pth',
        
        # network related parameters
        'n_epochs': int(n_epochs),
        'lr': lr,
        'b1': b1,
        'b2': b2,
        'fmap': int(fmap),
        'GPU': GPU,
        'num_workers': num_workers,
        'visualize_images_per_epoch': visualize_images_per_epoch,
        'save_test_images_per_epoch': save_test_images_per_epoch,
        
    }
    
    print("Training param:")
    for key, value in train_dict.items():
        print(f"  {key}: {value}")
    
    # 创建训练类对象
    tc = training_class(train_dict)
    
    # 开始训练
    tc.run()
    
    
except Exception as e:
    print(f"Error during training: {str(e)}")
    import traceback
    traceback.print_exc()
    raise
    
finally:
    # 恢复原始工作目录
    os.chdir(original_dir)