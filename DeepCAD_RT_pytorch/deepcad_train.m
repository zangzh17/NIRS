function denoised_data = deepcad_train(input_data, deepcad_path, varargin)
% DEEPCAD_TRAIN 使用DeepCAD训练降噪模型
%
% 输入:
% training_data_path - 训练数据路径 (包含tif文件的文件夹)
% deepcad_path - DeepCAD项目根路径
% varargin - 可选参数对，例如:
%   'model_name', 'my_model_20250617' - 模型名称
%   'n_epochs', 50 - 训练轮数
%   'patch_xy', 200 - xy方向patch大小
%   'patch_t', 150 - 时间方向patch大小
%   'train_datasets_size', 6000 - 训练数据集大小
%   'lr', 0.00005 - 学习率
%   'GPU', '0' - GPU索引
%   'overlap_factor', 0.25 - patch重叠因子
%   'visualize_images_per_epoch', true - 是否每轮可视化
%   'save_test_images_per_epoch', true - 是否每轮保存测试图像
%
% 输出:
% model_name - 训练完成的模型名称

% 解析输入参数
p = inputParser;
addRequired(p, 'deepcad_path', @ischar);

% 添加可选参数
addParameter(p, 'model_name', 'tmp', @ischar);
addParameter(p, 'n_epochs', 10, @isnumeric);
addParameter(p, 'patch_xy', 150, @isnumeric);
addParameter(p, 'patch_t', 150, @isnumeric);
addParameter(p, 'train_datasets_size', 3000, @isnumeric);
addParameter(p, 'lr', 0.00002, @isnumeric);
addParameter(p, 'GPU', '0', @ischar);
addParameter(p, 'overlap_factor', 0.4, @isnumeric);
addParameter(p, 'visualize_images_per_epoch', false, @islogical);
addParameter(p, 'save_test_images_per_epoch', true, @islogical);
addParameter(p, 'num_workers', 0, @isnumeric); % Windows兼容，默认为0
addParameter(p, 'fmap', 16, @isnumeric);
addParameter(p, 'b1', 0.5, @isnumeric);
addParameter(p, 'b2', 0.999, @isnumeric);
addParameter(p, 'scale_factor', 1, @isnumeric);
addParameter(p, 'select_img_num', 1000000, @isnumeric);
parse(p, deepcad_path, varargin{:});

% 检查训练数据路径是否存在
% 转换为uint16以节省空间
if ~isa(input_data, 'uint16')
    input_data = uint16(65535*rescale(input_data));
end
% 创建临时目录
training_data_path = fullfile(deepcad_path,'datasets', p.Results.model_name);
if ~exist(training_data_path, 'dir')
    mkdir(training_data_path);
end
% 临时输入文件路径
temp_input_file = fullfile(training_data_path, 'input.tif');
fprintf('正在保存临时Tiff文件...\n');
save_3d_tiff(input_data, temp_input_file);


try
    % 调用Python训练脚本
    pyrunfile(fullfile(deepcad_path, 'train_interface.py'), ...
        training_data_path=training_data_path, ...
        deepcad_project_path=deepcad_path, ...
        n_epochs=p.Results.n_epochs, ...
        patch_xy=p.Results.patch_xy, ...
        patch_t=p.Results.patch_t, ...
        train_datasets_size=p.Results.train_datasets_size, ...
        lr=p.Results.lr, ...
        GPU=p.Results.GPU, ...
        overlap_factor=p.Results.overlap_factor, ...
        visualize_images_per_epoch=p.Results.visualize_images_per_epoch, ...
        save_test_images_per_epoch=p.Results.save_test_images_per_epoch, ...
        num_workers=p.Results.num_workers, ...
        fmap=p.Results.fmap, ...
        b1=p.Results.b1, ...
        b2=p.Results.b2, ...
        scale_factor=p.Results.scale_factor, ...
        select_img_num=p.Results.select_img_num);
    % 清理临时文件和输出文件
    delete(temp_input_file);
    if exist(training_data_path, 'dir')
        rmdir(training_data_path, 's');
    end
    %读取
    output_dir = fullfile(deepcad_path,'pth', p.Results.model_name);
    all_pth_files = dir(fullfile(output_dir, '*.pth'));
    all_tif_files = dir(fullfile(output_dir, '*.tif'));
    if ~isempty(all_pth_files)
        % 按日期排序，找到最新的.pth
        [~, pth_idx] = max([all_pth_files.datenum]);
        % 删除其他.pth文件
        for i = 1:length(all_pth_files)
            if i ~= pth_idx
                file_to_delete = fullfile(output_dir, all_pth_files(i).name);
                delete(file_to_delete);
            end
        end
    end
    if ~isempty(all_tif_files)
        % 按日期排序，找到最新的.tif
        [~, tif_idx] = max([all_tif_files.datenum]);
        latest_tif_file = all_tif_files(tif_idx);
        latest_tif_path = fullfile(output_dir, latest_tif_file.name);
        % 删除其他.tif文件
        for i = 1:length(all_tif_files)
            if i ~= tif_idx
                file_to_delete = fullfile(output_dir, all_tif_files(i).name);
                delete(file_to_delete);
            end
        end
    end
    denoised_data = rescale(double(tiffreadVolume(latest_tif_path)));
    
catch ME
    fprintf('Error: %s\n', ME.message);
    if exist(training_data_path, 'dir')
        rmdir(training_data_path, 's');
    end
    rethrow(ME);
end

end

function save_3d_tiff(data, filename)
    % 保存3D数据为TIFF文件
    % 使用ImageJ兼容的格式
    
    [H, W, T] = size(data);
    
    % 创建TIFF对象
    t = Tiff(filename, 'w');
    
    % 设置第一帧的标签
    tagstruct = struct();
    tagstruct.ImageLength = H;
    tagstruct.ImageWidth = W;
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 16;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Software = 'MATLAB';
    tagstruct.ImageDescription = sprintf('ImageJ=1.0\nimages=%d\nslices=%d', T, T);
    
    % 写入所有帧
    for i = 1:T
        t.setTag(tagstruct);
        t.write(data(:,:,i));
        if i < T
            t.writeDirectory();
        end
    end
    
    t.close();
end