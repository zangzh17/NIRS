function [ds, imds, pxds] = buildDataset(img_3d, segmentation_data, output_dir, dataset_name, imageSize, varargin)
% 构建灰度归一化数据集 (增强版 - 包含数据增强功能)
% 输入:
% img_3d - 原始3D图像数据
% segmentation_data - keyframeSegTool的输出结果
% output_dir - 数据集保存目录
% dataset_name - 数据集名称
% imageSize - 目标图像尺寸 [height, width]
% 可选参数 (name-value pairs):
%   'EnableAugmentation' - 是否启用数据增强 (默认: false)
%   'AugmentationFactor' - 每个原始图像生成的增强图像数量 (默认: 4)
%   'RotationRange' - 旋转角度范围 (默认: [-15, 15])
%   'FlipProbability' - 翻转概率 (默认: 0.5)
%   'ScaleRange' - 缩放范围 (默认: [0.9, 1.1])
%   'TranslationRange' - 平移范围像素 (默认: [-10, 10])
%   'BrightnessRange' - 亮度调整范围 (默认: [-0.1, 0.1])
%   'ContrastRange' - 对比度调整范围 (默认: [0.9, 1.1])
%   'NoiseStd' - 高斯噪声标准差 (默认: 0.01)
% 输出:
% ds - 组合的数据集(ImageDatastore + PixelLabelDatastore)
% imds - 图像数据集
% pxds - 像素标签数据集

% 解析输入参数
p = inputParser;
addParameter(p, 'EnableAugmentation', false, @islogical);
addParameter(p, 'AugmentationFactor', 4, @(x) isnumeric(x) && x > 0);
addParameter(p, 'RotationRange', [-15, 15], @(x) isnumeric(x) && length(x) == 2);
addParameter(p, 'FlipProbability', 0.5, @(x) isnumeric(x) && x >= 0 && x <= 1);
addParameter(p, 'ScaleRange', [0.9, 1.1], @(x) isnumeric(x) && length(x) == 2);
addParameter(p, 'TranslationRange', [-10, 10], @(x) isnumeric(x) && length(x) == 2);
addParameter(p, 'BrightnessRange', [-0.1, 0.1], @(x) isnumeric(x) && length(x) == 2);
addParameter(p, 'ContrastRange', [0.9, 1.1], @(x) isnumeric(x) && length(x) == 2);
addParameter(p, 'NoiseStd', 0.01, @(x) isnumeric(x) && x >= 0);

parse(p, varargin{:});
opts = p.Results;

% 创建输出目录结构
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

imageDir = fullfile(output_dir, dataset_name, 'images');
labelDir = fullfile(output_dir, dataset_name, 'labels');

if ~exist(imageDir, 'dir')
    mkdir(imageDir);
end
if ~exist(labelDir, 'dir')
    mkdir(labelDir);
end

% 获取分割帧的信息
segmented_indices = segmentation_data.frame_indices;
masks = segmentation_data.masks;
num_frames = length(segmented_indices);

if opts.EnableAugmentation
    fprintf('数据增强已启用 - 每帧将生成 %d 个增强版本\n', opts.AugmentationFactor);
    total_images = num_frames * (1 + opts.AugmentationFactor);
else
    total_images = num_frames;
end

fprintf('共有 %d 个关键帧，将生成 %d 个训练样本\n', num_frames, total_images);

% 图像计数器
img_counter = 1;

% 逐帧处理并保存
for i = 1:num_frames
    frame_idx = segmented_indices(i);
    
    % 获取当前帧图像和mask
    current_img = img_3d(:, :, frame_idx);
    current_mask = masks{i};
    
    % 预处理：调整大小
    processed_img = rescale(double(current_img));
    processed_img = imresize(processed_img, imageSize(1:2), "bilinear");
    processed_mask = imresize(current_mask, imageSize(1:2), "nearest"); % 使用最近邻避免标签插值
    
    % 保存原始图像
    [img_path, label_path] = saveImageAndLabel(processed_img, processed_mask, ...
        imageDir, labelDir, img_counter, imageSize);
    img_counter = img_counter + 1;
    
    % 如果启用数据增强，生成增强图像
    if opts.EnableAugmentation
        for aug_idx = 1:opts.AugmentationFactor
            [aug_img, aug_mask] = applyDataAugmentation(processed_img, processed_mask, opts);
            
            % 保存增强图像
            [img_path, label_path] = saveImageAndLabel(aug_img, aug_mask, ...
                imageDir, labelDir, img_counter, imageSize);
            img_counter = img_counter + 1;
        end
    end
    
    if mod(i, 10) == 0 || i == num_frames
        if opts.EnableAugmentation
            fprintf('已处理 %d/%d 帧 (生成了 %d 个训练样本)\n', i, num_frames, img_counter-1);
        else
            fprintf('已处理 %d/%d 帧\n', i, num_frames);
        end
    end
end

% 创建ImageDatastore
imds = imageDatastore(imageDir);

% 对于分类任务，使用pixelLabelDatastore
classNames = ["object", "background"];
labelIDs = [1, 0];
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);

% 组合数据集
fprintf('组合数据集...\n');
ds = combine(imds, pxds);

% 验证数据集
fprintf('验证数据集格式...\n');
reset(ds);
sample = read(ds);
img = sample{1};
label = sample{2};

fprintf('图像格式: %s, %s\n', mat2str(size(img)), class(img));
fprintf('图像值范围: [%.3f, %.3f]\n', min(img(:)), max(img(:)));
fprintf('标签格式: %s, %s\n', mat2str(size(label)), class(label));
fprintf('标签类别: %s\n', strjoin(string(categories(label)), ', '));
fprintf('标签统计: %s\n', mat2str(countcats(label(:))));

% 保存数据集信息
dataset_info = struct();
dataset_info.name = dataset_name;
dataset_info.num_original_frames = num_frames;
dataset_info.total_samples = img_counter - 1;
dataset_info.augmentation_enabled = opts.EnableAugmentation;
dataset_info.augmentation_factor = opts.AugmentationFactor;
dataset_info.image_dir = imageDir;
dataset_info.label_dir = labelDir;
dataset_info.augmentation_params = opts;

info_path = fullfile(output_dir, dataset_name, 'dataset_info.mat');
save(info_path, 'dataset_info');

fprintf('\n数据集构建完成!\n');
fprintf('数据集位置: %s\n', fullfile(output_dir, dataset_name));
fprintf('原始帧数: %d\n', num_frames);
fprintf('总训练样本数: %d\n', img_counter - 1);

% 重置数据集指针
reset(ds);
end

function [img_path, label_path] = saveImageAndLabel(img, mask, imageDir, labelDir, counter, imageSize)
% 保存图像和标签的辅助函数
    
    % 归一化图像到[0,1]范围
    img_normalized = rescale(img);
    
    % 生成文件名
    img_filename = sprintf('frame_%06d.png', counter);
    label_filename = sprintf('frame_%06d.png', counter);
    
    % 保存图像
    img_path = fullfile(imageDir, img_filename);
    imwrite(img_normalized, img_path);
    
    % 处理mask：创建0和1的二元标签
    label_img = uint8(mask > 0); % 转为0和1的uint8格式
    
    % 保存标签图像
    label_path = fullfile(labelDir, label_filename);
    imwrite(label_img, label_path);
end

function [aug_img, aug_mask] = applyDataAugmentation(img, mask, opts)
% 应用数据增强的辅助函数
    
    aug_img = img;
    aug_mask = mask;
    
    % 1. 几何变换（同时应用于图像和mask）
    tform = getRandomTransform(opts, size(img));
    
    if ~isempty(tform)
        % 应用几何变换
        aug_img = imwarp(aug_img, tform, 'OutputView', imref2d(size(img)), ...
                        'FillValues', mean(img(:)), 'Interp', 'linear');
        aug_mask = imwarp(aug_mask, tform, 'OutputView', imref2d(size(mask)), ...
                         'FillValues', 0, 'Interp', 'nearest');
    end
    
    % 2. 翻转（同时应用于图像和mask）
    if rand < opts.FlipProbability
        if rand < 0.5
            % 水平翻转
            aug_img = fliplr(aug_img);
            aug_mask = fliplr(aug_mask);
        else
            % 垂直翻转
            aug_img = flipud(aug_img);
            aug_mask = flipud(aug_mask);
        end
    end
    
    % 3. 亮度调整（仅应用于图像）
    brightness_delta = opts.BrightnessRange(1) + ...
        (opts.BrightnessRange(2) - opts.BrightnessRange(1)) * rand;
    aug_img = aug_img + brightness_delta;
    
    % 4. 对比度调整（仅应用于图像）
    contrast_factor = opts.ContrastRange(1) + ...
        (opts.ContrastRange(2) - opts.ContrastRange(1)) * rand;
    img_mean = mean(aug_img(:));
    aug_img = (aug_img - img_mean) * contrast_factor + img_mean;
    
    % 5. 添加高斯噪声（仅应用于图像）
    if opts.NoiseStd > 0
        noise = randn(size(aug_img)) * opts.NoiseStd;
        aug_img = aug_img + noise;
    end
    
    % 确保图像值在合理范围内
    aug_img = max(0, min(1, aug_img));
    
    % 确保mask只包含0和1
    aug_mask = double(aug_mask > 0.5);
end

function tform = getRandomTransform(opts, img_size)
% 生成随机几何变换
    
    % 随机旋转
    rotation_angle = opts.RotationRange(1) + ...
        (opts.RotationRange(2) - opts.RotationRange(1)) * rand;
    
    % 随机缩放
    scale_factor = opts.ScaleRange(1) + ...
        (opts.ScaleRange(2) - opts.ScaleRange(1)) * rand;
    
    % 随机平移
    tx = opts.TranslationRange(1) + ...
        (opts.TranslationRange(2) - opts.TranslationRange(1)) * rand;
    ty = opts.TranslationRange(1) + ...
        (opts.TranslationRange(2) - opts.TranslationRange(1)) * rand;
    
    % 组合变换
    if abs(rotation_angle) > 0.1 || abs(scale_factor - 1) > 0.01 || abs(tx) > 0.1 || abs(ty) > 0.1
        % 计算图像中心
        center_x = img_size(2) / 2; % 列中心 (x)
        center_y = img_size(1) / 2; % 行中心 (y)
        
        % 计算旋转和缩放的组合矩阵
        cos_theta = cosd(rotation_angle);
        sin_theta = sind(rotation_angle);
        
        % 构建正确格式的affine2d变换矩阵 [a b 0; c d 0; tx ty 1]
        % 先缩放再旋转，然后平移
        a = scale_factor * cos_theta;
        b = -scale_factor * sin_theta;
        c = scale_factor * sin_theta;
        d = scale_factor * cos_theta;
        
        % 计算总的平移量（包括由于旋转缩放导致的中心偏移）
        tx_total = (1 - a) * center_x - b * center_y + tx;
        ty_total = -c * center_x + (1 - d) * center_y + ty;
        
        % 构建affine2d变换矩阵
        tform_matrix = [a b 0; c d 0; tx_total ty_total 1];
        
        tform = affine2d(tform_matrix);
    else
        tform = [];
    end
end