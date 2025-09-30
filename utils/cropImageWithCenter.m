function [im_cropped, crop_params] = cropImageWithCenter(im, config_file, crop_size)
% CROPIMAGEWITHCENTER 基于固定大小和中心点的图像裁剪函数
%
% 输入参数:
%   im - 输入图像 (height × width × slices)
%   crop_size - 裁剪大小 [width, height] (不输入则读取配置文件）
%   config_file - 配置文件路径字符串
%
% 输出参数:
%   im_cropped - 裁剪后的图像
%   crop_params - 裁剪参数结构体 {center_x, center_y, crop_width, crop_height}

    % 不输入crop_size则尝试从配置文件加载参数
    if nargin < 3
        try
            loaded_config = load(config_file);
            if isfield(loaded_config, 'crop_params')
                crop_params = loaded_config.crop_params;
            end
        catch
            fprintf('Read file failed...\n');
        end
    else
        % 初始化裁剪参数
        crop_params = struct();
        % 交互式选择中心点
        [center_x, center_y] = selectCenterPoint(im, crop_size(1), crop_size(2));
        % 保存裁剪参数
        crop_params.center_x = center_x;
        crop_params.center_y = center_y;
        crop_params.crop_width = crop_size(1);
        crop_params.crop_height = crop_size(2);
        crop_params.timestamp = datetime('now');
        % 保存配置到文件
        try
            save([config_file,'.mat'], 'crop_params');
            fprintf('Config file: %s\n', config_file);
        catch
            warning('Cannot save: %s', config_file);
        end
    end

    % 执行裁剪
    im_cropped = performCrop(im, crop_params);
    
end

function [center_x, center_y] = selectCenterPoint(im, crop_width, crop_height)
% 交互式选择裁剪中心点
    
    [height, width, ~] = size(im);
    
    % 显示图像
    f = figure('Name', 'Select center', 'NumberTitle', 'off');
    imshow(im(:,:,1), [], 'InitialMagnification', 'fit');
    title(sprintf('Click to select center of cropping box (Crop size: %d × %d)', crop_width, crop_height));
    axis on;
    hold on;
    
    % 绘制可裁剪区域提示
    valid_x_min = ceil(crop_width/2);
    valid_x_max = width - floor(crop_width/2);
    valid_y_min = ceil(crop_height/2);
    valid_y_max = height - floor(crop_height/2);
    
    % 绘制有效区域边界
    rectangle('Position', [valid_x_min-0.5, valid_y_min-0.5, ...
                          valid_x_max-valid_x_min+1, valid_y_max-valid_y_min+1], ...
                          'EdgeColor', 'g', 'LineWidth', 1, 'LineStyle', '--');
    
    % 等待用户点击
    fprintf('Wait user to click...\n');
    [x_click, y_click] = ginput(1);
    
    % 约束中心点到有效范围
    center_x = round(max(valid_x_min, min(valid_x_max, x_click)));
    center_y = round(max(valid_y_min, min(valid_y_max, y_click)));
    
    % 绘制预览框
    x1 = center_x - floor(crop_width/2);
    y1 = center_y - floor(crop_height/2);
    
    rectangle('Position', [x1-0.5, y1-0.5, crop_width, crop_height], ...
                            'EdgeColor', 'r', 'LineWidth', 2);
    plot(center_x, center_y, 'r+', 'MarkerSize', 10, 'LineWidth', 2);
    pause(0.5);
    close(f);
end

function im_cropped = performCrop(im, crop_params)
% 执行实际的裁剪操作
    
    center_x = crop_params.center_x;
    center_y = crop_params.center_y;
    crop_width = crop_params.crop_width;
    crop_height = crop_params.crop_height;
    
    % 计算裁剪边界
    x1 = center_x - floor(crop_width/2);
    x2 = x1 + crop_width - 1;
    y1 = center_y - floor(crop_height/2);
    y2 = y1 + crop_height - 1;
    
    % 直接裁剪（之前已经确保了边界的有效性）
    im_cropped = im(y1:y2, x1:x2, :);
end
