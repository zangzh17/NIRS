img_3d = load_tif_block('D:\NIR_SLIM\data\250528_mutant_1',...
                    '19_fish_600hz_rl_15.tif');
%% 
im = double(img_3d);
window_size = 10;
[height, width, num_frames] = size(im);
result_adaptive = zeros(size(im));
% 对每一帧计算自适应背景
for t = 1:num_frames
    % 确定窗口范围
    start_frame = max(1, t - floor(window_size/2));
    end_frame = min(num_frames, t + floor(window_size/2));
    
    % 提取窗口内的数据
    window_data = im(:, :, start_frame:end_frame);
    
    % 计算窗口内的中值背景
    adaptive_background = median(window_data, 3);
    
    % 背景减除
    result_adaptive(:, :, t) = im(:, :, t) - adaptive_background;
end
sliceViewer(rescale(result_adaptive));

%% train
terminate(pyenv)
pyenv("Version",'C:\Research\deep-learning\.venv\Scripts\python.exe')
addpath("DeepCAD_RT_pytorch\")
deepcad_path = 'D:\NIR_SLIM\DeepCAD_RT_pytorch';
denoise_model = 'record_11062025_170611_20hz_ss_2_filtered';
deepcad_train(filtered_video, deepcad_path, ...
    'model_name', denoise_model,...
    'n_epochs', 8,...
    'train_datasets_size',1000);

%% test
addpath("DeepCAD_RT_pytorch\")
deepcad_path = 'D:\NIR_SLIM\DeepCAD_RT_pytorch';
im = deepcad_denoise(filtered_video,denoise_model, deepcad_path);


%% 保存为MP4视频文件
[file, path] = uiputfile('*.mp4', '保存蒙版后的3D图像为视频',denoise_model);
if file ~= 0
    full_path = fullfile(path, file);
    fprintf('正在保存到: %s\n', full_path);
    
    % 转换为uint8格式 (MP4视频通常使用8位)
    im_uint8 = uint8(rescale(im) * 255);
    
    % 创建VideoWriter对象
    v = VideoWriter(full_path, 'MPEG-4');
    v.FrameRate = 10;  % 设置帧率，可根据需要调整
    v.Quality = 99;    % 设置质量 (0-100)
    
    % 打开视频文件
    open(v);
    
    % 保存视频帧
    h = waitbar(0,'保存视频中...');
    try
        for i = 1:size(im_uint8, 3)
            % 获取当前帧
            frame = im_uint8(:,:,i);
            
            % 如果是灰度图像，转换为RGB格式
            if size(frame, 3) == 1
                frame = repmat(frame, [1, 1, 3]);
            end
            
            % 写入视频帧
            writeVideo(v, frame);
            
            % 更新进度条
            if mod(i, 10) == 0
                waitbar(i/size(im_uint8, 3), h, '保存视频中...');
            end
        end
        fprintf('保存完成！\n');
    catch ME
        fprintf('保存过程中出现错误: %s\n', ME.message);
    end
    
    % 关闭视频文件和进度条
    close(v);
    close(h);
end