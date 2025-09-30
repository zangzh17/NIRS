%% 1. 数据加载和预处理
% 创建视频读取对象
videoFile = 'D:\NIR_SLIM\data\250528_mutant_1\30_fish_600hz_rl_depth.mp4';
v = VideoReader(videoFile);
% 获取视频基本信息
T = floor(v.Duration * v.FrameRate);
W = v.Width;
H = v.Height;
fps = 600;
fprintf('视频信息: %dx%d, 预估%d帧, %.1f fps\n', W, H, T, fps);
% 读取原始彩色视频数据
original_color_video = zeros(H, W, 3, T,'uint8'); % 彩色视频：[height, width, RGB, time]
video_data = zeros(H, W, T); % 灰度数据用于处理
fprintf('正在读取视频数据...\n');
tic;
k = 1;
while hasFrame(v) && k <= T
    frame_color = readFrame(v);
    original_color_video(:,:,:,k) = uint8(frame_color); % 保存彩色帧
    video_data(:,:,k) = uint8(rgb2gray(frame_color)); % 灰度用于处理
    k = k + 1;
    if mod(k-1, 200) == 0
        fprintf('已读取 %d/%d 帧\n', k-1, T);
    end
end
fprintf('视频读取完成，实际帧数: %d，耗时: %.2f秒\n', T, toc);

%% 2. 裁剪区域选择
figure
imshow(rescale(video_data(:,:,1)))
title('选择要处理的区域')
[~, crop_rect] = imcrop();
close
% 处理裁剪参数
x_start = max(1, round(crop_rect(1)));
y_start = max(1, round(crop_rect(2)));
crop_width = round(crop_rect(3));
crop_height = round(crop_rect(4));
x_end = min(W, x_start + crop_width - 1);
y_end = min(H, y_start + crop_height - 1);
% 裁剪灰度数据用于处理
video_data_crop = video_data(y_start:y_end, x_start:x_end, :);
[H_crop, W_crop, ~] = size(video_data_crop);
fprintf('裁剪区域: (%d,%d) 到 (%d,%d), 尺寸: %dx%d\n', ...
    x_start, y_start, x_end, y_end, W_crop, H_crop);
%% 3. 频域滤波增强心脏节律 - 优化版本
fprintf('正在进行心脏节律滤波...\n');
tic;
heart_rate_range = [1, 3]; % Hz
% 设计滤波器
[b, a] = butter(2, heart_rate_range/(fps/2), 'bandpass');
% 向量化滤波处理 - 这部分可以安全向量化
video_2d = reshape(video_data_crop, H_crop * W_crop, T);
cardiac_filtered_2d = filtfilt(b, a, double(video_2d'))'; % 转置进行滤波，再转置回来
cardiac_filtered = rescale(reshape(cardiac_filtered_2d, H_crop, W_crop, T));
[N,edges] = histcounts(cardiac_filtered,1024);
[~,idx] = max(N); 
mid_ratio = edges(idx);
cardiac_norm = cardiac_filtered - mid_ratio;

% 自适应权重叠加 - 向量化
alpha = 0.2; % 原图权重
adaptive_enhanced = rescale(alpha * rescale(double(video_data_crop)) + (1-alpha) * cardiac_norm);
fprintf('心脏节律滤波完成，耗时: %.2f秒\n', toc);
sliceViewer(adaptive_enhanced)
%% 4. 将增强数据放回原始彩色视频
fprintf('灰度替换...\n');
tic;
% 创建另一个副本用于灰度替换
gray_replaced_video = original_color_video;
% 将增强的灰度数据直接替换到所有RGB通道
figure
for t = 1:T
    for c = 1:3
        gray_replaced_video(y_start:y_end, x_start:x_end, c, t) = adaptive_enhanced(:,:,t) * 255;
    end
    if mod(t-1,20)==0
        imshow(rescale(gray_replaced_video(:,:,:,t)));
    end
end
fprintf('方法2完成：直接灰度替换，耗时: %.2f秒\n', toc);

%% 方法3: cardiac_filtered伪彩色热图叠加 (高级选项)
fprintf('生成高级方案：cardiac_filtered伪彩色热图叠加...\n');
tic;

% 创建伪彩色热图版本
heatmap_overlay_video = original_color_video;

% 设置叠加参数
alpha_heatmap = 0.5;  % 越大热图越明显
amp = 2; % 热图幅度
colormap_choice = 'hot'; % 可选: 'jet', 'hot', 'parula', 'turbo', 'viridis'

% 获取colormap
cmap = eval([colormap_choice, '(256)']);

for t = 1:T
    % 提取当前帧的裁剪区域
    crop_rgb = squeeze(original_color_video(y_start:y_end, x_start:x_end, :, t));
    % 将cardiac信号转换为伪彩色
    cardiac_frame = abs(cardiac_norm(:,:,t)) * amp;
    % 将归一化数据映射到colormap索引
    cardiac_indices = round(cardiac_frame * 255) + 1;
    cardiac_indices = max(1, min(256, cardiac_indices));
    
    % 创建RGB热图
    heatmap_rgb = uint8(reshape( cmap(cardiac_indices,:), [H_crop, W_crop, 3]) * 255);
    
    % 叠加热图和原始图像
    overlay_rgb = (1 - alpha_heatmap) * double(crop_rgb) + alpha_heatmap * double(heatmap_rgb);
    
    % 放回原视频
    heatmap_overlay_video(y_start:y_end, x_start:x_end, :, t) = overlay_rgb;

    if mod(t-1,20)==0
        imshow(heatmap_overlay_video(:,:,:,t));
    end
end

fprintf('cardiac_filtered伪彩色热图叠加，耗时: %.2f秒\n', toc);

%% 6. 保存增强后的视频 (可选)


% 创建输出文件名
[file_path, base_name, ~] = fileparts(videoFile);
fps_save = 100;

filename_suffix = '_amp';
output_filename = fullfile(file_path,[base_name, filename_suffix, '.mp4']);
outputVideo = VideoWriter(output_filename, 'MPEG-4');
outputVideo.FrameRate = fps_save;
outputVideo.Quality = 95;
open(outputVideo);
% 写入每一帧
for t = 1:T
    frame = squeeze(gray_replaced_video(:,:,:,t));
    writeVideo(outputVideo, frame);
    if mod(t, 100) == 0
        fprintf('已保存 %d/%d 帧\n', t, T);
    end
end
close(outputVideo);
fprintf('视频保存完成: %s\n', output_filename);

filename_suffix = '_amp_colored';
output_filename = fullfile(file_path,[base_name, filename_suffix, '.mp4']);
outputVideo = VideoWriter(output_filename, 'MPEG-4');
outputVideo.FrameRate = fps_save;
open(outputVideo);
% 写入每一帧
for t = 1:T
    frame = squeeze(heatmap_overlay_video(:,:,:,t));
    writeVideo(outputVideo, frame);
    if mod(t, 100) == 0
        fprintf('已保存 %d/%d 帧\n', t, T);
    end
end
close(outputVideo);
fprintf('视频保存完成: %s\n', output_filename);

