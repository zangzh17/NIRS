img_3d = load_tif_block('E:\250611\2',...
                    'record_11062025_171154_10hz_ss_2.tif');
model_name = 'record_11062025_171154_10hz_ss_2';
%% add mask
f = figure;
first_frame = img_3d(:,:,1);
[I,rect] = imcrop(first_frame);
% 裁剪
img_data = zeros(size(I,1),size(I,2),size(img_3d,3));
for i=1:size(img_3d,3)
    img_data(:,:,i) = imcrop(img_3d(:,:,i) ,rect);
end
close;
fprintf('Size after crop: %d x %d x %d\n', size(img_data));

%% train
terminate(pyenv)
pyenv("Version",'C:\Research\deep-learning\.venv\Scripts\python.exe')
addpath("DeepCAD_RT_pytorch\")
deepcad_path = 'D:\NIR_SLIM\DeepCAD_RT_pytorch';
deepcad_train(uint16(img_data), deepcad_path, ...
    'model_name', model_name,...
    'n_epochs', 5,...
    'train_datasets_size',1250);

%% test
addpath("DeepCAD_RT_pytorch\")
deepcad_path = 'D:\NIR_SLIM\DeepCAD_RT_pytorch';
im = deepcad_denoise(img_data,model_name, deepcad_path);
figure;
sliceViewer(im);

%% circular crop

% I = im(:,:,1000);
% % J = adapthisteq(rescale(I));
% % gamma = 2;
% J = rescale(imadjust(I,[],[],gamma));
% figure
% imshow(J)
figure;
imshow(im(:,:,1), []);
hold on;
circ = drawcircle();
wait(circ);
BW = repmat(createMask(circ),[1,1,size(im,3)]);
close
im(~BW) = min(im(BW),[],"all");
figure;sliceViewer(im);
%% 全部：调整对比度和gamma
im_masked2 = im;
for i=1:size(im_masked2,3)
    I = im_masked2(:,:,i);
    
    % % option#1
    % I0 = mean(I(mask_2d),"all");
    % I(~mask_2d) = I0;
    % gamma = 5;
    % J = adapthisteq(rescale(I));
    % J = imadjust(J,[],[],gamma);

    % % option#2
    gamma = 1.5;
    % J = adapthisteq(rescale(I));
    J = rescale(imadjust(I,[],[],gamma));

    im_masked2(:,:,i) = J;
    if mod(i,100)==0
        fprintf('Processing %d/%d\n', i, size(im_masked,3));
    end
end
figure;sliceViewer(im_masked2);

%%
low_freq = 0.2; % 默认低频
high_freq = 5; % 默认高频
fps = 20;
% im_stack = im_masked;
im_stack = double(im);
[h,w,n_frames] = size(im_stack);

% % 设计带通滤波器
% nyquist = fps / 2;
% low_norm = low_freq / nyquist;
% high_norm = high_freq / nyquist;
% [b, a] = butter(4, [low_norm, high_norm], 'bandpass');

% 设计高通滤波器
nyquist = fps / 2;
[b, a] = butter(4, low_freq / nyquist, 'high');

% 对每个像素滤波
filtered_video = zeros(size(im_stack));
for i = 1:h
    for j = 1:w
        pixel_timeseries = squeeze(im_stack(i, j, :));
        filtered_series = filtfilt(b, a, pixel_timeseries);
        filtered_video(i, j, :) = rescale(filtered_series);
    end
    if mod(i, 50) == 0
        fprintf('处理进度: %d/%d\n', i, h);
    end
end
sliceViewer(rescale(filtered_video));

%% 保存为TIFF volume文件
[file, path] = uiputfile('*.tif', '保存蒙版后的3D图像');
if file ~= 0
    full_path = fullfile(path, file);
    
    fprintf('正在保存到: %s\n', full_path);
    im_masked_unit16 = uint16(rescale(im) * 65535);
    % 保存多页TIFF
    h = waitbar(0,'Saving');
    for i = 1:size(im_masked_unit16, 3)
        if i == 1
            imwrite(im_masked_unit16(:,:,i), full_path, 'tif', 'Compression', 'none');
        else
            imwrite(im_masked_unit16(:,:,i), full_path, 'tif', 'WriteMode', 'append', 'Compression', 'none');
        end
        if mod(i,50)==0
            waitbar(i/size(im_masked_unit16, 3),h,'Saving');
        end
    end
    fprintf('保存完成！\n');
end
close(h)

%% 保存为MP4视频文件
[file, path] = uiputfile('*.mp4', '保存蒙版后的3D图像为视频',denoise_model);
if file ~= 0
    full_path = fullfile(path, file);
    fprintf('正在保存到: %s\n', full_path);
    
    % 转换为uint8格式 (MP4视频通常使用8位)
    im_masked_uint8 = uint8(rescale(im) * 255);
    
    % 创建VideoWriter对象
    v = VideoWriter(full_path, 'MPEG-4');
    v.FrameRate = 10;  % 设置帧率，可根据需要调整
    v.Quality = 99;    % 设置质量 (0-100)
    
    % 打开视频文件
    open(v);
    
    % 保存视频帧
    h = waitbar(0,'保存视频中...');
    try
        for i = 1:size(im_masked_uint8, 3)
            % 获取当前帧
            frame = im_masked_uint8(:,:,i);
            
            % 如果是灰度图像，转换为RGB格式
            if size(frame, 3) == 1
                frame = repmat(frame, [1, 1, 3]);
            end
            
            % 写入视频帧
            writeVideo(v, frame);
            
            % 更新进度条
            if mod(i, 10) == 0
                waitbar(i/size(im_masked_uint8, 3), h, '保存视频中...');
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