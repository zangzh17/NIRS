%% load data
data_path = 'D:\NIR_SLIM\data\250528_mutant_1';
geo_file = 'D:\NIR_SLIM\data\250528_control_fish_1_1\250607_geo_1.mat';
psf_file = 'D:\NIR_SLIM\data\250528_control_fish_1_1\250607_psf_1.mat';

num_depth = 16; % 重建层数

% 读取校准文件
load(geo_file);
load(psf_file);

% 获取所有文件
allItems = {dir(data_path).name};

% 初始化结构体数组
fileData = [];
fileList = [];

% 遍历所有文件，找到非.txt的数据文件
for k = 1:numel(allItems)
    [~, baseName, ext] = fileparts(allItems{k});

    % 跳过隐藏文件和当前目录标识
    if startsWith(allItems{k}, '.') || strcmp(allItems{k}, '..') || strcmp(allItems{k}, '.')
        continue;
    end
    % 如果不是.txt文件，认为是数据文件
    if ~strcmp(ext, '.txt')
        % 检查是否有对应的配置文件
        configFileName = [baseName, '.txt'];
        configFilePath = fullfile(data_path, configFileName);
        currentFileData = struct();
        % 如果配置文件存在，读取配置参数
        if isfile(configFilePath)
            fid = fopen(configFilePath, 'r');
            % 读取depth_range行
            line1 = fgetl(fid);
            if ischar(line1)
                depth_values = sscanf(line1, 'depth_range = [%f, %f]');
                if length(depth_values) >= 2
                    currentFileData.depth_range = depth_values;
                end
            end
            % 读取PSF_n行
            line2 = fgetl(fid);
            if ischar(line2)
                PSF_n_value = sscanf(line2, 'PSF_n = %f');
                if ~isempty(PSF_n_value)
                    currentFileData.PSF_n = PSF_n_value;
                end
            end
            % 读取nIters行
            line3 = fgetl(fid);
            if ischar(line3)
                nIters_value = sscanf(line3, 'nIters = %d');
                if ~isempty(nIters_value)
                    currentFileData.nIters = nIters_value;
                end
            end
            % 读取clip_range行
            line4 = fgetl(fid);
            if ischar(line4)
                clip_range = sscanf(line4, 'clip_range = [%f, %f]');
                if length(clip_range) >= 2
                    currentFileData.clip_range = clip_range;
                end
            end
            % 读取gamma行
            line5 = fgetl(fid);
            if ischar(line5)
                gamma = sscanf(line5, 'gamma = %f');
                if ~isempty(gamma)
                    currentFileData.gamma = gamma;
                end
            end
            fclose(fid);
            % 将配置信息添加到结构体数组
            fileData = [fileData, currentFileData];
            fileList = [fileList, string(allItems{k})];
        end
    end
end
disp([num2str(length(fileData)), ' files loaded!'])
disp(fileList)

%%% 准备校准参数
% crop mask
y_crop_range = zeros(length(ellipse_mask),2);
x_crop_range = zeros(length(ellipse_mask),2);
for i=1:length(ellipse_mask)
    % Find the bounding box of each ellipse region
    [rows, cols] = find(ellipse_mask{i});
    row_min = min(rows);
    row_max = max(rows);
    col_min = min(cols);
    col_max = max(cols);
    % Extract the subimages
    y_crop_range(i,:) = [row_min,row_max];
    x_crop_range(i,:) = [col_min,col_max];
end

% Prepare transformation
% % prepare forward transformation
lambda = linspace(depth_range(1), depth_range(2), num_depth);
% invert
forward_tform_list = optimized_tform_list;
num_frame = size(optimized_tform_list,2);
num_view = size(optimized_tform_list,1);
for i=1:num_view
    for j=1:num_frame
        forward_tform_list(i,j) = forward_tform_list(i,j).invert;
    end
end
% interp transformation
% reconstruction number of depth setting
forward_tform_list = interp_transformations(forward_tform_list,lambda);
backward_tform_list = interp_transformations(optimized_tform_list,lambda);
% % interp rel transformation
tform_bkwd_ref = repmat(affinetform2d(),[1,num_view]);
tform_bkwd_rel = repmat(affinetform2d(),[num_view,num_depth]);
mid_z = floor(num_depth/2)+1;
for i=1:num_view
    tform_tmp = invert(backward_tform_list(i,mid_z));
    A_ref_fwd = tform_tmp.A;
    % bkwd_ref: LF space -> 3D obj space (depth invariant)
    tform_bkwd_ref(i) = backward_tform_list(i,mid_z);
    for j=1:num_depth
        A = backward_tform_list(i,j).A * A_ref_fwd;
        tform_bkwd_rel(i,j) = affinetform2d(A);
    end
end

% Prepare ROI mask
roi_ratio = 0.85; % ROI ratio (both height and width)
depth_center= 128; % choose depth for ROI mask generation (geo. transform)
depth_bw = 0.25;
depth_avg_num = 3;
blur_range = 10; % gaussian filter param. for soft mask
lambda = linspace(depth_center-depth_bw/2,depth_center+depth_bw/2,depth_avg_num);
tform_list_roi = interp_transformations(forward_tform_list,lambda);
roi_mask = generate_roi_masks(tform_list_roi,...
    size(measurements(:,:,1)), roi_ratio, blur_range);
roi_bound = cell(1,size(measurements,3));
for c = 1:length(roi_bound)
    mask = imbinarize(roi_mask(:,:,c));
    roi_bound{c} = bwboundaries(mask);
end

%% process
%%% 默认参数
num_avg = 1; % 取平均数目
chunk_size = 10;
for i=1:length(fileList)
    %%% 默认参数
    PSF_n = 2;
    depth_range = [0,1];
    gamma = 1;
    clip_range = [0,1];

    %%% read file
    [~, filename, ext] = fileparts(fileList(i));
    if strcmp(ext,'.hdf5')
        im_data = load_hdf5_cam(data_path,fileList(i));
        im_data = rescale(double(im_data));
    elseif strcmp(ext,'.mat')
        S = load(fullfile(data_path,filename));
        im_data = permute(S.imgcube,[2,1,3]);
    elseif strcmp(ext,'.raw')
        im_data = load_raw_block(data_path,filename);
        im_data = rescale(double(im_data));
    end

    % load parameter
    if isfield(fileData(i),'PSF_n')
        PSF_n = fileData(i).PSF_n;
    end
    if isfield(fileData(i),'depth_range')
        depth_range = fileData(i).depth_range;
    end
    if isfield(fileData(i),'nIters')
        nIters = fileData(i).nIters;
    end
    if isfield(fileData(i),'gamma')
        gamma = fileData(i).gamma;
    end
    if isfield(fileData(i),'clip_range')
        clip_range = fileData(i).clip_range;
    end
    
    % 对im_data每隔num_avg帧取平均，抛弃多余的帧
    num_images = size(im_data, 3);
    num_frame = ceil(num_images / num_avg);
    for j=1:num_avg
        im_data(:,:,1:end-j+1) = im_data(:,:,1:end-j+1) + im_data(:,:,j:end);
    end
    im_data = im_data(:,:,1:num_avg:end)/num_avg;
    
    msg = sprintf('File %s: %d Frames -> %d Frames after Avg\n', filename, num_images, num_frame);
    
    % 对平均后的帧进行分块处理
    num_chunks = ceil(num_frame / chunk_size);
    for chunk_idx = 1:num_chunks
        start_frame = (chunk_idx - 1) * chunk_size + 1;
        end_frame = min(chunk_idx * chunk_size, num_frame);
        
        % 提取当前块的数据
        im = im_data(:,:,start_frame:end_frame);
        
        %%% 处理
        % bad pixel correction
        for k = 1:size(im,3)
            im(:,:,k) = correctHotPixels(im(:,:,k),...
                        'zScore', 2,...
                        'WinSize', 5,...
                        'MaxSize', 3);
        end
        % 对比度
        im = imadjustn(im, clip_range, [0,1], gamma);

        % 切除和缩放
        % Crop out sub-apertures and stretch data images
        % % prepare test data as measurements
        measurements = cell(1,length(ellipse_mask));
        im = imresize(permute(im,[2,1,3]),resample_size);
        for j=1:length(ellipse_mask)
            % Extract the subimages
            im_data_sub = im;
            im_data_sub(~ellipse_mask{j}) = 0;
            im_data_sub = im_data_sub(y_crop_range(1):y_crop_range(2), x_crop_range(1):x_crop_range(2), :);
            % Stretch the subimage to make the ellipse a circle
            im_data_sub = imresize(im_data_sub,stretched_size);
            % save sub-images
            measurements{j} = rescale(im_data_sub);
        end
        % prepare data
        measurements = permute(cat(4, measurements{:}),[1,2,4,3]);
        

        sliceViewer(im);
        pause(1);
    end
end