%% load

addpath('./utils/');
dataset_name = 'record_02092025_183907_rl_10';
im = load_tif_block('E:\250902_fish\6',...
                    [dataset_name,'.tif']);

% add rectangular mask
dataset_path = fullfile(pwd,'dataset_seg',dataset_name);
if ~exist(fullfile('./dataset_seg/',dataset_name),"dir")
    mkdir(fullfile('./dataset_seg/',dataset_name))
end

% manual crop and save crop patameters
config_path = 'E:\250902_fish\6';
config_file = fullfile(dataset_path,'crop_config');
crop_size = [100,100]; img = cropImageWithCenter(im, config_file, crop_size);

% load and apply exsiting crop patameters (if already saved)
% img = cropImageWithCenter(im, config_file);

fprintf('Data loaded!\n');


%% RUN segmentation
segmentation_data = keyframeSegTool(img);
seg_file = fullfile(dataset_path,'segmentation_data.mat');
save(seg_file,"segmentation_data");

%% load segmentation
seg_file = fullfile(dataset_path,'segmentation_data.mat');
load(seg_file,"segmentation_data");

%% gen dataset and train U-net
% generate dataset
imageSize = [128,128,1];
% Custom augmentation parameters
[dsTrain, ~, ~] = buildDataset(img, segmentation_data, dataset_path, dataset_name, imageSize, ...
    'EnableAugmentation', true, ...
    'AugmentationFactor', 8, ...           % Generate n augmented versions per frame
    'RotationRange', [-3.5, 3.5], ...      % Rotation range ±20 degrees
    'FlipProbability', 0, ...              % Flip probability
    'ScaleRange', [0.8, 1.1], ...          % Scaling range 0.8–1.2x
    'TranslationRange', [-7, 7], ...       % Translation range ±15 pixels
    'BrightnessRange', [-0.1, 0.1], ...    % Brightness adjustment range
    'ContrastRange', [0.85, 1.15], ...     % Contrast adjustment range
    'NoiseStd', 0);                        % Gaussian noise standard deviation

%% init U-Net

% lr = 2e-5;
% load('D:\NIR_SLIM\dataset_seg\19_fish_600hz_rl_20\net.mat'); unetNetwork = net;

lr = 5e-5;
unetNetwork = unet(imageSize, 2, EncoderDepth=3);

% Dice Loss + Cross Entropy
function loss = diceCrossEntropyLoss(Y, T)
    % Dice Loss part
    smooth = 1e-6;
    intersection = sum(Y .* T, [1,2]);
    dice_coeff = (2 * intersection + smooth) ./ (sum(Y, [1,2]) + sum(T, [1,2]) + smooth);
    dice_loss = 1 - mean(dice_coeff, 'all');
    % Cross Entropy part
    ce_loss = crossentropy(Y, T);
    % Combined loss
    loss = 0.7 * dice_loss + 0.3 * ce_loss;
end

% train U-Net
options = trainingOptions("adam", ...       % Use Adam optimizer
    InitialLearnRate=lr, ...                % Slightly higher learning rate
    LearnRateSchedule="piecewise", ...      % Add learning rate decay
    LearnRateDropFactor=0.5, ...
    LearnRateDropPeriod=100, ...
    MaxEpochs=80, ...
    MiniBatchSize=36, ...                   % Add batch size
    Metrics = ["accuracy","fscore"], ...
    Verbose=false, ...
    Plots="training-progress");
reset(dsTrain);
net = trainnet(dsTrain, unetNetwork, @diceCrossEntropyLoss, options);
save(fullfile('./dataset_seg/',dataset_name,'net.mat'),'net');

%% test u-net

% Get image size info
[height, width, numLayers] = size(img);
inputSize = net.Layers(1).InputSize;

% Store all segmentation results
all_masks = false(height, width, numLayers);
all_overlays = cell(numLayers, 1);

fprintf('Step 1: Running U-Net segmentation...\n');
classes = categorical(["object", "background"]);
% First pass: get all segmentation results
for idx = 1:numLayers
    if mod(idx, 10) == 0 || idx == numLayers
        fprintf('Segmenting layer %d/%d\n', idx, numLayers);
    end
    % Extract current layer and preprocess
    currentLayer = img(:,:,idx);
    I = imresize(double(currentLayer), inputSize(1:2));
    I = uint8(255 * rescale(I));
    % Perform semantic segmentation
    [C, scores] = semanticseg(I, net, Classes = classes);
    % Basic morphological processing
    mask = C == "object";
    se_open = strel('disk', 1);
    mask = imopen(mask, se_open);
    se_close = strel('disk', 3);
    mask = imclose(mask, se_close);
    min_area = 3;
    mask = bwareaopen(mask, min_area);
    % Store mask and resize back to original size
    all_masks(:,:,idx) = imresize(mask, [height, width], 'nearest');
    % Store overlay for later display
    C_updated = C;
    C_updated(C == "object") = classes(1);
    C_updated(C ~= "object") = classes(2);
    all_overlays{idx} = labeloverlay(I, C_updated, 'Transparency', 0.91);
end

% Temporal smoothing
smoothed_masks = pca_temporal_smoothing(all_masks, 4, 3);
save(fullfile('./dataset_seg/',dataset_name,'masks.mat'),'smoothed_masks','all_masks');

% Write video
frameRate = 60;
outputVideoFile = fullfile('./dataset_seg/',dataset_name,'segmentation_video_smooth.mp4');
outputTiffFile = fullfile('./dataset_seg/',dataset_name,'segmentation_video_smooth.tif');
generateMaskVideo(img, smoothed_masks, outputVideoFile, frameRate);
generateMaskTiffStack(img, smoothed_masks, outputTiffFile);


%% 3D Mask Volume Rendering and Video Generation

folder_range = 4:23;             % Folder index range
base_name = './dataset_seg/19_fish_600hz_rl_';
num_layers = length(folder_range);  % Number of Z-layers (20 layers)
frame_size = [120, 120];         % XY size
num_frames = 2000;               % Number of time frames

% Read all masks data
fprintf('Reading masks data...\n');
all_masks = zeros(frame_size(1), frame_size(2), num_layers, num_frames);

for i = 1:num_layers
    folder_idx = folder_range(i);
    folder_name = sprintf('%s%d', base_name, folder_idx);
    mat_file = fullfile(folder_name, 'masks.mat');
    
    if exist(mat_file, 'file')
        fprintf('Reading: %s\n', mat_file);
        data = load(mat_file);
        all_masks(:, :, i, :) = data.smoothed_masks;
    else
        warning('File not found: %s', mat_file);
    end
end

fprintf('Data reading complete! Data size: %s\n', mat2str(size(all_masks)));

%% Render a single frame
% Create figure
fig = figure('Position', [100, 100, 800, 600]);
% Clear current axes content (keep figure window)
clf;
test_frame = 100;
volume_data = all_masks(:, :, :, test_frame);
% Pad zeros on all boundaries to ensure a closed surface
padded_volume = padarray(volume_data, [1 1 1], 0, 'both');
% 3D smoothing
smooth_sigma = 1.5; % Standard deviation for 3D Gaussian smoothing
volume_smooth = imgaussfilt3(padded_volume, smooth_sigma);
% Create coordinate grid (note the size change)
[X, Y, Z] = meshgrid(0:size(volume_smooth, 2)-1, ...
                     0:size(volume_smooth, 1)-1, ...
                     0:size(volume_smooth, 3)-1);
% Generate isosurface
[faces, vertices] = isosurface(X, Y, Z, volume_smooth, 0.5);
% Check if a valid isosurface was generated
if ~isempty(faces) && ~isempty(vertices)
    % Render isosurface
    patch_handle = patch('Vertices', vertices, 'Faces', faces, ...
                        'FaceColor', 'red', 'EdgeColor', 'none', ...
                        'FaceAlpha', 0.8);
    isonormals(X, Y, Z, volume_smooth, patch_handle);
end

% Set view and lighting
axis equal tight;
view(76, -23);
camlight; camlight(-80, -10); camlight(80, -10);
lighting gouraud;
material shiny;
set(gca, 'Color', 'black');
grid on; grid minor;
xlabel('X'); ylabel('Y'); zlabel('Z');

% Add title showing current frame index
title(sprintf('Frame: %d/%d', test_frame, 2000));


%% Rendering test
% Create video writer
video_filename = 'isosurface_animation.mp4';
v = VideoWriter(video_filename, 'MPEG-4');
v.FrameRate = 120; % Set frame rate
open(v);

% Create figure
fig = figure('Position', [100, 100, 800, 600]);
% set(fig, 'Color', 'white'); % Set white background for better video quality

% Loop to render each frame
for test_frame = 1:2000
    % Clear current axes content (keep figure window)
    clf;
    
    % Extract 3D data at a single time point
    volume_data = all_masks(:, :, :, test_frame);
    
    % Pad zeros on all boundaries to ensure a closed surface
    padded_volume = padarray(volume_data, [1 1 1], 0, 'both');
    
    % 3D smoothing
    smooth_sigma = 1.5; % Standard deviation for 3D Gaussian smoothing
    volume_smooth = imgaussfilt3(padded_volume, smooth_sigma);
    
    % Create coordinate grid (note the size change)
    [X, Y, Z] = meshgrid(0:size(volume_smooth, 2)-1, ...
                         0:size(volume_smooth, 1)-1, ...
                         0:size(volume_smooth, 3)-1);
    
    % Generate isosurface
    [faces, vertices] = isosurface(X, Y, Z, volume_smooth, 0.5);
    
    % Check if a valid isosurface was generated
    if ~isempty(faces) && ~isempty(vertices)
        % Render isosurface
        patch_handle = patch('Vertices', vertices, 'Faces', faces, ...
                            'FaceColor', 'red', 'EdgeColor', 'none', ...
                            'FaceAlpha', 0.8);
        isonormals(X, Y, Z, volume_smooth, patch_handle);
    end
    
    % Set view and lighting
    axis equal tight;
    view(200, 18);
    camlight; camlight(-80, -10); camlight(80, -10);
    lighting gouraud;
    material shiny;
    set(gca, 'Color', 'black');
    grid on; grid minor;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    
    % Add title showing current frame index
    title(sprintf('Frame: %d/%d', test_frame, 2000));
    % Ensure rendering is finished
    drawnow;
    
    % Capture current frame
    frame = getframe(fig);
    
    % Write to video
    writeVideo(v, frame);
end
% Close video file
close(v);
% Close figure
close(fig);
fprintf('Video saved as: %s\n', video_filename);


%% Save video
function generateMaskVideo(img, smoothed_masks, outputVideoFile, frameRate)
%
%   generateSmoothedMaskVideo(img, smoothed_masks, outputVideoFile, frameRate)
%   creates an MP4 video at outputVideoFile with the given frameRate
%   by overlaying smoothed_masks onto each slice of img.
%
%   Inputs:
%     img             - HxWxD numeric or uint8 array of grayscale frames
%     smoothed_masks  - HxWxD logical or numeric mask array (same size as img)
%     outputVideoFile - String, path to output .mp4 file
%     frameRate       - Numeric, frames per second for the video
%
%   Example:
%     generateSmoothedMaskVideo(img, masks, 'out.mp4', 10);


% Video writer setup
videoWriter = VideoWriter(outputVideoFile, 'MPEG-4');
videoWriter.FrameRate = frameRate;
videoWriter.Quality = 95;
open(videoWriter);

% Determine dimensions and classes
[numRows, numCols, numLayers] = size(img);
inputSize = [numRows, numCols];
% Define class labels: foreground = 1, background = 0
classes = [1, 0];

% Create figure
hFig = figure('Position', [100, 100, 800, 600]);

for idx = 1:numLayers
    if mod(idx, 10) == 0 || idx == numLayers
        fprintf('Generating frame %d/%d\n', idx, numLayers);
    end

    % Extract and normalize image
    currentLayer = img(:,:,idx);
    I = imresize(double(currentLayer), inputSize);
    I = uint8(255 * rescale(I));

    % Resize mask
    smm = imresize(smoothed_masks(:,:,idx), inputSize, 'nearest');

    % Build categorical overlay
    C = repmat(classes(2), inputSize);
    C(smm > 0) = classes(1);
    C = categorical(C);

    % Overlay and display
    B = labeloverlay(I, C, 'Transparency', 0.91);
    imshow(B, 'InitialMagnification', 'fit');
    title(sprintf('%d/%d depth layer (Smoothed)', idx, numLayers));
    drawnow;

    % Write frame
    frame = getframe(hFig);
    writeVideo(videoWriter, frame);
end

% Clean up
close(videoWriter);
fprintf('Output file: %s\n', outputVideoFile);
end
function generateMaskTiffStack(img, smoothed_masks, outputTiffFile)
%
%   generateMaskTiffStack(img, smoothed_masks, outputTiffFile)
%   creates a TIFF stack at outputTiffFile by overlaying smoothed_masks 
%   onto each slice of img.
%
%   Inputs:
%     img             - HxWxD numeric or uint8 array of grayscale frames
%     smoothed_masks  - HxWxD logical or numeric mask array (same size as img)
%     outputTiffFile  - String, path to output .tif file
%
%   Example:
%     generateMaskTiffStack(img, masks, 'out.tif');

% Determine dimensions and classes
[numRows, numCols, numLayers] = size(img);
inputSize = [numRows, numCols];

% Define class labels: foreground = 1, background = 0
classes = [1, 0];

for idx = 1:numLayers
    if mod(idx, 10) == 0 || idx == numLayers
        fprintf('Generating frame %d/%d\n', idx, numLayers);
    end
    
    % Extract and normalize image
    currentLayer = img(:,:,idx);
    I = imresize(double(currentLayer), inputSize);
    I = uint8(255 * rescale(I));
    
    % Resize mask
    smm = imresize(smoothed_masks(:,:,idx), inputSize, 'nearest');
    
    % Build categorical overlay
    C = repmat(classes(2), inputSize);
    C(smm > 0) = classes(1);
    C = categorical(C);
    
    % Overlay
    B = labeloverlay(I, C, 'Transparency', 0.91);
    
    % Write to TIF stack
    if idx == 1
        imwrite(B, outputTiffFile, 'tiff');
    else
        imwrite(B, outputTiffFile, 'tiff', 'WriteMode', 'append');
    end
end

fprintf('Output file: %s\n', outputTiffFile);
end