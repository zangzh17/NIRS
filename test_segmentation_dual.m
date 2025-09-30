%% load
addpath('./utils/');
dataset_name = 'record_02092025_183907_rl_9';
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

% % load and apply exsiting crop patameters (if already saved)
% img = cropImageWithCenter(im, config_file);

fprintf('Data loaded!\n');

%% 1. CALLING THE SEGMENTATION TOOL
% Run dual-label segmentation
segmentation_data = keyframeSegToolDualLabel(img);
seg_file = fullfile(dataset_path, 'segmentation_data_dual.mat');
save(seg_file, "segmentation_data");

%% 2. GENERATE DATASETS FOR BOTH LABELS
% Generate dataset for Label 1
imageSize = [128, 128, 1];

% Dataset for Label 1
[dsTrain_label1, ~, ~] = buildDataset(img, segmentation_data.label1, ...
    fullfile(dataset_path, 'label1'), [dataset_name '_label1'], imageSize, ...
    'EnableAugmentation', true, ...
    'AugmentationFactor', 8, ...
    'RotationRange', [-3.5, 3.5], ...
    'FlipProbability', 0, ...
    'ScaleRange', [0.8, 1.1], ...
    'TranslationRange', [-7, 7], ...
    'BrightnessRange', [-0.1, 0.1], ...
    'ContrastRange', [0.85, 1.15], ...
    'NoiseStd', 0);

% Dataset for Label 2
[dsTrain_label2, ~, ~] = buildDataset(img, segmentation_data.label2, ...
    fullfile(dataset_path, 'label2'), [dataset_name '_label2'], imageSize, ...
    'EnableAugmentation', true, ...
    'AugmentationFactor', 8, ...
    'RotationRange', [-3.5, 3.5], ...
    'FlipProbability', 0, ...
    'ScaleRange', [0.8, 1.1], ...
    'TranslationRange', [-7, 7], ...
    'BrightnessRange', [-0.1, 0.1], ...
    'ContrastRange', [0.85, 1.15], ...
    'NoiseStd', 0);

%% 3. INITIALIZE AND TRAIN TWO U-NETS
lr = 5e-5;

% Initialize U-Net for Label 1
unetNetwork_label1 = unet(imageSize, 2, EncoderDepth=3);

% Initialize U-Net for Label 2
unetNetwork_label2 = unet(imageSize, 2, EncoderDepth=3);

% Define combined loss function
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

% Training options
options = trainingOptions("adam", ...
    InitialLearnRate=lr, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.5, ...
    LearnRateDropPeriod=100, ...
    MaxEpochs=80, ...
    MiniBatchSize=36, ...
    Metrics = ["accuracy","fscore"], ...
    Verbose=false, ...
    Plots="training-progress");

% Train U-Net for Label 1
fprintf('Training U-Net for Label 1...\n');
reset(dsTrain_label1);
net_label1 = trainnet(dsTrain_label1, unetNetwork_label1, @diceCrossEntropyLoss, options);
save(fullfile(dataset_path, 'label1', 'net_label1.mat'), 'net_label1');

% Train U-Net for Label 2
fprintf('Training U-Net for Label 2...\n');
reset(dsTrain_label2);
net_label2 = trainnet(dsTrain_label2, unetNetwork_label2, @diceCrossEntropyLoss, options);
save(fullfile(dataset_path, 'label2', 'net_label2.mat'), 'net_label2');

%% load existing u-nets
dataset_name = 'record_02092025_183907_rl_10';
dataset_path = fullfile(pwd,'dataset_seg',dataset_name);
load(fullfile(dataset_path, 'label1', 'net_label1.mat'));
load(fullfile(dataset_path, 'label2', 'net_label2.mat'));

%% 4. TEST BOTH U-NETS AND CREATE OVERLAID VIDEO


% Get image info
[height, width, numLayers] = size(img);
inputSize = net_label1.Layers(1).InputSize;

% Store segmentation results for both labels
all_masks_label1 = false(height, width, numLayers);
all_masks_label2 = false(height, width, numLayers);

fprintf('Running dual-label U-Net segmentation...\n');
classes = categorical(["object", "background"]);

% Process each frame
for idx = 1:numLayers
    if mod(idx, 10) == 0 || idx == numLayers
        fprintf('Processing frame %d/%d\n', idx, numLayers);
    end
    
    % Extract and preprocess current frame
    currentLayer = img(:,:,idx);
    I = imresize(double(currentLayer), inputSize(1:2));
    I = uint8(255 * rescale(I));
    
    % Segment with Label 1 network
    [C1, scores1] = semanticseg(I, net_label1, Classes = classes);
    mask1 = C1 == "object";
    % Morphological processing for Label 1
    se_open = strel('disk', 1);
    mask1 = imopen(mask1, se_open);
    se_close = strel('disk', 3);
    mask1 = imclose(mask1, se_close);
    mask1 = bwareaopen(mask1, 3);
    all_masks_label1(:,:,idx) = imresize(mask1, [height, width]);
    
    % Segment with Label 2 network
    [C2, scores2] = semanticseg(I, net_label2, Classes = classes);
    mask2 = C2 == "object";
    % Morphological processing for Label 2
    mask2 = imopen(mask2, se_open);
    mask2 = imclose(mask2, se_close);
    mask2 = bwareaopen(mask2, 3);
    all_masks_label2(:,:,idx) = imresize(mask2, [height, width]);
    
end

% Apply temporal smoothing to both masks
fprintf('Applying temporal smoothing...\n');
smoothed_masks_label1 = pca_temporal_smoothing(all_masks_label1, 4, 3);
smoothed_masks_label2 = pca_temporal_smoothing(all_masks_label2, 4, 3);

% Save results
save(fullfile(dataset_path, 'dual_masks.mat'), ...
     'smoothed_masks_label1', 'smoothed_masks_label2', ...
     'all_masks_label1', 'all_masks_label2');

%% 5. GENERATE DUAL-LABEL VIDEO WITH HIGH QUALITY OVERLAY
frameRate = 60;
outputVideoFile = fullfile(dataset_path, 'dual_segmentation_video_2.mp4');
outputTiffFile = fullfile(dataset_path, 'dual_segmentation_stack_2.tif');

% Create video writer with high quality settings
v = VideoWriter(outputVideoFile, 'MPEG-4');
v.FrameRate = frameRate;
v.Quality = 100;  % Maximum quality
open(v);

% Create figure for high-resolution rendering
fig = figure('Position', [100, 100, 800, 600], 'Color', 'black');
ax = axes('Position', [0 0 1 1]); % Full figure axes
axis off;

fprintf('Creating high-quality dual-label video...\n');

for idx = 1:numLayers
    if mod(idx, 30) == 0 || idx == numLayers
        fprintf('Writing frame %d/%d\n', idx, numLayers);
    end
    
    % Get current frame and masks
    frame = img(:,:,idx);
    mask1 = smoothed_masks_label1(:,:,idx);
    mask2 = smoothed_masks_label2(:,:,idx);
    
    % % Method 1: with overlap
    % % Display the base frame first
    % imshow(frame, 'Parent', ax);
    % % Add filled contours for both masks
    % if any(mask1(:)) || any(mask2(:))
    %     hold(ax, 'on');
    %     % Add red filled contour with red edge for mask1
    %     if any(mask1(:))
    %         contourf(ax, mask1, [0.5 0.5], '--', 'FaceColor', 'r', 'EdgeColor', 'r', ...
    %             'FaceAlpha', 0.15, 'LineWidth', 1);
    %     end
    %     % Add blue filled contour with blue edge for mask2
    %     if any(mask2(:))
    %         contourf(ax, mask2, [0.5 0.5], '--', 'FaceColor', 'b', 'EdgeColor', 'b', ...
    %             'FaceAlpha', 0.15, 'LineWidth', 1);
    %     end
    %     hold(ax, 'off');
    % end

    % Method 2: no overlap
    % Create combined label matrix
    combined_labels = zeros(size(frame));
    combined_labels(mask1) = 1;
    combined_labels(mask2) = 2;
    % Create overlay with custom colormap
    overlay_img = labeloverlay(frame, combined_labels, ...
        'Colormap', [1 0 0; 0 0 1], ...  % Red for label 1, Blue for label 2
        'Transparency', 0.93);
    % Display and capture frame
    imshow(overlay_img, 'Parent', ax);
    
    % % Method 3: can overlap, only overlay
    % overlay1 = labeloverlay(frame, mask1, 'Colormap', [1 0 0], 'Transparency', 0.9);
    % overlay2 = labeloverlay(frame, mask2, 'Colormap', [0 0 1], 'Transparency', 0.9);
    % % Blend them (this preserves both colors in overlaps)
    % overlay_img = 0.5 * overlay1 + 0.5 * overlay2;
    % % Display and capture frame
    % imshow(overlay_img, 'Parent', ax);

    % % Alternative: Add contours only
    % if any(mask1(:)) || any(mask2(:))
    %     hold(ax, 'on');
    %     % Add contours
    %     if any(mask1(:))
    %         h =contour(ax, mask1, [0.5 0.5], 'r-', 'LineWidth', 1);
    %         h.EdgeAlpha = 0.8;
    %     end
    %     if any(mask2(:))
    %         h = contour(ax, mask2, [0.5 0.5], 'b-', 'LineWidth', 1);
    %         h.EdgeAlpha = 0.8;
    %     end
    % 
    %     hold(ax, 'off');
    % end
    
    % Capture high-quality frame
    drawnow;
    frame_data = getframe(fig);
    
    % Write to video
    writeVideo(v, frame_data);
    
    % Save to TIFF stack (convert RGB back to match original format)
    if idx == 1
        imwrite(frame_data.cdata, outputTiffFile);
    else
        imwrite(frame_data.cdata, outputTiffFile, 'WriteMode', 'append');
    end
end

close(v);
close(fig);

fprintf('High-quality dual-label video saved to: %s\n', outputVideoFile);
fprintf('High-quality TIFF stack saved to: %s\n', outputTiffFile);
