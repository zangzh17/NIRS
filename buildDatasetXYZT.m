function [ds, imds, pxds] = buildDatasetXYZT(img_4d, segmentation_data, label_idx, output_dir, dataset_name, imageSize, varargin)
% Build grayscale normalized dataset for XYZT data with dual-label support
% Inputs:
% img_4d - 4D image data in XYZT format [height, width, num_layers, num_frames]
% segmentation_data - Output from keyframeSegToolDualLabel_XYZT
% label_idx - Which label to process (1 or 2)
% output_dir - Dataset save directory
% dataset_name - Dataset name
% imageSize - Target image size [height, width]
% Optional parameters (name-value pairs) - same as original buildDataset

% Parse input parameters
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

% Create output directory structure
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

% Get segmentation info for specified label
if label_idx == 1
    label_data = segmentation_data.label1;
elseif label_idx == 2
    label_data = segmentation_data.label2;
else
    error('label_idx must be 1 or 2');
end

segmented_frames = label_data.segmented_frames;  % [num_layers x num_frames]
masks = label_data.masks;  % Cell array [num_layers x num_frames]

% Get dimensions
[height, width, num_layers, num_frames] = size(img_4d);

% Count total segmented frames
total_segmented = sum(segmented_frames(:));

if opts.EnableAugmentation
    fprintf('Data augmentation enabled - generating %d augmented versions per frame\n', opts.AugmentationFactor);
    total_images = total_segmented * (1 + opts.AugmentationFactor);
else
    total_images = total_segmented;
end

fprintf('Label %d: Found %d segmented frames across %d layers, generating %d training samples\n', ...
        label_idx, total_segmented, num_layers, total_images);

% Image counter
img_counter = 1;

% Process each segmented frame
for layer_idx = 1:num_layers
    for frame_idx = 1:num_frames
        if segmented_frames(layer_idx, frame_idx)
            % Get current image and mask
            current_img = img_4d(:, :, layer_idx, frame_idx);
            current_mask = masks{layer_idx, frame_idx};
            
            % Skip if mask is empty
            if isempty(current_mask)
                fprintf('Warning: Empty mask at layer %d, frame %d\n', layer_idx, frame_idx);
                continue;
            end
            
            % Preprocess: resize
            processed_img = rescale(double(current_img));
            processed_img = imresize(processed_img, imageSize(1:2), "bilinear");
            processed_mask = imresize(current_mask, imageSize(1:2), "nearest");
            
            % Save original image
            % Pass layer_idx and num_layers to encode depth information
            saveImageAndLabel(processed_img, processed_mask, ...
                imageDir, labelDir, img_counter, imageSize, layer_idx, num_layers);

            img_counter = img_counter + 1;
            
            % Generate augmented images if enabled
            if opts.EnableAugmentation
                for aug_idx = 1:opts.AugmentationFactor
                    [aug_img, aug_mask] = applyDataAugmentation(processed_img, processed_mask, opts);
                    
                    % Save augmented image
                    % Pass layer_idx and num_layers for augmented images too
                    saveImageAndLabel(aug_img, aug_mask, ...
                        imageDir, labelDir, img_counter, imageSize, layer_idx, num_layers);
                    img_counter = img_counter + 1;
                end
            end
        end
    end
    
    if mod(layer_idx, 2) == 0 || layer_idx == num_layers
        fprintf('Processed layer %d/%d (generated %d samples so far)\n', ...
                layer_idx, num_layers, img_counter-1);
    end
end

% Create ImageDatastore
imds = imageDatastore(imageDir);

% Create PixelLabelDatastore
classNames = ["object", "background"];
labelIDs = [1, 0];
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);

% Combine datasets
fprintf('Combining datasets...\n');
ds = combine(imds, pxds);

% Validate dataset
fprintf('Validating dataset...\n');
reset(ds);
if hasdata(ds)
    sample = read(ds);
    img = sample{1};
    label = sample{2};
    
    fprintf('Image format: %s, %s\n', mat2str(size(img)), class(img));
    fprintf('Image value range: [%.3f, %.3f]\n', min(img(:)), max(img(:)));
    fprintf('Label format: %s, %s\n', mat2str(size(label)), class(label));
    fprintf('Label categories: %s\n', strjoin(string(categories(label)), ', '));
    fprintf('Label statistics: %s\n', mat2str(countcats(label(:))));
end

% Save dataset info
dataset_info = struct();
dataset_info.name = dataset_name;
dataset_info.label_idx = label_idx;
dataset_info.num_layers = num_layers;
dataset_info.num_frames = num_frames;
dataset_info.num_segmented_frames = total_segmented;
dataset_info.total_samples = img_counter - 1;
dataset_info.augmentation_enabled = opts.EnableAugmentation;
dataset_info.augmentation_factor = opts.AugmentationFactor;
dataset_info.image_dir = imageDir;
dataset_info.label_dir = labelDir;
dataset_info.augmentation_params = opts;

info_path = fullfile(output_dir, dataset_name, 'dataset_info.mat');
save(info_path, 'dataset_info');

fprintf('\nDataset generation complete!\n');
fprintf('Dataset location: %s\n', fullfile(output_dir, dataset_name));
fprintf('Segmented frames: %d\n', total_segmented);
fprintf('Total training samples: %d\n', img_counter - 1);

% Reset dataset pointer
reset(ds);

    % Nested helper functions
    function saveImageAndLabel(img, mask, imageDir, labelDir, counter, imageSize, layer_idx, num_layers)
        % Normalize image to [0,1]
        img_normalized = rescale(img); % This is Channel 1 (Red)

        % --- Create Z-depth channel (Channel 2 - Green) ---
        % Normalize layer index to [0, 1] range
        if num_layers > 1
            z_normalized = (layer_idx - 1) / (num_layers - 1);
        else
            z_normalized = 0; % Handle case with only one layer
        end
        % Create a constant matrix with the normalized z-value
        z_channel = ones(size(img_normalized), 'like', img_normalized) * z_normalized;
        
        % --- Create the 3-channel image ---
        % Channel 1 (R): Original image
        % Channel 2 (G): Z-depth information
        % Channel 3 (B): Zeroes (placeholder)
        rgb_image = cat(3, img_normalized, z_channel, zeros(size(img_normalized), 'like', img_normalized));
        
        % Generate filenames
        img_filename = sprintf('frame_%06d.png', counter);
        label_filename = sprintf('frame_%06d.png', counter);
        
        % Save 3-channel image
        img_path = fullfile(imageDir, img_filename);
        imwrite(rgb_image, img_path);
        
        % Process mask: create binary labels (0 and 1)
        label_img = uint8(mask > 0);
        
        % Save label image
        label_path = fullfile(labelDir, label_filename);
        imwrite(label_img, label_path);
    end

    function [aug_img, aug_mask] = applyDataAugmentation(img, mask, opts)
        aug_img = img;
        aug_mask = mask;
        
        % 1. Geometric transformations
        tform = getRandomTransform(opts, size(img));
        
        if ~isempty(tform)
            aug_img = imwarp(aug_img, tform, 'OutputView', imref2d(size(img)), ...
                            'FillValues', mean(img(:)), 'Interp', 'linear');
            aug_mask = imwarp(aug_mask, tform, 'OutputView', imref2d(size(mask)), ...
                             'FillValues', 0, 'Interp', 'nearest');
        end
        
        % 2. Flipping
        if rand < opts.FlipProbability
            if rand < 0.5
                aug_img = fliplr(aug_img);
                aug_mask = fliplr(aug_mask);
            else
                aug_img = flipud(aug_img);
                aug_mask = flipud(aug_mask);
            end
        end
        
        % 3. Brightness adjustment
        brightness_delta = opts.BrightnessRange(1) + ...
            (opts.BrightnessRange(2) - opts.BrightnessRange(1)) * rand;
        aug_img = aug_img + brightness_delta;
        
        % 4. Contrast adjustment
        contrast_factor = opts.ContrastRange(1) + ...
            (opts.ContrastRange(2) - opts.ContrastRange(1)) * rand;
        img_mean = mean(aug_img(:));
        aug_img = (aug_img - img_mean) * contrast_factor + img_mean;
        
        % 5. Add Gaussian noise
        if opts.NoiseStd > 0
            noise = randn(size(aug_img)) * opts.NoiseStd;
            aug_img = aug_img + noise;
        end
        
        % Ensure values are in [0,1]
        aug_img = max(0, min(1, aug_img));
        aug_mask = double(aug_mask > 0.5);
    end

    function tform = getRandomTransform(opts, img_size)
        % Generate random geometric transformation
        rotation_angle = opts.RotationRange(1) + ...
            (opts.RotationRange(2) - opts.RotationRange(1)) * rand;
        
        scale_factor = opts.ScaleRange(1) + ...
            (opts.ScaleRange(2) - opts.ScaleRange(1)) * rand;
        
        tx = opts.TranslationRange(1) + ...
            (opts.TranslationRange(2) - opts.TranslationRange(1)) * rand;
        ty = opts.TranslationRange(1) + ...
            (opts.TranslationRange(2) - opts.TranslationRange(1)) * rand;
        
        if abs(rotation_angle) > 0.1 || abs(scale_factor - 1) > 0.01 || abs(tx) > 0.1 || abs(ty) > 0.1
            center_x = img_size(2) / 2;
            center_y = img_size(1) / 2;
            
            cos_theta = cosd(rotation_angle);
            sin_theta = sind(rotation_angle);
            
            a = scale_factor * cos_theta;
            b = -scale_factor * sin_theta;
            c = scale_factor * sin_theta;
            d = scale_factor * cos_theta;
            
            tx_total = (1 - a) * center_x - b * center_y + tx;
            ty_total = -c * center_x + (1 - d) * center_y + ty;
            
            tform_matrix = [a b 0; c d 0; tx_total ty_total 1];
            tform = affine2d(tform_matrix);
        else
            tform = [];
        end
    end
end