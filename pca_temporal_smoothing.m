%% PCA-based temporal sequence denoising
function smoothed_masks = pca_temporal_smoothing(masks, num_components, smooth_factor)
% masks: binary masks of size [height, width, numLayers]
% num_components: number of principal components to retain
% smooth_factor: smoothing strength parameter (optional, default = 5)

    if nargin < 3
        smooth_factor = 5; % default smoothing factor
    end
    
    [h, w, num_layers] = size(masks);
    
    % Reshape 3D masks into 2D matrix: rows = time points, cols = spatial positions
    masks_2d = reshape(masks, h*w, num_layers)'; % [num_layers, h*w]
    
    % Convert to double for PCA computation
    masks_2d = double(masks_2d);
    
    % Compute mean over time for each spatial position
    mean_temporal = mean(masks_2d, 1); % [1, h*w]
    
    % Center the data (subtract temporal mean at each spatial position)
    masks_centered = masks_2d - mean_temporal; % [num_layers, h*w]
    
    % Perform PCA along temporal dimension
    % Compute covariance matrix (covariance between spatial positions)
    cov_matrix = (masks_centered' * masks_centered) / (num_layers - 1); % [h*w, h*w]
    
    % Since covariance matrix may be large, use SVD for PCA
    % Direct SVD on centered data: masks_centered = U * S * V'
    [U, S, V] = svd(masks_centered, 'econ'); % U: [num_layers, min(num_layers, h*w)]
    
    % Columns of U are temporal PCs, columns of V are spatial modes
    % Keep first num_components principal components
    U_reduced = U(:, 1:num_components); % [num_layers, num_components]
    S_reduced = S(1:num_components, 1:num_components); % [num_components, num_components]
    V_reduced = V(:, 1:num_components); % [h*w, num_components]
    
    % Compute principal component coefficients (temporal coefficients)
    temporal_coefficients = U_reduced * S_reduced; % [num_layers, num_components]
    
    % Smooth temporal coefficients
    smoothed_coefficients = zeros(size(temporal_coefficients));
    
    for i = 1:num_components
        % Apply moving average smoothing
        if smooth_factor > 1
            % Create smoothing kernel
            kernel = ones(1, smooth_factor) / smooth_factor;
            % Convolution smoothing, keeping original length
            smoothed_coefficients(:, i) = conv(temporal_coefficients(:, i), kernel, 'same');
        else
            smoothed_coefficients(:, i) = temporal_coefficients(:, i);
        end
    end
    
    % Reconstruct centered data with smoothed coefficients
    reconstructed_centered = smoothed_coefficients * V_reduced'; % [num_layers, h*w]
    
    % Add back temporal mean
    reconstructed_2d = reconstructed_centered + mean_temporal; % [num_layers, h*w]
    
    % Reshape back into original 3D format
    smoothed_masks = reshape(reconstructed_2d', h, w, num_layers);
    
    % Ensure output is binary mask (thresholding)
    % Use 0.5 as threshold, or adaptive threshold
    smoothed_masks = smoothed_masks > 0.5;
    
    % Display some statistics
    singular_values = diag(S_reduced).^2 / (num_layers - 1);
    total_variance = sum(diag(S).^2) / (num_layers - 1);
    variance_ratio = singular_values / total_variance;
    
end