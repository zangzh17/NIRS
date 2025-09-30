function H = makeRamLakFilters(img_size, f0, a0, tform_list)
    % This function generates Ram-Lak-type 1D filters and applies them
    % to the given image.
    %
    % Parameters:
    %   img_size            - Input image size (2D)
    %   f0                 - Offset frequency
    %   a0                 - Minimum value for the transfer function
    %   fitted_tform_list  - List of transformation matrices
    %   depth_idx          - Depth index for selecting transformation
    %
    % Returns:
    %   H - Output filtered image
    
    % Calculate the transfer function
    rx = abs(fftshift(fftfreq(img_size(2),1/img_size(2)))) * 2;
    TF = rx - f0;
    TF(TF > 1) = 1;
    TF(TF < 0) = a0 - f0;
    TF = TF + f0;
    
    % Initialize the filter matrix
    num_view = size(tform_list,1);
    H = zeros([img_size, num_view]);
    
    % Find matrix center
    center_i = floor(size(H, 1) / 2) + 1;
    center_j = floor(size(H, 2) / 2) + 1;
    [J, I] = meshgrid(1:size(H, 2), 1:size(H, 1));
    
    
    % choose center depth
    depth_idx = ceil(size(tform_list,2)/2);
    for i = 1:num_view
        % Obtain rotation angles
        A = tform_list(i, depth_idx).A;
        theta = -atan2d(A(2, 1), A(1, 1));
        % Interpolate for 2D Ram-Lak-type filter
        x_prime = (I - center_i) * sind(theta) + (J - center_j) * cosd(theta) + length(TF) / 2;
        H0 = interp1(1:length(TF), TF, x_prime, 'nearest', 'extrap');
        % Normalize
        H(:, :, i) = ifftshift(H0) / sum(H0(:)) * prod(img_size);
    end
end