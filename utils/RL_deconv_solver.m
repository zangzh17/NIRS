function reconResult = RL_deconv_solver(measurements, nIter, forward_fun, backward_fun, show_depth_ratio, mask_roi, mask_obj)
%%%%%%%%
% measurements: 3D array with dimension of [height,width,numViews]
% mask_roi/mask_obj: cropping mask for measurements/final reconstruction
% nIter: number of total iterations
% reconResult: reconstructed 3D cube result; 3D array with dimension of [height,width,numDepths]
%%%%%%%%

if nargin<5
    show_depth_ratio = -1;
end
if nargin<6
    mask_roi = ones(size(measurements),'like',measurements);
end



% init normalization matrix from cropping mask, for option 1
norm_matrix = backward_fun(mask_roi);
% regulate normalization matrix
delta= 1e-6;
norm_matrix(norm_matrix<delta) = delta;
norm_div = norm_matrix; % normalization denominator
norm_div(norm_div<delta) = inf;

% init input paramters, apply ROI mask
measurements = measurements .* mask_roi;
measurements_back = backward_fun(measurements);
if nargin<7
    mask_obj = ones(size(measurements_back),'like',measurements_back);
end
% init guess
% %%% option #1
% estimate = norm_matrix; % Initial guess for the volume

%%% option #2
estimate = measurements_back .* mask_obj;

% normalization
estimate = estimate / sum(estimate, 'all') * sum(measurements, 'all');
estimate(norm_matrix<0.001) = 0;
numDepths = size(estimate,3);

% Create figure for real-time display
if show_depth_ratio>0
    f = figure;
    depth_idx = round(numDepths *show_depth_ratio);
    h = imshow(rescale(double(gather(estimate(:, :, depth_idx)))), []);
    title('Reconstruction Progress');
end

% begin iteration loops
tic
for idxIter=1:nIter
    fprintf('Iter # %d/%d\n', idxIter,nIter);
    %%%% option 1 : O' = O H'(M/(HO))
    % Forward projection step
    reblurred = forward_fun(estimate .* mask_obj);
    % Calculation of back-projection ratio
    ratio = measurements ./ (reblurred + eps);
    % backward, update estimation
    estimate = estimate .* backward_fun(ratio);
    % normalization for cropping mask
    estimate = estimate./norm_div;

    % %%%% option 2 : O' = O H'M/(H'(HO))
    % % Forward projection step
    % reblurred = forward_fun(estimate.* mask_obj) .* mask_roi; % HO: in mea. space
    % % ratio for back-projected
    % ratio = backward_fun(measurements) ./ (backward_fun(reblurred)  + eps); % H'M / H'HO:in obj. space
    % % update estimation
    % estimate = estimate .* ratio .* mask_obj;

    % normalizaton - test
    % estimate = estimate/sum(estimate,"all") * sum(measurements,"all");
    % estimate(norm_matrix<0.001) = 0;
    
    if show_depth_ratio>0
        % Update figure window
        set(h, 'CData', rescale(double(gather(estimate(:, :, depth_idx)))));
        % update progress
        overallProgress = idxIter / nIter;
        elapsedTime = toc;
        remainingTime = elapsedTime / overallProgress * (1 - overallProgress);
        title(sprintf('Iter #%d; Remaining: %.0fs Depth# %d',idxIter,remainingTime,depth_idx))
        drawnow
    end
end
if show_depth_ratio>0
    if isvalid(f)
        close(f)
    end
end
% save results
reconResult = double(gather(estimate));
end