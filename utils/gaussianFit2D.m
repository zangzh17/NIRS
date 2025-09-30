function [psf_fit,optimal_params] = gaussianFit2D(psf)
patch_size = size(psf);
[X, Y] = meshgrid(1:patch_size(2), 1:patch_size(1));
% 初始参数估计 [amplitude, x0, y0, sigma_x, sigma_y, theta]
% 使用图像矩来估计中心位置、标准差和角度
psf = psf-min(psf(:));
[x0, y0, sigma_x, sigma_y, theta] = estimateInitialParams(psf, X, Y);
initial_params = [max(psf(:)), x0, y0, sigma_x, sigma_y, theta];
% 使用 fminsearch 进行全局拟合
options = optimset('Display', 'off');
optimal_params = fminsearch(@(p) sum((gaussian2D(p, [X(:), Y(:)]) - psf(:)).^2), initial_params, options);
% 生成拟合的PSF
psf_fit = reshape(gaussian2D(optimal_params, [X(:), Y(:)]), patch_size);
% normalization
psf_fit = psf_fit/sum(psf_fit,'all');
end

function F = gaussian2D(params, xy)
% 高斯函数模型
% params = [amplitude, x0, y0, sigma_x, sigma_y, theta]
A = params(1);
x0 = params(2);
y0 = params(3);
sigma_x = params(4);
sigma_y = params(5);
theta = params(6);

x = xy(:, 1);
y = xy(:, 2);

a = (cos(theta)^2 / (2 * sigma_x^2)) + (sin(theta)^2 / (2 * sigma_y^2));
b = -(sin(2 * theta) / (4 * sigma_x^2)) + (sin(2 * theta) / (4 * sigma_y^2));
c = (sin(theta)^2 / (2 * sigma_x^2)) + (cos(theta)^2 / (2 * sigma_y^2));

F = A * exp(-(a * ((x - x0).^2) + 2 * b * (x - x0) .* (y - y0) + c * ((y - y0).^2)));
end

function [x0, y0, sigma_x, sigma_y, theta] = estimateInitialParams(patch, X, Y)
% 使用图像矩估计初始参数
totalIntensity = sum(patch(:));
x0 = sum(X(:) .* patch(:)) / totalIntensity;
y0 = sum(Y(:) .* patch(:)) / totalIntensity;

x_var = sum((X(:) - x0).^2 .* patch(:)) / totalIntensity;
y_var = sum((Y(:) - y0).^2 .* patch(:)) / totalIntensity;
% xy_cov = sum((X(:) - x0) .* (Y(:) - y0) .* patch(:)) / totalIntensity;

sigma_x = sqrt(x_var);
sigma_y = sqrt(y_var);

% 通过计算图像的二阶中心矩来估计旋转角度theta
mu20 = sum((X(:) - x0).^2 .* patch(:)) / totalIntensity;
mu02 = sum((Y(:) - y0).^2 .* patch(:)) / totalIntensity;
mu11 = sum((X(:) - x0) .* (Y(:) - y0) .* patch(:)) / totalIntensity;

theta = 0.5 * atan2(2 * mu11, mu20 - mu02);
% 处理接近90度的情况
if abs(theta) > pi/4
    theta = theta - sign(theta) * pi/2;
end
end