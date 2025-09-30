function [y, cost, grad] = forward_RIM(x, vx, U_FFT, showProgress)
% forward_RIM 计算目标函数及其（可选）梯度，其中目标函数为
%       J(x) = sum( |forward(x, U) - vx|.^2 ),
%   其中 forward(x, U) = sum_k [ conv3(x, U_k) ]^2.
%
%   如果调用时只要求一个输出参数（如 [cost] = forward_RIM(...)),
%   则只计算 forward operator 和 cost，不计算梯度，以节省计算量.
%
%   梯度推导为：
%       grad = 4 * sum_k ifftn( fftn( ifftshift( r .* f_k ) ) .* conj( U_FFT_k ) ),
%   其中 f_k = conv3(x, U_k) ， r = forward(x, U) - vx.
%
%   输入：
%       x          - 3D 输入数据.
%       vx         - 测量数据，3D 数组，与 x 尺寸相同.
%       U_FFT      - 预先计算好的核的 FFT，4D 数组，尺寸为 [nx, ny, nz, K].
%       showProgress - （可选）是否显示进度条，布尔型，默认为 false.
%

if nargin < 4
    showProgress = false;
end

if showProgress
    hWait = waitbar(0, 'Computing forward operator...');
end

[nx,ny,nz] = size(x);
[nx_pad,ny_pad,nz_pad,K]  = size(U_FFT);

% 对 x 进行 padding, ifftshift 后 FFT
Fx = fftn(ifftshift(padPSF3D(x,[nx_pad,ny_pad,nz_pad])));

% 计算 forward operator：y = sum_k f_k.^2
y = zeros(nx,ny,nz, 'like', x);
if nargout > 1
    % 当需要梯度时，保存每个卷积结果 f_k
    f_all = zeros([nx,ny,nz, K], 'like', x);
end

for k = 1:K
    % 计算卷积 f_k = fftshift3(ifftn(Fx .* U_FFT(:,:,:,k)))
    convResult = fftshift(ifftn(Fx .* U_FFT(:,:,:,k)));
    convResult = cropPSF3D(convResult, [nx,ny,nz]);
    if nargout > 1
        f_all(:,:,:,k) = convResult;
    end
    % 累加各个 f_k 的平方构成 forward(x)
    y = y + abs(convResult).^2;
    
    % if showProgress
    %     waitbar(k / K, hWait, sprintf('Computing forward: kernel %d of %d', k, K));
    % end
end

% 做一次简单的线性缩放，使 y 和 vx 大小相匹配
mean_y = mean(y(:));
mean_vx = mean(vx(:));
alpha = mean_vx / mean_y;  % 或者其它合适的比例
y = alpha * y;


% 计算残差 r = forward(x,U) - vx
r = y - vx;

% 计算 cost = sum(r.^2)
cost = sum(r(:).^2);

wait(gpuDevice);

if nargout > 2
    if showProgress
        waitbar(0.5, hWait, 'Computing gradient...');
    end
    grad = zeros(nx,ny,nz, 'like', x);
    for k = 1:K
        % 计算 r .* f_k，并在频域中与 U_FFT(:,:,:,k) 的共轭相乘，进而反变换回时域
        prod_rf = r .* f_all(:,:,:,k);
        prod_rf = ifftshift(padPSF3D(prod_rf, [nx_pad,ny_pad,nz_pad]));
        grad_k = fftshift(ifftn( fftn(prod_rf).* conj(U_FFT(:,:,:,k)) ));
        grad_k = cropPSF3D(grad_k, [nx,ny,nz]);
        grad = grad + real(grad_k);
        % if showProgress
        %     waitbar(k / K, hWait, sprintf('Computing grad: kernel %d of %d', k, K));
        % end
    end
    grad = alpha * 4 * real(grad);
end

wait(gpuDevice);
if showProgress
    close(hWait);
end
end
