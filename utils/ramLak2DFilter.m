function outImg = ramLak2DFilter(inImg, alphax, alphay)
%RAMLAK2DFILTER  Apply a 2D Ram-Lak (ramp) filter in frequency domain
%   outImg = ramLak2DFilter(inImg, freqCut, windowType)
%
% 输入:
%   inImg      - 2D 输入图像 (Ny x Nx)
%   freqCut    - 截止频率(0~1之间)，默认=1.0
%                1.0 表示最高可用频率(即图像的Nyquist频率)全保留，
%                <1 时对高频做截断。
%   windowType - 窗口类型(可选字符串):
%                'none', 'hann', 'hamming', 'cosine' 等
%
% 输出:
%   outImg     - 滤波后图像 (同大小)
%
% 说明:
%   1) 构建 2D 频率坐标网格 (从 -0.5~0.5 单位化频率)，计算 Ram-Lak = radialFreq
%      并在 freqCut 之外归零或使用软窗切除。
%   2) 可在 Ram-Lak 上再乘以一个窗口函数, 减弱极高频噪声放大.
%   3) 最后对原图像做 (fft -> shift -> 乘滤波器 -> ifft -> shift)，
%      得到高通滤波结果.
%
%   参考:
%     - 1D Ram-Lak 用于滤波反投影(CT)，此处是 2D 版本(径向斜坡).
%     - 注意高通滤波会增强噪声, 需酌情使用.


% 1) 获取尺寸
[Ny, Nx] = size(inImg);

% 2) 生成频率坐标 (以 -0.5~+0.5 归一化频率为例)
%    fx, fy: 大小与图像相同. 单位频率(周波/像素)
%    meshgrid 里注意 x 对应列方向, y 对应行方向
[fxx, fyy] = meshgrid( ...
    (-floor(Nx/2) : floor((Nx-1)/2)) / Nx, ...
    (-floor(Ny/2) : floor((Ny-1)/2)) / Ny ...
);

% 4) 构建 2D Ram-Lak 核
% Ram-lak ~ radial freq
r = 2*abs(fxx);
Hx = (1 - alphax)*1  +  alphax * r;

r = 2*abs(fyy);
Hy = (1 - alphay)*1  +  alphay * r;

H = Hx.*Hy;

% 6) 频域滤波: fftshift->fft2->multiply->ifft2->ifftshift
%    对输入图像先做 shift + fft2
fIn = fftshift( fft2(inImg) );

%    滤波: 乘以 H (大小相同). 但要注意 H 自身中心也在 (1,1) ~ Nx,Ny
%    我们生成 H 时已经让DC分量在 [1,1]. => 直接乘就行
fOut = fIn .* H;

%    再逆变换
tmp = ifft2( ifftshift(fOut) );

% 7) 取实部(一般有极小虚部浮点误差)
outImg = real(tmp);

end
