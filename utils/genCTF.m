function CTF = genCTF(Nx, Ny, pitch, NA, lam)
% generateCTF 生成低通滤波器 CTF
%
%   CTF = generateCTF(Nx, Ny, pitch, NA, lam) 返回大小为 [Ny, Nx] 的低通滤波器，
%   用于限制空间频率满足 sqrt(FX.^2 + FY.^2) <= NA/lam.
%
% 输入参数：
%   Nx    - x 方向像素数
%   Ny    - y 方向像素数
%   pitch - 像元尺寸（单位：m）
%   NA    - 数值孔径
%   lam   - 波长（单位：m）
%
% 输出参数：
%   CTF   - 低通滤波器（逻辑数组，1 表示通过该频率，0 表示阻断）

% 计算二维频域坐标
fx = fftfreq(Nx, pitch);
fy = fftfreq(Ny, pitch);
[FX, FY] = meshgrid(fx, fy);

% 计算 NA 限制下的最大空间频率
fprintf('Speckle size: %.3f um\n', lam/2/NA*1e6)
fprintf('Depth of focus: %.3f um\n', 2*lam/NA^2*1e6)

% 构造低通滤波器 CTF（限制空间频率低于 NA/lam）
CTF = double(sqrt(FX.^2 + FY.^2) <= (NA / lam));

end
