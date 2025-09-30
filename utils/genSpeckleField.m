function I_stack = genSpeckleField(CTF, pitch, lam, z_list, num_batch)
% genSpeckleField 利用输入的 CTF 生成 3D 散斑场
%
%   I_stack = genSpeckleField(CTF, pitch, lam)
%
% 输入参数：
%   CTF   - 低通滤波器（CTF），大小为 [Ny, Nx]，用于限制空间频率
%   pitch - 像元尺寸（单位：m）
%   lam   - 波长（单位：m）
%   z_list  - 传播距离列表（单位：m）
%
% 输出参数：
%   I_stack - 归一化后的强度堆栈，大小为 [Ny, Nx, L]
%
% 说明：
%   此函数利用随机相位产生初始散斑场，再结合角谱法（ASM）
%   计算不同传播距离 z 上的散斑场。CTF 是预先生成的低通滤波器，
%   用于限制初始场的频率范围。

if nargin<5
    num_batch = 1;
end

[Ny, Nx] = size(CTF);

% 计算角谱法传播核
H = ASM_Kernel(z_list, pitch, lam, Nx, Ny);
H = gpuArray(H);
CTF = gpuArray(CTF);

% (1) 随机相位放在 GPU 上
random_phases = 2*pi * rand(Ny, Nx, num_batch, 'single', 'gpuArray'); % [0,2π)

% (2) 初始复振幅场 -- 用 CTF 限制频率后做 ifft2
complex_amplitude = fftshift2(ifft2(CTF .* exp(1i * random_phases)));

% (3) 角谱传播[Ny,Nx,B]->[Ny,Nx,Nz,B]
I_stack = abs(prop(complex_amplitude, H)).^2;

% % (4) 零均值
% I = I - mean(I_3D,[1,2,3]);
I_stack = I_stack ./ max(I_stack,[],[1,2,3]);  % 归一化到 [0,1]
I_stack = gather(I_stack);
end
