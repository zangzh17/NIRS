function C_kernel_theory = covKernel(CTF, z_list, pitch, lam)
% covKernel 计算理论上的 3D 强度协方差核
%
%   C_kernel_theory = computeIntensityKernel(CTF, z_list, pitch, lam, Nx, Ny)
%
% 输入参数：
%   CTF    - 低通滤波器（CTF），大小为 [Ny, Nx]
%   z_list - 传播距离列表（单位：m）
%   pitch  - 像元尺寸（单位：m）
%   lam    - 波长（单位：m）
% 输出参数：
%   C_kernel_theory - 归一化后的 3D 强度协方差核，大小为 [Ny, Nx, numel(z_list)]
%
% 计算流程：
%   1. 计算焦平面处的 CSF（点扩散函数）。
%   2. 利用 CSF 和角谱法传播核 H 计算 3D 理论场。
%   3. 计算场自相关，再计算得到强度协方差核。

%% Step 1: 计算焦平面处的 CSF（点扩散函数）
CSF0 = fftshift(ifft2(CTF));
CSF0 = CSF0 / max(abs(CSF0(:)));  % 归一化

%% Step 2: 利用 CSF 进行传播得到 3D 理论场
[Ny, Nx] = size(CTF);
H = ASM_Kernel(z_list, pitch, lam, Nx, Ny);
CSF = prop(CSF0, H);

%% Step 3: 计算场自相关，再计算强度协方差核
CSF_fft   = fftn(ifftshift(CSF));
R_E_fft   = abs(CSF_fft).^2;
R_E       = real(fftshift(ifftn(R_E_fft)));
C_kernel_theory = R_E.^2;
C_kernel_theory = C_kernel_theory / max(C_kernel_theory(:));

end
