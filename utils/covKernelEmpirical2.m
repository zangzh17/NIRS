function corr3D = covKernelEmpirical2(CTF, z_list, pitch, lam, num_samples)
% covKernelEmpirical 计算基于随机相位样本的 3D 强度协方差核（经验核），使用 GPU 加速。

%% Step 0: 准备尺寸、GPU 初始化
[Ny, Nx] = size(CTF);
L = numel(z_list);  % 传播层数


% 如果希望使用 single 精度，可在此处转换
CTF = gpuArray(single(CTF));  % 将输入 CTF 放到 GPU
z_list = single(z_list);      % 同理，如果需要
pitch = single(pitch);
lam   = single(lam);

% 准备存储 3D 强度傅里叶结果的数组 (GPU 上)
S_samples = gpuArray.zeros(Ny, Nx, L, num_samples, 'single');

%% Step 1: 预先计算角谱法传播算子（结果也放在 GPU 上）
H = ASM_Kernel(z_list, pitch, lam, Nx, Ny);  % 假设返回的 H 已经兼容 gpuArray
H = gpuArray(single(H));                    % 若 ASM_Kernel 没有自动转换，可手动转换

% 创建进度条
h = waitbar(0, 'Starting GPU calculations...');

%% Step 2:  第一遍：只做累加
% 这里先初始化存放累加结果的数组
sumI = gpuArray.zeros(Ny, Nx, L, 'single');  
for m = 1 : num_samples
    % 1) 生成第 m 个样本的 3D speckle 场
    % 随机相位放在 GPU 上
    random_phases = 2*pi * rand(Ny, Nx, 'single', 'gpuArray'); % [0,2π)
    % 与 CTF 相乘，逆傅里叶变换到空域 (GPU 上)
    complex_amplitude = fftshift(ifft2(CTF .* exp(1i * random_phases)));
    % 角谱传播 -> 计算强度 -> [Ny, Nx, L]
    I_sample = abs(prop(complex_amplitude, H)).^2;
    % 2) 累加
    sumI = sumI + I_sample;
    % 更新进度条
    waitbar(m/num_samples, h, sprintf('1st: Sample %d / %d', m, num_samples));
end
% 计算平均场
meanI = sumI / num_samples;
%%  Step 3: 第二遍：累加功率谱
sumPS = gpuArray.zeros(Ny, Nx, L, 'single');  % 用来存所有样本的 |F|^2 之和
for m = 1 : num_samples
    % 再次生成第 m 个样本 (确保与第一遍对应的 speckle 相同)
    % 随机相位放在 GPU 上
    random_phases = 2*pi * rand(Ny, Nx, 'single', 'gpuArray'); % [0,2π)
    % 与 CTF 相乘，逆傅里叶变换到空域 (GPU 上)
    complex_amplitude = fftshift(ifft2(CTF .* exp(1i * random_phases)));
    % 角谱传播 -> 计算强度 -> [Ny, Nx, L]
    I_sample = abs(prop(complex_amplitude, H)).^2;
    
    % 扣除平均
    dI = I_sample - meanI; 
    
    % 做 3D FFT
    F = fftn(dI);
    
    % 累加 |F|^2
    sumPS = sumPS + abs(F).^2;
    % 更新进度条
    waitbar(m/num_samples, h, sprintf('2nd: Sample %d / %d', m, num_samples));
end
% 求平均功率谱
avgPS = sumPS / num_samples;

% 计算关联函数 (做 inverse FFT，并除以总体素数)
corr3D = real(ifftn(avgPS)) / (Nx * Ny * L);

% 为了把原点(∆r=0)移到中心，做一个 fftshift
corr3D = fftshift(corr3D);

% 关闭进度条
close(h);

end
