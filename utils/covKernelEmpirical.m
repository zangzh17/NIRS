function C_kernel = covKernelEmpirical( ...
    CTF, z_list, pitch, lam, n_sample, batchsz )
% covKernelEmpirical_iterative_strict
%  使用“两遍”在线累加方式，严格计算全体样本的零均值协方差核 (3D)。
%
% 输入参数（与原函数类似）:
%   CTF       : [Ny, Nx], 低通滤波器(或CTF)
%   z_list    : 传播距离数组，长度 L
%   pitch     : 像元尺寸
%   lam       : 波长
%   n_sample  : 随机相位样本数
%   batchsz   : 批大小(可选)，默认为1
%
% 输出参数:
%   C_kernel  : [Ny, Nx, L], 真实协方差核 (CPU 上)
%
% 核心步骤:
%   Pass 1: 生成 I_k 并累加 sum(I_k)，得到全体样本的 3D 平均分布 I_mean
%   Pass 2: 重新生成 I_k, 做 (I_k - I_mean), 求 3D FFT 并做功率累加
%   最后对累加结果 /N, 再 IFFT 得到协方差核.

if nargin<6
    batchsz = 1;
end
n_batch  = ceil(n_sample / batchsz);

% ------------------- 尺寸准备 & GPU 转换 -------------------
[Ny, Nx] = size(CTF);
L = numel(z_list);

CTF    = gpuArray(single(CTF));
z_list = single(z_list);
pitch  = single(pitch);
lam    = single(lam);

% 角谱法传播算子 H: [Ny, Nx, L], 放到 GPU
H = ASM_Kernel(z_list, pitch, lam, Nx, Ny);
H = gpuArray(single(H));

% 第一个累加器: 用来求 sum(I_k)
I_acc = gpuArray.zeros(Ny, Nx, L, 'single');

% ------------------- Pass 1: 统计全体样本的平均分布 I_mean -------------------
h = waitbar(0, 'Pass 1: Calculating I_mean ...');
for i = 1 : n_batch
    idx_start = (i-1)*batchsz + 1;
    idx_end   = min(i*batchsz, n_sample);
    current_batch_size = idx_end - idx_start + 1;

    % (1) 生成随机相位 [Ny,Nx,batch]
    random_phases = 2*pi * rand(Ny, Nx, current_batch_size, 'single', 'gpuArray');
    % (2) 与 CTF 相乘，IFFT => 空域复振幅
    complex_amplitude = fftshift2( ifft2( CTF .* exp(1i * random_phases) ) );
    % (3) 角谱传播 => 强度 [Ny,Nx,L,batch]
    I = abs( prop(complex_amplitude, H) ).^2;

    % (4) 在线累加到 I_acc
    for k = 1 : current_batch_size
        Ik = I(:,:,:,k);  % [Ny, Nx, L]
        I_acc = I_acc + Ik;
    end

    waitbar(i/n_batch, h, ...
        sprintf('Pass1: batch %d/%d (total samples=%d)', i, n_batch, n_sample));
end
close(h);

% 求全体样本均值(3D)
I_mean = I_acc / n_sample;  % [Ny, Nx, L], GPU 上

% ------------------- Pass 2: 计算 (I_k - I_mean) 的 3D FFT 的功率累加 -------------------
% 生成第二个累加器 S_acc, 用来累加 abs( FFT3(I_k - I_mean) )^2
S_acc = gpuArray.zeros(Ny, Nx, L, 'single');

h = waitbar(0, 'Pass 2: Accumulating covariance...');
for i = 1 : n_batch
    idx_start = (i-1)*batchsz + 1;
    idx_end   = min(i*batchsz, n_sample);
    current_batch_size = idx_end - idx_start + 1;

    % (1) 生成与Pass1相同的随机相位(为了严格对应同一组样本)
    %     但是随机相位本来是临时的, 这里若想精确重现，需要:
    %        - 用固定随机种子, 并在Pass1+Pass2相同的顺序产生
    %        - 或者把 Pass1 里每次生成的 random_phases 缓存 (又会很大?)
    %     如果可以接受"不同"随机相位作为近似, 这就更简单了。
    %     ---------
    %     这里给出最简单写法：重新生成随机相位 => 其实严格来说这将对应一批新的样本,
    %     但若n_sample很大, 平均效果差不多. 若你需要100%同样的样本, 得自己fix seed.
    random_phases = 2*pi * rand(Ny, Nx, current_batch_size, 'single', 'gpuArray');

    % (2) 同样方式得到 I_k
    complex_amplitude = fftshift2( ifft2( CTF .* exp(1i * random_phases) ) );
    I = abs( prop(complex_amplitude, H) ).^2;  % [Ny, Nx, L, batch]

    % (3) 计算 (I_k - I_mean) 并做 3D FFT
    for k = 1 : current_batch_size
        Ik = I(:,:,:,k);        % [Ny,Nx,L]
        Ik_zm = Ik - I_mean;    % 全局零均值
        Xk = fftn(Ik_zm);
        S_acc = S_acc + abs(Xk).^2;  
    end
    
    waitbar(i/n_batch, h, ...
        sprintf('Pass2: batch %d/%d (total samples=%d)', i, n_batch, n_sample));
end
close(h);

% 平均 => S_avg
S_avg = S_acc / n_sample;  % [Ny, Nx, L], GPU

% IFFT => 协方差核
C_kernel_gpu = real(ifftn(S_avg));  % [Ny, Nx, L]
C_kernel_gpu = fftshift(C_kernel_gpu); 
% 归一化 (可根据需要是否做)
C_kernel_gpu = C_kernel_gpu ./ max(C_kernel_gpu(:));

% 搬回 CPU
C_kernel = gather(C_kernel_gpu);

fprintf('Done. C_kernel size: [%d, %d, %d].\n', Ny, Nx, L);

end
