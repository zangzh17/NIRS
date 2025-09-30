function [PSF_approx, basis_psfs, weights] = psf_lowrank(PSFs, K)
% 计算3D PSF的低秩近似
% 输入:
% PSFs: 3D PSF，维度为[nx, ny, nz]
% K: 低秩近似的秩
% 输出:
% basis_psfs: K个基础PSF，维度为[nx, ny, K]
% weights: 权重矩阵，维度为[nz, K]
% 获取维度
[nx, ny, nz] = size(PSFs);
% 重塑PSF为矩阵: [nz, nx * ny]
M = zeros(nz, nx * ny);
for d = 1:nz
    M(d, :) = reshape(PSFs(:, :, d), 1, []);
end
% 执行SVD
[U, S, V] = svd(M, 'econ');
% 截断为秩K
K = min(K, min(size(M))); % 确保K不超过M的秩
U_k = U(:, 1:K);
S_k = diag(S(1:K, 1:K));
V_k = V(:, 1:K);
weights = U_k * diag(S_k); % 将奇异值包含在权重中
% 基础PSF - 空间模式
basis_psfs = zeros(nx, ny, K);
for k = 1:K
    basis_psfs(:, :, k) = reshape(V_k(:, k), [nx, ny]);
end
% PSF的重构
PSF_approx = zeros(nx, ny, nz);
for d = 1:nz
    for k = 1:K
        PSF_approx(:, :, d) = PSF_approx(:, :, d) + weights(d, k) * basis_psfs(:, :, k);
    end
end
end