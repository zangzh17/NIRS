function P = padPSF3D(PSF, m, n, p)
%PADPSF3D   Pad a 3D PSF array with zeros to make it an m-by-n-by-p array.
%   The output is centered based on PSF's center pixel/voxel.
%
%   P = padPSF3D(PSF, m);          % If m is a 3-element vector [m n p (batch)]
%   P = padPSF3D(PSF, m, n, p);    % If m, n, p are separate scalars
%
% Input:
%   PSF   - 3D array containing the point spread function, size = [m0, n0, p0].
%   m,n,p - Desired dimensions of the padded array. Either pass as
%           three separate scalars, or pass a single vector [m n p].
%
% Output:
%   P     - Padded array of size [m, n, p], with the PSF centered.

    % -------- 参数解析 --------
    if nargin == 2
        % 可能用户调用: padPSF3D(PSF, [m n p])
        if numel(m) == 3
            p = m(3);
            n = m(2);
            m = m(1);
        else
            error('When calling with two arguments, the second must be a 3-element size vector [m n p].');
        end
    end

    % -------- 原PSF大小 --------
    [m0, n0, p0, B] = size(PSF);

    % -------- 创建零填充输出 --------
    P = zeros(m, n, p, B, 'like', PSF);  
    % 'like' 可以确保与PSF数据类型(单/双精度)一致，也可改成 'single'/'double'

    % -------- 将PSF复制到左上后上角 --------
    %   注意索引范围别越界
    P(1:m0, 1:n0, 1:p0, :) = PSF;

    % -------- 计算中心索引，利用 circshift 将PSF中心移到新中心 --------
    center  = floor([m, n, p, 0]   / 2) + 1;    % 新尺寸的“中心”
    center0 = floor([m0, n0, p0, 0]/ 2) + 1;    % 原PSF的“中心”

    shiftAmount = center - center0;  
    % shiftAmount是3个整数构成的向量。例如 [2, -1, 0] 表示在 x 方向+2, y方向-1, z方向不动

    P = circshift(P, shiftAmount);

end
