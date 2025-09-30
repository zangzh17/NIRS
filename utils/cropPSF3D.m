function PSF_cropped = cropPSF3D(P, m, n, p)
%CROPPSF3D  Crop a 3D padded PSF array to recover its original 3D PSF.
%
%   PSF_cropped = cropPSF3D(P, [m, n, p])
%   PSF_cropped = cropPSF3D(P, m, n, p)
%
%   给定一个经过 padPSF3D(...) 处理后的 3D PSF 数组 P，和原始 PSF 的尺寸 m-by-n-by-p，
%   该函数将从 P 中裁剪出中心区域，并配合 circshift 的逆移位，以恢复原始的 3D PSF。
%
%   输入:
%     P       - padded 3D PSF 数组, 大小 [M, N, P, (B)].
%     m, n, p - 原始 PSF 的尺寸.
%       或者把 m 写成一个三元素向量 [m n p].
%
%   输出:
%     PSF_cropped - 大小为 [m, n, p] 的裁剪结果.
%
%   注意:
%     - 默认假设在 padPSF3D 中使用的是 "中心对准" 的移位策略，亦即:
%           center = floor([m, n, p]/2) + 1;
%           center0= floor([m0,n0,p0]/2) + 1;
%           shift = center - center0; 
%       如果你使用了不同的移位/对齐方式，请对应修改。
%
%   参考:
%     padPSF3D(PSF, m, n, p)

    %% 1) 参数解析
    if nargin == 2
        % 用户可能调用: cropPSF3D(P, [m n p])
        if numel(m) == 3
            p = m(3);
            n = m(2);
            m = m(1);
        else
            error('When calling cropPSF3D with two arguments, the second must be a 3-element vector [m n p].');
        end
    end

    %% 2) 获取 padded PSF 尺寸
    [Mp, Np, Pp, ~] = size(P);

    %% 3) 计算中心位置
    center  = floor([Mp, Np, Pp]/2) + 1;  % padded后数组中心
    center0 = floor([m,  n,  p ]/2) + 1;  % 原PSF的中心

    % 与 padPSF3D 中 shiftAmount = center - center0 对应
    % 这里一样 computed shift = center - center0
    shift_amount = center - center0;


    %% 4) 确定裁剪区域 (行/列/深度起止)
    % 根据移位量确定裁剪区域
    row_start = shift_amount(1) + 1;
    row_end   = row_start + m - 1;

    col_start = shift_amount(2) + 1;
    col_end   = col_start + n - 1;

    dep_start = shift_amount(3) + 1;
    dep_end   = dep_start + p - 1;

    %% 6) 裁剪
    PSF_cropped = P(row_start:row_end, col_start:col_end, dep_start:dep_end, :);

end
