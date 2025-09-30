% 通用的任意维度padPSF函数
function P = padPSF(PSF, padSize)
%PADPSF 将N维PSF数组填充到指定大小
% P = padPSF(PSF, padSize)
%
% 输入:
% PSF - N维数组
% padSize - 填充后的大小，必须是一个向量，长度与PSF的维度相同或更长
%
% 输出:
% P - 填充后的数组

% 获取原始PSF的大小
origSize = size(PSF);
if isa(PSF, 'gpuArray')
    origSize = gather(origSize);
end

% 确保维度匹配
ndims_max = max(length(origSize), length(padSize));
if length(origSize) < ndims_max
    origSize(end+1:ndims_max) = 1;
end
if length(padSize) < ndims_max
    padSize(end+1:ndims_max) = 1;
end

% 创建填充后的数组
P = zeros(padSize, 'like', PSF);

% 计算中心坐标
centerOrig = floor(origSize / 2) + 1;
centerPad = floor(padSize / 2) + 1;
shift = centerPad - centerOrig;

% 创建原始PSF索引和填充后目标索引
srcIdx = cell(1, ndims_max);
destIdx = cell(1, ndims_max);

for d = 1:ndims_max
    srcIdx{d} = 1:origSize(d);
    destIdx{d} = shift(d) + (1:origSize(d));
    
    % 确保目标索引在有效范围内
    destIdx{d} = min(max(destIdx{d}, 1), padSize(d));
end

% 将原始PSF复制到目标位置
P(destIdx{:}) = PSF(srcIdx{:});

end