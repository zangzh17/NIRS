% 并行化版本的otf2psf处理函数 - 支持任意维度的OTF (3D、4D等)
function psfs = otf2psf_parallel(otfs, outSize)
    % otfs: 多维数组，前两维是空间维度 [height, width, ...]
    % outSize: 2D数组 [height, width]，指定输出PSF的大小
    
    % 获取输入尺寸
    otfSize = size(otfs);
    otfSpatialSize = otfSize(1:2);  % 提取空间维度大小
    
    % 验证输出尺寸小于等于输入尺寸
    if any(outSize > otfSpatialSize)
        error('输出大小不能超过OTF的空间维度大小');
    end
    
    % 确定OTF的维度数
    ndim = ndims(otfs);
    
    % 准备对高维数组进行重塑
    if ndim > 2
        % 获取除空间维度外的所有维度大小
        otherDims = otfSize(3:end);
        
        % 为输出分配内存，保持与输入相同的维度结构
        outputSize = [outSize, otherDims];
        psfs = zeros(outputSize, 'like', otfs);
    else
        % 2D情况 (单个OTF)
        psfs = zeros(outSize, 'like', otfs);
    end
    
    % 检查是否所有OTF都为零
    if ~all(otfs(:) == 0)
        % 对整个多维数组应用ifft2
        temp_psfs = ifft2(otfs);
        
        % 检查并移除小于舍入误差的虚部
        max_real = max(abs(real(temp_psfs(:))));
        max_imag = max(abs(imag(temp_psfs(:))));
        
        % 估计IFFT计算中涉及的大致操作数
        nElem = prod(otfSpatialSize);
        nOps = 0;
        for k = 1:2  % 只考虑空间维度
            nffts = nElem / otfSpatialSize(k);
            nOps = nOps + otfSpatialSize(k) * log2(otfSpatialSize(k)) * nffts;
        end
        
        % 如果虚部在舍入误差范围内，则丢弃
        if max_imag / max_real <= nOps * eps
            temp_psfs = real(temp_psfs);
        end
        
        % 计算移位值 (只在空间维度移位)
        shift_vals = floor(outSize/2);
        
        % 准备移位索引 (只移动空间维度)
        shift_idx = [shift_vals, zeros(1, ndim-2)];
        
        % 循环移位 (只在空间维度移位)
        temp_psfs = circshift(temp_psfs, shift_idx);
        
        % 对多维数组进行裁剪
        % 创建索引元胞数组
        idx = cell(1, ndim);
        for k = 1:ndim
            if k <= 2
                % 对空间维度进行裁剪
                idx{k} = 1:outSize(k);
            else
                % 保留其他维度
                idx{k} = 1:otfSize(k);
            end
        end
        
        % 使用索引元胞数组裁剪多维数组
        psfs = temp_psfs(idx{:});
    end
end