function forward_handle = linear_convolution_model(PSF, padSize)
%CONVOLUTION_MODEL 创建用于deconvlucy_gpu的循环卷积模型函数句柄
%   [FORWARD_HANDLE] = CONVOLUTION_MODEL(PSF) 返回基于PSF的前向和伴随卷积函数句柄，
%   使用与原始MATLAB deconvlucy函数相同的FFT实现的循环卷积
%
%   [FORWARD_HANDLE] = CONVOLUTION_MODEL(PSF, padSize)
%   为PSF2OTF指定padding大小，默认使用与输入图像小1的大小
%
%   内部使用psf2otf_gpu将PSF转换为光学传递函数(OTF)，然后使用FFT实现卷积
%
%   示例:
%   ------
%   % 创建标准循环卷积模型
%   PSF = fspecial('gaussian', 15, 2);
%   PSF = gpuArray(PSF);
%   [fwd, adj] = convolution_model(PSF);
%
%   % 使用指定输出大小
%   img_size = [256, 256];
%   [fwd, adj] = convolution_model(PSF, img_size);
%
%   参见 PSF2OTF_GPU, DECONVLUCY_GPU

% 确保PSF在GPU上
if ~isa(PSF, 'gpuArray')
    PSF = gpuArray(PSF);
end
if nargin<2
    padSize = gather(size(PSF)) - 1;
end

H = [];

% 创建前向卷积函数句柄（与原始deconvlucy一致）
forward_handle = @(x, adjoint) forward_conv(x, PSF, padSize, adjoint);



    % 前向卷积函数（与原始deconvlucy的逻辑一致）
    function output = forward_conv(input, PSF, padSize, adjoint)
        if nargin<4
            adjoint = false;
        end
        if isa(input, 'gpuArray')
            psfSize = gather(size(input));
        else
            psfSize = size(input);
        end

        % 计算PSF的OTF（一次性计算，后续多次使用）
        if isempty(H)
            H = psf2otf_gpu(PSF, psfSize+padSize);
        end
        % 输入pad
        input = padarray(input, padSize, 'post');
        % Circularly shift so that the "center" is at (1,1)
        input = circshift(input, -floor(psfSize/2));

        if ~adjoint
            output = otf2psf_gpu(H .* fftn(input), psfSize);
        else
            % 执行伴随FFT卷积（使用共轭OTF）
            output = otf2psf_gpu(conj(H) .* fftn(input), psfSize);
        end
    end
end