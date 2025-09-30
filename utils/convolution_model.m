function [forward_handle] = convolution_model(PSF)
%CONVOLUTION_MODEL 创建用于deconvlucy_gpu的循环卷积模型函数句柄
%   [FORWARD_HANDLE] = CONVOLUTION_MODEL(PSF) 返回基于PSF的前向和伴随卷积函数句柄，
%   使用与原始MATLAB deconvlucy函数相同的FFT实现的循环卷积
%
%   [FORWARD_HANDLE] = CONVOLUTION_MODEL(PSF, OUTPUTSIZE)
%   为PSF2OTF指定输出大小，默认使用与输入图像相同的大小
%
%   内部使用psf2otf_gpu将PSF转换为光学传递函数(OTF)，然后使用FFT实现卷积
%
%   示例:
%   ------
%   % 创建标准循环卷积模型
%   PSF = fspecial('gaussian', 15, 2);
%   PSF = gpuArray(PSF);
%   [fwd] = convolution_model(PSF);
%
%   % 使用指定输出大小
%   img_size = [256, 256];
%   [fwd] = convolution_model(PSF, img_size);
%
%   参见 PSF2OTF_GPU, DECONVLUCY_GPU

% 确保PSF在GPU上
if ~isa(PSF, 'gpuArray')
    PSF = gpuArray(PSF);
end

H = [];
% 创建前向卷积函数句柄（与原始deconvlucy一致）
forward_handle = @(x, adjoint) forward_conv(x, PSF, adjoint);


% 前向卷积函数（与原始deconvlucy的逻辑一致）
    function output = forward_conv(input, PSF, adjoint)
        % 获取输入尺寸
        if nargin<3
            adjoint = false;
        end
        if isa(input, 'gpuArray')
            outputSize = gather(size(input));
        else
            outputSize = size(input);
        end
        % 计算PSF的OTF（一次性计算，后续多次使用）
        if isempty(H)
            H = psf2otf_gpu(PSF, outputSize);
        end

        if ~adjoint
            % 执行FFT卷积
            output = real(ifftn(H .* fftn(input)));
        else
            % 执行伴随FFT卷积（使用共轭OTF）
            output = real(ifftn(conj(H) .* fftn(input)));
        end
    end
end