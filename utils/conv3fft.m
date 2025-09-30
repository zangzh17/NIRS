function Y = conv3fft(X, H)
%CONV3FFT  Perform 3D convolution of X and H using FFT-based approach
%   Y = conv3fft(X, H)
%
% 输入:
%   X, H   - 两个 3D 数组, 大小为 [nx, ny, nz, (B)]
%
% 输出:
%   Y      - 卷积结果


    %% padding
    [nx, ny, nz, ~] = size(X);
    nx_pad = 2*nx;
    ny_pad = 2*ny;
    nz_pad = 2*nz;
    X = padPSF3D(X, nx_pad, ny_pad, nz_pad);
    H = padPSF3D(H, nx_pad, ny_pad, nz_pad);
    %% 3D FFT
    Y = ifft3(fft3(ifftshift3(X)) .* fft3(ifftshift3(H)));
    Y = fftshift3(Y);
    %% crop
    Y = cropPSF3D(Y, [nx, ny, nz]);
    if isreal(X) && isreal(H)
        Y = real(Y);
    end
end
