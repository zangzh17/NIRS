function Y = fft3(X)
%FFT3  3D FFT on the first three dimensions of X
%   Y = fft3(X) 对 X 的第1维、第2维、第3维分别执行 FFT (不改变其他维度)
%
%   等价于:
%       Y = fft( fft( fft(X, [], 1), [], 2 ), [], 3 );
%
%   如果 X 的大小是 [Nx, Ny, Nz, ...]，则只对 (x,y,z) 维做变换。

Y = fft(X, [], 1);      % 沿第 1 维做 FFT
Y = fft(Y, [], 2);      % 沿第 2 维做 FFT
Y = fft(Y, [], 3);      % 沿第 3 维做 FFT
end
