function Y = ifft3(X)
%IFFT3  3D inverse FFT on the first three dimensions of X
%   Y = ifft3(X) 对 X 的前3维执行 3D IFFT (其余维度保持不变)
%
%   等价于:
%       Y = ifft( ifft( ifft(X, [], 1), [], 2 ), [], 3 );

Y = ifft(X, [], 1);
Y = ifft(Y, [], 2);
Y = ifft(Y, [], 3);
end
