function Y = fftshift3(X)
%FFTSHIFT3  Shift zero-frequency component to center in the first 3 dims
%   Y = fftshift3(X) 对 X 的前三个维度分别执行 fftshift
%
%   与 fftshift(X, [1 2 3]) 功能相同（只移位前 3 个维度）。

Y = fftshift(X, 1);
Y = fftshift(Y, 2);
Y = fftshift(Y, 3);
end
