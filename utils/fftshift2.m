function Y = fftshift2(X)
%FFTSHIFT2  Shift zero-frequency component to center in the first 2 dims
%   Y = fftshift2(X) 只对 X 的前两个维度执行 fftshift，
%   其余维度(若有)不变。
%
%   与在较新版本中使用 fftshift(X, [1 2]) 等价。

% 对第 1 维做 shift
Y = fftshift(X, 1);
% 再对第 2 维做 shift
Y = fftshift(Y, 2);

end
