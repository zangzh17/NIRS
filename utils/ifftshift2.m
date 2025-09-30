function Y = ifftshift2(X)
%IFFTSHIFT2  Shift zero-frequency component to the "corner" in the first 2 dims
%   Y = ifftshift2(X) 只对 X 的前两个维度执行 ifftshift，
%   其余维度(若有)不变。

Y = ifftshift(X, 1);
Y = ifftshift(Y, 2);

end
