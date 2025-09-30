function Y = ifftshift3(X)
%IFFTSHIFT3  Shift zero-frequency component to the "corner" in the first 3 dims
%   Y = ifftshift3(X) 对 X 的前三个维度分别执行 ifftshift

Y = ifftshift(X, 1);
Y = ifftshift(Y, 2);
Y = ifftshift(Y, 3);
end
