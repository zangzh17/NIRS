% for版本
function dark_channel = get_dark_channel(image, win_size)
% 修复索引问题的get_dark_channel函数

[m, n, ~] = size(image);

% 确保win_size不会过大
win_size = min(win_size, min(m, n)/2); % 限制窗口大小不超过图像尺寸的一半
win_size = max(3, win_size); % 确保win_size至少为3

pad_size = floor(win_size/2);
padded_image = padarray(image, [pad_size pad_size], Inf);

dark_channel = zeros(m, n); 

% 预先分配数组防止动态大小变化
dark_channel_temp = zeros(m*n, 1);

% 使用更安全的索引计算
for k = 1:m*n       
    row = ceil(k/n); % 行索引
    col = mod(k-1, n) + 1; % 列索引

    % 从padded_image中获取相应的窗口
    r_start = row;
    r_end = row + win_size - 1;
    c_start = col;
    c_end = col + win_size - 1;

    % 确保索引不超出边界
    r_end = min(r_end, r_start + 2*pad_size);
    c_end = min(c_end, c_start + 2*pad_size);

    patch = padded_image(r_start:r_end, c_start:c_end, :);
    dark_channel_temp(k) = min(patch(:));
end
% 填充dark_channel
for k = 1:m*n
    row = ceil(k/n);
    col = mod(k-1, n) + 1;
    dark_channel(row, col) = dark_channel_temp(k);
end
end
