function im = imadj(im,range,gamma_val)

% 获取输入范围参数
low_in = range(1);
high_in = range(2);

% 1. 裁剪到输入范围
im = max(im, low_in);
im = min(im, high_in);

% 2. 归一化到 [0,1] 用于伽马校正
normalized = (double(im) - low_in) / (high_in - low_in);

% 3. 应用伽马校正
gamma_corrected = normalized .^ gamma_val;

% 4. 映射回原始范围
im = gamma_corrected * (high_in - low_in) + low_in;