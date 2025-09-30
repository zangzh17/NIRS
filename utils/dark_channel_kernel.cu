__global__ void dark_channel_kernel(
    const float* __restrict__ input,  
    float* output,                     
    int height, int width,
    int win_size, int pad_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // 0…m‑1
    int col = blockIdx.y * blockDim.y + threadIdx.y; // 0…n‑1
    if (row >= height || col >= width) return;

    int padded_h = height + 2 * pad_size;

    // ① 把“原图坐标”映射到 padded 坐标
    int center_r = row + pad_size;
    int center_c = col + pad_size;

    int r_start = center_r - pad_size;
    int r_end   = center_r + pad_size;
    int c_start = center_c - pad_size;
    int c_end   = center_c + pad_size;

    float min_val = INFINITY;          // 直接用 IEEE Inf

    for (int r = r_start; r <= r_end; ++r)
        for (int c = c_start; c <= c_end; ++c)
            min_val = fminf(min_val, input[r + c * padded_h]); // 列主序

    output[row + col * height] = min_val; // ② row + col*h
}