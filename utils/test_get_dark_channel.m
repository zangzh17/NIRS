test_img = imread('cameraman.tif');
ws = 5;
tic
result1 = get_dark_channel(test_img, ws);
toc

tic
result2 = get_dark_channel_gpu(test_img, ws);
toc

figure;
subplot(311);imagesc(result1)
subplot(312);imagesc(result2)
subplot(313);imagesc(result2-result1)