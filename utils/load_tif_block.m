function img_3d = load_tif_block(folderpath, filename)
    % 获取文件信息
    fullname = fullfile(folderpath, filename);
    % 预分配内存
    img_3d = tiffreadVolume(fullname);
end