function denoised_data = deepcad_denoise(input_data, denoise_model,deepcad_path)
    % DEEPCAD_DENOISE 使用DeepCAD模型对3D数据进行降噪
    %
    % 输入:
    %   input_data - 3D矩阵数据 (H x W x T)
    %   denoise_model - 模型名称字符串 (例如: 'mouse_202506142239')
    %
    % 输出:
    %   denoised_data - 降噪后的3D矩阵数据
   
    
    % 转换为uint16以节省空间
    if ~isa(input_data, 'uint16')
        input_data = uint16(65535*rescale(input_data));
    end
    
    % 生成唯一的临时文件名
    temp_id = char(java.util.UUID.randomUUID);
    temp_id = strrep(temp_id, '-', '');
    
    % 创建临时目录
    temp_dir = fullfile(deepcad_path,'datasets', ['temp_' temp_id]);
    if ~exist(temp_dir, 'dir')
        mkdir(temp_dir);
    end
    output_dir = fullfile(deepcad_path,'results', ['temp_' temp_id]);
    if ~exist(temp_dir, 'dir')
        mkdir(output_dir);
    end
    
    % 临时输入文件路径
    temp_input_file = fullfile(temp_dir, 'input.tif');
    
    try
        % 保存3D数据为tif文件
        fprintf('Saving...\n');
        save_3d_tiff(input_data, temp_input_file);
        
        % 调用Python降噪脚本
        % 设置patch size
        % patch_t = min(150,size(input_data,3));
        fprintf('正在运行降噪处理...\n');
        pyrunfile(fullfile(deepcad_path,'denoise_interface.py'), ...
                input_tif_path=fullfile('datasets', ['temp_' temp_id]), ...
                deepcad_project_path=deepcad_path, ...
                denoise_model=denoise_model,...
                output_dir = output_dir);
        % 查找输出文件
        output_files = dir(fullfile(output_dir, '*.tif'));
        
        if isempty(output_files)
            error('未找到降噪输出文件');
        end
        
        % 读取第一个输出文件（通常只有一个）
        output_file = fullfile(output_dir, output_files(1).name);
        denoised_data = rescale(double(tiffreadVolume(output_file)));
        
        % 清理临时文件和输出文件
        delete(temp_input_file);
        if exist(temp_dir, 'dir')
            rmdir(temp_dir, 's');
        end
        if exist(output_dir, 'dir')
            rmdir(output_dir, 's');
        end
        
        fprintf('降噪完成！\n');
        
    catch ME
        % 出错时也要清理临时文件
        if exist(temp_input_file, 'file')
            delete(temp_input_file);
        end
        if exist(temp_dir, 'dir')
            rmdir(temp_dir, 's');
        end
        if exist(output_dir, 'dir')
            rmdir(output_dir, 's');
        end
        rethrow(ME);
    end
end

function save_3d_tiff(data, filename)
    % 保存3D数据为TIFF文件
    % 使用ImageJ兼容的格式
    
    [H, W, T] = size(data);
    
    % 创建TIFF对象
    t = Tiff(filename, 'w');
    
    % 设置第一帧的标签
    tagstruct = struct();
    tagstruct.ImageLength = H;
    tagstruct.ImageWidth = W;
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 16;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Software = 'MATLAB';
    tagstruct.ImageDescription = sprintf('ImageJ=1.0\nimages=%d\nslices=%d', T, T);
    
    % 写入所有帧
    for i = 1:T
        t.setTag(tagstruct);
        t.write(data(:,:,i));
        if i < T
            t.writeDirectory();
        end
    end
    
    t.close();
end