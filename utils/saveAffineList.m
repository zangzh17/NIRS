function saveAffineList(affineList, filename)
% 假设 affineList 为一个 affinetform2d 对象数组
n = numel(affineList);
tforms = cell(size(affineList));
for i = 1:n
    tforms{i} = affineList(i).A;
end
% 保存到当前路径下的 affineTransforms.mat 文件
[pathname,file,~] =fileparts(filename);
filename = fullfile(pathname,strcat(file,"_tforms"));
save(filename, 'tforms');