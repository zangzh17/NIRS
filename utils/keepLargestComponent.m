function mask_out = keepLargestComponent(mask_in)
    % Keep only the largest connected component
    cc = bwconncomp(mask_in);
    
    if cc.NumObjects == 0
        mask_out = mask_in;
        return;
    end
    
    % Find the largest component
    numPixels = cellfun(@numel, cc.PixelIdxList);
    [~, idx] = max(numPixels);
    
    % Create output mask with only the largest component
    mask_out = false(size(mask_in));
    mask_out(cc.PixelIdxList{idx}) = true;
end