function mask_out = morphoProcessing(mask_in)
    % Advanced morphological post-processing
    
    % Step 1: Remove very small components early
    mask_out = bwareaopen(mask_in, 10);
    
    % Step 2: Morphological closing to connect nearby regions
    se_close = strel('disk', 2);
    mask_out = imclose(mask_out, se_close);
    
    % Step 3: Fill small holes
    mask_out = imfill(mask_out, 'holes');
    
    % Step 4: Morphological opening to remove thin connections
    se_open = strel('disk', 1);
    mask_out = imopen(mask_out, se_open);
    
    % Step 5: Smooth boundaries
    mask_out = activecontour(double(mask_out), mask_out, 50, 'Chan-Vese', ...
                            'SmoothFactor', 2, 'ContractionBias', -0.1);
    mask_out = mask_out > 0.5;
    
    % Step 6: Remove components smaller than threshold
    min_area = 20;  % Adjust based on expected object size
    mask_out = bwareaopen(mask_out, min_area);
end
