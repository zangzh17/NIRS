function imp = showImageJ(I)
    % Update for your ImageJ2 (or Fiji)
    addpath('C:\Fiji.app\scripts')
    ImageJ; 
    if ndims(I) == 3
        result_ij = permute(I,[2,1,3]);
        result_ij = reshape(result_ij, size(result_ij, 1), size(result_ij, 2), 1, size(result_ij, 3), 1);
    elseif ndims(I) == 2
        I = reshape(I, size(I, 1), size(I, 2), 1);
        result_ij = permute(I, [2,1,3]);
    end
    imp = copytoImagePlus(result_ij,'XYCZT');
    imp.show();
end
