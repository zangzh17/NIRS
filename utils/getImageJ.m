function I = getImageJ(imp,sz)

if length(sz)>2
    num_z = sz(3);
else
    num_z = 1;
end

I = zeros(sz(1),sz(2),num_z);

imp.setT(1);imp.setC(1)
for z = 1:size(imgcube,3)
    imp.setZ(z)
    ip = imp.getProcessor();
    I(:,:,z) = ip.getFloatArray()';
end
end