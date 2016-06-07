function [x y opts] = polygen_randSparse(opts)
% need opts.nOrder, opts.nVar, opts.density, opts.numTrain, opts.numTest
%

genDim = (opts.nOrder+1)*ones(1,opts.nVar);
coeffMat = randn(genDim);

mult = single(rand(size(coeffMat)) <= opts.density);
coeffMat = coeffMat .* mult;

x = randn(opts.numTrain+opts.numTest,opts.nVar);

for i = 1:size(x,1)
    if mod(i,500) == 0
        disp(['preparing points...  i: ' num2str(i)]);
    end
    y(i) = poly_eval(x(i,:),coeffMat); 
end

opts.coeffMat = coeffMat;
