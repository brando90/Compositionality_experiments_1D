function [x y opts] = polygen_n3_d3_v02(opts)
% need opts.nVar, opts.numTrain, opts.numTest

poly_eval_func = @poly_eval_n3_d3_v02;

x = randn(opts.numTrain+opts.numTest,opts.nVar);

for i = 1:size(x,1)
    if mod(i,500) == 0
        disp(['preparing points...  i: ' num2str(i)]);
    end
    y(i) = poly_eval_func(x(i,:)); 
end


