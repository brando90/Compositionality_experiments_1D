function A = poly_eval_general_2_n(x,coeffMat)
for i = 1:size(x,1)
    A(i) = poly_eval_general_2(x(i,:),coeffMat);
end

