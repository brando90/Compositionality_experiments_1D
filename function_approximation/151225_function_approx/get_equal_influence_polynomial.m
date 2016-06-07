x = [-pi:0.01:pi];
for i = 1:10
    coeff(i) =  mean(abs(x.^i));
end

p = [-pi:0.01:pi];
for i = 1:numel(p)
    y(i) = poly_eval_n1_pi_balanced_d10(p(i));
end
    