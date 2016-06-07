function y = poly_eval_n1_pi_balanced_d10(x,dummy)
s1 = RandStream.create('mrg32k3a','seed', 103 ); 
try
    RandStream.setDefaultStream(s1);
catch
    RandStream.setGlobalStream(s1);
end

ref = [-pi:0.01:pi];
degree = 6;  
for i = 1:degree
    coeff(i) =  mean(abs(ref.^i));
end
  
final_coeff = randn(numel(coeff),1) ./ coeff(:);
assert(numel(x) == 1);  
x = x.^(1:degree);
y = final_coeff(:) .* x(:); 
y = sum(y(:));


