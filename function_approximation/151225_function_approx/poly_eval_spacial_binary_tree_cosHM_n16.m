function [out comp] = poly_eval_spacial_binary_tree_cosHM_n16(x,dummy1)
%  out = poly_eval_spacial_binary_tree_cosHM_n16(randn(16,1))
n = 16;
h = [];
for i = 1:4
    h(i) = cos(  x(1+(i-1)*3) + 2.2*x(2+(i-1)*3) - 0.7*x(3+(i-1)*3) - 1.6 );
end
out = cos(3*h(1) - h(2) - 2.3*h(3) +  0.8*h(4) - 1);





