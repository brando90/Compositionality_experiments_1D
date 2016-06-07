function [out comp] = poly_eval_spacial_binary_tree_ReLU_v01_n8(x,dummy1)
%  [out comp] = poly_eval_spacial_binary_tree_ReLU_v01_n8(randn(8,1))
n = 8;

assert(numel(x) == n);
assert(size(x,1) == 1 || size(x,2) == 1);
d = log2(n)+1;
comp = NaN(d,numel(x));
comp(1,:) = x(:);
for i = 2:d
    for j = 1:2^(d-i)
        comp(i,j) = max( [comp(i-1,2*(j-1)+1) - comp(i-1,2*j) , 0] );
    end
    assert(2*j+1 > n || isnan( comp(i-1,2*j+1) ));
end

assert( isnan(comp(end,2)) );
assert( ~isnan(comp(end,1)) );
out = comp(end,1);



