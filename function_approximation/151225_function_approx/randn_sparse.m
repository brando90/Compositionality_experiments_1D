function A = randn_sparse(density,varargin)
A = single(randn(varargin{:}));
B = rand(size(A)) <= density;
A = A .* single(B);
