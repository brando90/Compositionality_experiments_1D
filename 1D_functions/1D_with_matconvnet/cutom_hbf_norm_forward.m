function [ z, Delta_tilde ] = cutom_hbf_norm_forward( X,W,S )
% computes pairwise -S|| x -  W||^2
A = squeeze_for_data(X);% ( M x D^(l-1) )
W = squeeze_keep_dimensions(W); % ( D^(l-1) x D^(l) )
WW = sum(W.^2, 1); % ( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 1 )
XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
% -|x-w|^2 = 2<x,w> - (|x|^2 + |w|^2) = - (|x|^2 + |w|^2) - 2<x,w>)
Delta_tilde = 2*(A*W) - bsxfun(@plus, WW, XX) ; % (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
% -S|x-w|^2
Z = S*( Delta_tilde ); % (M x D^(l))
[M, D_l] = size(Z);
z = zeros(1,1,D_l,M);
z(1,1,:,:) = Z';
end