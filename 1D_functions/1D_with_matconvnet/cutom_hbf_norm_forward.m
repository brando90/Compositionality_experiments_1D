function [ Z ] = cutom_hbf_norm_forward( X,W,S )
% computes pairwise -S|| x -  W||^2
A = X;
WW = sum(W.^2, 1); % ( 1 x D^(l-1)= sum( (M x D^(l)), 1 )
XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
% -|x-w|^2 = 2<x,w> - (|x|^2 + |w|^2) = - (|x|^2 + |w|^2) - 2<x,w>)
Delta_tilde = 2*(A*W) - bsxfun(@plus, WW, XX) ; % 
Z = S*( Delta_tilde ); % -S|x-w|^2
end