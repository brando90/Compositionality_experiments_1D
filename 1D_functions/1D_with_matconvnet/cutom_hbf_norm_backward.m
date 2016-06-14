function [ dzdx, dzdw, dzds ] = cutom_hbf_norm_backward( X,W,p )
% computes dzdw
% get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
W = squeeze(W);
P = squeeze(p);
p_sum = sum(p,1); % (1 x M)= sum(D^(l) x M)
XP = bsxfun(@times, X, p_sum'); % (M x D^(l-1) =  (M x D^(l-1)) .x (M x 1)
dzdx = (-2*S)*(XP - W * p); % (M x D^(l-1))
%dzdx = zeros(1,1,D,M); % TODO add singleton dim
end