function [ dzdx, dzdw, dzds ] = cutom_hbf_norm_backward( X,W,S,Delta_tilde,p )
% computes derivatives of block
W = squeeze(W); % (D^(l-1) x D^(l))
P = squeeze(p); % (D^(l) x M)
X = squeeze(X)'; % (M x D^(l-1))
%% compute dzdx
p_sum = sum(P,1)'; % (M x 1) = (1 x M)'= sum(D^(l) x M)'
XP = bsxfun(@times, X, p_sum); % (M x D^(l-1)) =  (M x D^(l-1)) .x (M x 1)
WP = (W * P)';
dx = (-2*S)*(XP - WP); % (M x D^(l-1)) = (M x D^(l-1)) - (M x D^(l)) x (D^(l) x D^(l-1))
[M, D_l_1] = size(dx);
dzdx = zeros(1,1,D_l_1,M); % TODO add singleton dim
dzdx(1,1,:,:) = dx';
%% compute dzdw
D_l = size(P,1);
P = P'; % (M x D^(l))
PP = reshape(P,[1, size(P) ]); % (1 x M x D^(l))
T_imj= bsxfun(@times, PP, X' ); % (D^(l-1) x M x D^(l)) = (1 x M x D^(l)) .* (D^(l-1) x M x 1) = (1 x M x D^(l)) .* (D^(l-1) x M)
PX = squeeze( sum(T_imj,2) ); % (D^(l-1) x D^(l))
PW = bsxfun(@times, W, sum(P,1)); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .* (1 x D^(l))
dw = 2*S*(PX - PW); % (D^(l-1) x D^(l))
dzdw = zeros(1,1,D_l_1,D_l); % TODO add singleton dim
dzdw(1,1,:,:) = dw;
%% compute dzds
% A = squeeze(X)';% ( M x D^(l-1) )
% W = squeeze(W);% ( D^(l-1) x D^(l) )
% WW = sum(W.^2, 1); % ( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 1 )
% XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
%%% -|x-w|^2 = 2<x,w> - (|x|^2 + |w|^2) = - (|x|^2 + |w|^2) - 2<x,w>)
%Delta_tilde = 2*(A*W) - bsxfun(@plus, WW, XX) ; % (M x D^(l)) - (M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l)) - (M x D^(l))
dzds = sum( Delta_tilde(:) .* P(:) );
end