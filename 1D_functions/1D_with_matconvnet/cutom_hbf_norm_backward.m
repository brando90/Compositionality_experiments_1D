function [ dzdx, dzdw, dzds ] = cutom_hbf_norm_backward( X,W,S,p )
% computes dzdw
% get gradient matrix dV_dW^(l) for parameters W^(l) at layer l
W = squeeze(W); % (D^(l-1) x D^(l))
P = squeeze(p); % (D^(l) x M)
X = squeeze(X)'; % (M x D^(l-1))
p_sum = sum(P,1)'; % (M x 1) = (1 x M)'= sum(D^(l) x M)'
XP = bsxfun(@times, X, p_sum); % (M x D^(l-1)) =  (M x D^(l-1)) .x (M x 1)
WP = (W * P)';
dx = (-2*S)*(XP - WP); % (M x D^(l-1)) = (M x D^(l-1)) - (M x D^(l)) x (D^(l) x D^(l-1))
[M, D_l_1] = size(dx);
dzdx = zeros(1,1,D_l_1,M); % TODO add singleton dim
dzdx(1,1,:,:) = dx';
%% TODO
D_l = size(P,1);
dzdw = zeros(1,1,D_l_1,D_l);
for d_l=1:D_l
    for d_l_1=1:D_l_1
        dzdw(1,1,d_l_1,d_l) = 2*S*(P(d_l,:)*X(:,d_l_1) - W(d_l_1,d_l)*sum(P(d_l,:)));
    end
end
%% TODO
dzds = nan;
end