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
dzdw1 = dzdw
%%
PP = reshape(P,[1, flip(size(P)) ]); % (1 x M x D^(l))
T_imj= bsxfun(@times,  PP, X' ); % (D^(l-1) x M x D^(l)) = (1 x M x D^(l)) .* (D^(l-1) x M)
PX = sum(T_imj,2);
PW = bsxfun(@times, W, sum(P,2)'); % (D^(l-1) x D^(l)) = (D^(l-1) x D^(l)) .* (1 x D^(l))
dzdw2 = 2*S*(PX - PW)
err = dzdw1 - dzdw2
%bsxfun(@times,  reshape(P,[1, flip(size(P)) ]), X' ); % () .x ()
%% TODO
dzds = nan;
end