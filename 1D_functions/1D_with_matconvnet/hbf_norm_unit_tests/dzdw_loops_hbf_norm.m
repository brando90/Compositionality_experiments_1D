function [ dzdw_loops ] = dzdw_loops_hbf_norm( X,W,S,p )
W = squeeze(W); % (D^(l-1) x D^(l))
P = squeeze(p); % (D^(l) x M)
X = squeeze(X)'; % (M x D^(l-1))
%% LOOPS
D_l = size(P,1);
dzdw = zeros(1,1,D_l_1,D_l);
for d_l=1:D_l
    for d_l_1=1:D_l_1
        %PX1(d_l_1,d_l) = P(d_l,:)*X(:,d_l_1);
        %PW1(d_l_1,d_l) = W(d_l_1,d_l)*sum(P(d_l,:));
        dzdw(1,1,d_l_1,d_l) = 2*S*(P(d_l,:)*X(:,d_l_1) - W(d_l_1,d_l)*sum(P(d_l,:)));
    end
end
dzdw_loops = squeeze(dzdw);
end