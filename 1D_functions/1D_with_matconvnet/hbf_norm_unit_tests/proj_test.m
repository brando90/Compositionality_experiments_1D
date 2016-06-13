%% projection test
X =randn(2,3,4,5);
P = randn(2,3,4,5);
%% 
% Z = proj(X,P) computes the projection Z of tensor X onto P.
Z1 = proj(X,P)
Z2 = X(:)' * P(:)