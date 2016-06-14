%%
folderName = fullfile('..');
p = genpath(folderName);
addpath(p);
%% data and params
D_l_1 = 2;
D_l = 3;
M = 4;
X = randn(1,1,D_l_1,M);
W = randn(D_l_1,D_l); 
S = randn(1);
%%
y = cutom_hbf_norm_forward( X,W,S ); % forward mode (get output)
p = randn(size(y), 'single'); % projection tensor
[dzdx, dzdw, dzds] = cutom_hbf_norm_backward( X,W,S,p );
%%


% Create a random input image batch
% x = randn(10, 10, 1, 2, 'single') ;
% % Forward mode: evaluate the conv follwed by ReLU
% y = vl_nnconv(x, w, []) ; % (8     8     1     2)
% z = vl_nnrelu(y) ; % (8     8     1     2)
% % Pick a random projection tensor
% p = randn(size(z), 'single') ; % (8     8     1     2)
% % Backward mode: projected derivatives
% dy = vl_nnrelu(z, p) ; % (8     8     1     2)
% [dx,dw] = vl_nnconv(x, w, [], dy) ; 
% %size(dx) = 10    10     1     2
% %size(dw) = 3     3
%% Check the derivative numerically
figure(22) ; clf('reset') ;
set(gcf, 'name', 'Part 2.2: two layers backrpop') ;
%%%Z = proj(X,P) computes the projection Z of tensor X onto P.
func = @(ARG) proj(p, cutom_hbf_norm_forward( ARG,W,S  ) ) ;
checkDerivativeNumerically(func, x, dzdx) ;