%%
folderName = fullfile('..');
p = genpath(folderName);
addpath(p);
%% data and params
D_l_1 = 2;
D_l = 3;
M = 4;
X = randn(1,1,D_l_1,M);
W = randn(1,1,D_l_1,D_l); 
S = randn(1);
%% compute block -s|| x - t ||^2
[y, Delta_tilde] = cutom_hbf_norm_forward( X,W,S ); % forward mode (get output)
p = randn(size(y), 'single'); % projection tensor
[dzdx, dzdw, dzds] = cutom_hbf_norm_backward( X,W,S,Delta_tilde,p );
%% Check the derivative numerically
print = 0;
if print
    figure(22) ; clf('reset') ;
    set(gcf, 'name', 'Part 2.2: two layers backrpop') ;
end
%%%Z = proj(X,P) computes the projection Z of tensor X onto P.
%% check numerically dx
% print_err_x = print;
% func = @(ARG_X) proj(p, cutom_hbf_norm_forward( ARG_X,W,S  ) ) ;
% [err_x, dx_numerical, dx] = checkDerivativeNumerically(func, X, dzdx, print_err_x) ;
% err_x_squeeze = squeeze(err_x)
% dx_numerical_squeeze = squeeze(dx_numerical)
% dx_squeeze = squeeze(dx)
% %% check numerically dw
% print_err_w = 0;
% func = @(ARG_W) proj(p, cutom_hbf_norm_forward( X,ARG_W,S  ) ) ;
% [err_w, dw_numerical, dw] = checkDerivativeNumerically(func, W, dzdw, print_err_w) ;
% err_w_squeeze = squeeze(err_w)
% dw_numerical_squeeze = squeeze(dw_numerical)
% dw_squeeze = squeeze(dw)
% %% check numerically ds
% print_err_s = 0;
% func = @(ARG_S) proj(p, cutom_hbf_norm_forward( X,W,ARG_S ) ) ;
% [err_s, ds_numerical, ds] = checkDerivativeNumerically(func, S, dzds, print_err_s) ;
% err_s_squeeze = squeeze(err_s)
% ds_numerical_squeeze = squeeze(ds_numerical)
% ds_squeeze = squeeze(ds)
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