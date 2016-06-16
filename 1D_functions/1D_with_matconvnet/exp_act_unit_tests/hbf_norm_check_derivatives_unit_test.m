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
%% compute block -s|| x - t ||^2
a = cutom_exp_act_forward( Z ); % forward mode (get output)
p = randn(size(y), 'single'); % projection tensor
[dadz] = cutom_exp_backward( Z, p );
%% Check the derivative numerically
print = 0;
if print
    figure(22) ; clf('reset') ;
    set(gcf, 'name', 'Part 2.2: two layers backrpop') ;
end
%%%Z = proj(X,P) computes the projection Z of tensor X onto P.
%% check numerically dx
print_err_x = print;
func = @(ARG_Z) proj(p, cutom_exp_act_forward( ARG_Z  ) ) ;
[err_a, da_numerical, da] = checkDerivativeNumerically(func, Z, dadz, print_err_z) ;
err_a_squeeze = squeeze(err_a)
da_numerical_squeeze = squeeze(da_numerical)
da_squeeze = squeeze(da)