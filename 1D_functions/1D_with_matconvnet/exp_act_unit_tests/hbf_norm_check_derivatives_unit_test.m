%% exp unit test
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
%% compute block -s|| x - t ||^2
[z, ~] = custom_hbf_norm_forward( X,W,S ); % forward mode (get output)
%% compute block -s|| x - t ||^2
a = custom_exp_forward( z ); % forward mode (get output)
p = randn(size(a), 'single'); % projection tensor
[dadz] = custom_exp_backward( a, p );
%% Check the derivative numerically
print = 0;
if print
    figure(22) ; clf('reset') ;
    set(gcf, 'name', 'Part 2.2: two layers backrpop') ;
end
%%%Z = proj(X,P) computes the projection Z of tensor X onto P.
%% check numerically dx
print_err_z = print;
func = @(ARG_Z) proj(p, custom_exp_forward( ARG_Z  ) ) ;
[err_a, da_numerical, da] = checkDerivativeNumerically(func, z, dadz, print_err_z) ;
err_a_squeeze = squeeze(err_a)
da_numerical_squeeze = squeeze(da_numerical)
da_squeeze = squeeze(da)