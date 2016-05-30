%%
mdl_file = 'current_results_nn.mat'
load(mdl_file);
fprintf('DATA SET: %s \n', data_set);
load(data_set);
%%
%low_x = -2*pi;
%high_x = 2*pi;
%nb_samples = 1000;
%X = low_x + (high_x - low_x) * rand(nb_samples,D);
%fp = nn2(1).F(nn2, X);
%F_X = fp(4).A;
X = X_train;
fp = nn2(1).F(nn2, X_train);
F_X = fp(4).A;
plotmatrix(X, F_X);