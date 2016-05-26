restoredefaultpath;clear;clc;clear;clc;
fprintf('HBF1 vs HBF2');
%% load multilayer libraries
folderName = fullfile('../../../hbf_research_ml_model_library/multilayer_HBF_multivariant_regression');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../../hbf_research_ml_model_library/common');
p = genpath(folderName);
addpath(p);
%% target function
data_set = './f8D_all_data_set';
%data_set = 'f8D_all_data_set_Id_interval';
fprintf('DATA SET: %s \n', data_set);
load(data_set);
D = size(X,2);
D_out = size(Y,2);
nb_samples = size(X,1);
%% make train/test data set
percentage_usage = 0.5;
N = nb_samples * percentage_usage;
X = X(1:N,:);
Y = Y(1:N,:);
percentage_split = 0.7;
N_train = int64(N * percentage_split);
N_test = int64(N * (1 - percentage_split));
X_train = X(1:N_train,:);
Y_train = Y(1:N_train,:);
X_test = X(1:N_test,:);
Y_test = Y(1:N_test,:);
%%
gpu_on = 0;
if gpu_on
    X_train = gpuArray(X_train);
    Y_train = gpuArray(Y_train);
    X_test = gpuArray(X_test);
    Y_test = gpuArray(Y_test);
end
%%%%%%%%%%%%%%%
%% Activation funcs
run('./activation_funcs');
%Act = relu_func;
%dAct_ds = dRelu_ds;
%Act = sigmoid_func;
%dAct_ds = dSigmoid_ds;
Act = gauss_func;
dAct_ds = dGauss_ds;
lambda = 0;
%% make 1 hidden NN model
L=2; % 2 layer, 1 hidden layer
hbf1_param = struct('Dim', cell(1,L), 'eps', cell(1,L) );
%dim of W
D_1 = 8;
hbf1_param(1).Dim = [D, D_1];
hbf1_param(2).Dim = [D_1, D_out];
%gaussian std/precision
%std_gau_hbf1 = 0.5;
%gau_precision_hbf1 = 1/(2*std_gau_hbf1);
a = 0.005;
b = 0.007;
hbf1_param(1).beta =  a + (b-a).*rand(1,1); % a + (b-a).*rand(1,1);
hbf1_param(2).beta = a + (b-a).*rand(1,1); % a + (b-a).*rand(1,1);
%scale of init W
eps_hbf1 = 0.01;
hbf1_param(1).eps = eps_hbf1;
hbf1_param(2).eps =eps_hbf1;
%activation funcs and F
hbf1_param(1).Act = Act;
hbf1_param(1).dAct_ds = dAct_ds;
hbf1_param(1).F = 'F_NO_activation_final_layer';
%regularization
hbf1_param(1).lambda = 0;
hbf1_param(2).lambda = 0;
% initialize
hbf1_param(1).init_method = 't_zeros_plus_eps';
%make NN mdl
hbf1 = make_HBF_model(L, hbf1_param);

%%%%%%%%%%%%%%%
%% make 2 hidden NN model
L=3; % 3 layer, 2 hidden layer
hbf2_param = struct('Dim', cell(1,L), 'eps', cell(1,L) );
%dim of W and b
D_1 = 4;
D_2 = 2;
hbf2_param(1).Dim = [D, D_1];
hbf2_param(2).Dim = [D_1, D_2];
hbf2_param(3).Dim = [D_2, D_out];
%gaussian std/precision
%std_gau_hbf2 = 0.5;
%gau_precision_hbf2 = 1/(2*std_gau_hbf2);
a = 0.005;
b = 0.007;
for l=1:L
    gau_precision_hbf2 = a + (b-a).*rand(1,1); % a + (b-a).*rand(100,1);
    hbf2_param(l).beta = gau_precision_hbf2;
end
%scale of init W
eps_hbf2 = 0.01;
for l=1:L
    hbf2_param(l).eps =eps_hbf2;
end
%activation funcs and F
for l=1:L-1
    hbf2_param(l).Act = Act;
    hbf2_param(l).dAct_ds = dAct_ds;
end
hbf2_param(1).F = 'F_NO_activation_final_layer';
%regularization
for l=1:L
    hbf2_param(l).lambda = 0;
end
% initialize
hbf2_param(1).init_method = 't_zeros_plus_eps';
%make NN mdl
hbf2 = make_HBF_model( L, hbf2_param);

%%%%%%%%%%%%%%%
%% mdl params for training
sgd_errors_hbf1 = 1; % record errors in SGS?
[ step_size_params_hbf1, nb_iterations_hbf1, batchsize_hbf1 ] = step_size_HBF1( );
%% mdl params for training
sgd_errors_hbf2 = 1; % record errors in SGS?
[ step_size_params_hbf2, nb_iterations_hbf2, batchsize_hbf2 ] = step_size_HBF2( );
%% train 1 hidden NN model
tic
%[ hbf1, iteration_errors_train_hbf1, iteration_errors_test_hbf1 ] = multilayer_learn_HBF_MiniBatchSGD( X_train, Y_train, hbf1, nb_iterations_hbf1, batchsize_hbf1, X_test,Y_test, step_size_params_hbf1, sgd_errors_hbf1);
time_passed = toc;
[secs_hbf1, minutes_hbf1, hours_hbf1, ~] = time_elapsed(nb_iterations_hbf1, time_passed )
%% train 2 hidden NN model
tic
[ hbf2, iteration_errors_train_hbf2, iteration_errors_test_hbf2 ] = multilayer_learn_HBF_MiniBatchSGD( X_train,Y_train, hbf2, nb_iterations_hbf2,batchsize_hbf2, X_test,Y_test, step_size_params_hbf2, sgd_errors_hbf2 );
time_passed = toc;
[secs_hbf2, minutes_hbf2, hours_hbf2, ~] = time_elapsed(nb_iterations_hbf2, time_passed );
%
save('current_results_hbf')
%%
beep;