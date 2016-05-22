restoredefaultpath;clear;clc;clear;clc;
%% load multilayer libraries
folderName = fullfile('../../../hbf_research_ml_model_library/multilayer_HModel_multivariant_regression');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../../hbf_research_ml_model_library/common');
p = genpath(folderName);
addpath(p);
%% target function
%data_set = './f8D_all_data_set';
data_set = 'f8D_all_data_set_Id_interval';
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
%% Activation funcs
run('./activation_funcs');
%Act = relu_func;
%dAct_ds = dRelu_ds;
Act = sigmoid_func;
dAct_ds = dSigmoid_ds;
lambda = 0;
%%%%%%%%%%%%%%%
%% make 1 hidden NN model
L=2; % 2 layer, 1 hidden layer
nn1_param = struct('Dim', cell(1,L), 'eps', cell(1,L) );
%dim of W and b
D_1 = 8;
nn1_param(1).Dim = [D, D_1];
nn1_param(2).Dim = [D_1, D_out];
%scale of init W
eps = 0.01;
nn1_param(1).eps = eps;
nn1_param(2).eps =eps;
%activation funcs and F
nn1_param(1).Act = Act;
nn1_param(1).dAct_ds = dAct_ds;
nn1_param(1).F = 'F_NO_activation_final_layer';
%regularization
nn1_param(1).lambda = 0;
nn1_param(2).lambda = 0;
%make NN mdl
nn1 = make_NN_model(L, nn1_param);

%%%%%%%%%%%%%%%
%% make 2 hidden NN model
L=3; % 3 layer, 2 hidden layer
nn2_param = struct('Dim', cell(1,L), 'eps', cell(1,L) );
%dim of W and b
D_1 = 4;
D_2 = 2;
nn2_param(1).Dim = [D, D_1];
nn2_param(2).Dim = [D_1, D_2];
nn2_param(3).Dim = [D_2, D_out];
%scale of init W
eps_nn1 = 0.01;
for l=1:L
    nn2_param(l).eps =eps_nn1;
end
%activation funcs and F
for l=1:L-1
    nn2_param(l).Act = Act;
    nn2_param(l).dAct_ds = dAct_ds;
end
nn2_param(1).F = 'F_NO_activation_final_layer';
%regularization
for l=1:L
    nn2_param(l).lambda = 0;
end
%make NN mdl
nn2 = make_NN_model( L, nn2_param);

%%%%%%%%%%%%%%%
%% mdl params for training
sgd_errors_nn1 = 1; % record errors in SGS?
[ step_size_params_nn1, nb_iterations_nn1, batchsize_nn1 ] = step_size_NN1( );
%% mdl params for training
sgd_errors_nn2 = 1; % record errors in SGS?
[ step_size_params_nn2, nb_iterations_nn2, batchsize_nn2 ] = step_size_NN2( );
%% train 1 hidden NN model
% tic
% [ nn1, iteration_errors_train_nn1, iteration_errors_test_nn1 ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train, Y_train, nn1, nb_iterations_nn1, batchsize_nn1, X_test,Y_test, step_size_params_nn1, sgd_errors_nn1);
% time_passed = toc;
% [secs_nn1, minutes_nn1, hours_nn1, ~] = time_elapsed(nb_iterations_nn1, time_passed )
%% train 2 hidden NN model
tic
[ nn2, iteration_errors_train_nn2, iteration_errors_test_nn2 ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train,Y_train, nn2, nb_iterations_nn2,batchsize_nn2, X_test,Y_test, step_size_params_nn2, sgd_errors_nn2 );
time_passed = toc;
[secs_nn2, minutes_nn2, hours_nn2, ~] = time_elapsed(nb_iterations_nn2, time_passed )
%
save('current_results')
%%
beep;