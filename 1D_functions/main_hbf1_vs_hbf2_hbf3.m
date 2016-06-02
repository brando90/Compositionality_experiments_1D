restoredefaultpath;clear;clc;clear;clc;
fprintf('NN1 vs NN2');
%% load multilayer libraries
folderName = fullfile('../../../hbf_research_ml_model_library/1D_special_HBF_activation_unit');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../../hbf_research_ml_model_library/common');
p = genpath(folderName);
addpath(p);
%% target function
%data_set = './f8D_all_data_set';
%data_set = 'f8D_all_data_set_Id_interval';
data_set = 'f1D_cos_snr_Inf';
fprintf('DATA SET: %s \n', data_set);
load(data_set);
%% make train/test data set
N_train = 60000;
N_test = 60000;
X_train = X_train(1:N_train,:);
Y_train = Y_train(1:N_train,:);
X_test = X_test(1:N_test,:);
Y_test = Y_test(1:N_test,:);
%%
mu_X_train = mean(X_train);
std_X_train = std(X_train);
X_train = bsxfun(@minus, X_train , mu_X_train)/std_X_train;
mu_X_test = mean(X_test);
std_X_test = std(X_test);
X_test = bsxfun(@minus, X_test , mu_X_test)/std_X_test;
%% Activation funcs
run('./activation_funcs');
Act = gauss_func;
dAct_ds = dGauss_ds;
lambda = 0;
%%%%%%%%%%%%%%%
%% make 1 hidden NN model
L=4;
hbf_params = struct('eps', cell(1,L) );
for l=1:L
    hbf_params(l).eps = 0.01;
end
%%
D_1 = 24;
D_2 = D_out;
hbf_params(1).W = zeros([D, D_1]);
hbf_params(2).W = zeros([D_1, D_2]);
%%
for l=1:L
    hbf_params(l).lambda = 0;
    hbf_params(l).beta = 0.001;
    %a = 0.0001;
    %b = 0.00015;
    %beta = 0.01;
    %hbf_params(l).beta = + a + (b-a).*rand(1,1);
end
hbf_params(1).Act = Act;
hbf_params(1).dAct_ds = dAct_ds;
hbf_params(1).init_method = 't_zeros_plus_eps';
hbf1 = make_hbf( hbf_params );
hbf1(1).msg = 'hbf1';

%%%%%%%%%%%%%%%
%% make 2 hidden NN model
L=4;
hbf_params = struct('eps', cell(1,L) );
for l=1:L
    hbf_params(l).eps = 0.01;
end
%%
D_1 = 24;
D_2 = 24;
D_3 = 24;
D_4 = D_out;
hbf_params(1).W = zeros([D, D_1]);
hbf_params(2).W = zeros([D_1, D_2]);
hbf_params(3).W = zeros([D_2, D_3]);
hbf_params(4).W = zeros([D_3, D_4]);
%%
for l=1:L
    hbf_params(l).lambda = 0;
    hbf_params(l).beta = 0.0001;
    %a = 0.0001;
    %b = 0.00015;
    %beta = 0.01;
    %hbf_params(l).beta = + a + (b-a).*rand(1,1);
end
hbf_params(1).Act = Act;
hbf_params(1).dAct_ds = dAct_ds;
hbf_params(1).init_method = 't_zeros_plus_eps';
hbf2 = make_hbf( hbf_params );
hbf2(1).msg = 'hbf2';

% 
% %% make 3 hidden NN model
% L=6;
% nn_params = struct('eps', cell(1,L) );
% nn_params(1).eps = 0.01;
% for l=1:L
%     nn_params(l).eps = 0.05;
% end
% %%
% D_1 = 24;
% D_2 = 1;
% D_3 = 24;
% D_4 = 1;
% D_5 = 5;
% D_6 = D_out;
% nn_params(1).W = zeros([D, D_1]);
% nn_params(2).W = zeros([D_1, D_2]);
% nn_params(3).W = zeros([D_2, D_3]);
% nn_params(4).W = zeros([D_3, D_4]);
% nn_params(5).W = zeros([D_2, D_3]);
% nn_params(6).W = zeros([D_3, D_4]);
% %%
% for l=1:L
%     nn_params(l).lambda = 0;
%     nn_params(l).beta = 0;
% end
% nn_params(1).Act = Act;
% nn_params(1).dAct_ds = dAct_ds;
% nn3 = make_nn( nn_params );
% nn3(1).msg = 'nn3';

%% number of params
%[ num_params_nn1 ] = number_of_params_NN( hbf1 )
% [ num_params_nn2 ] = number_of_params_NN( nn2 )
% [ num_params_nn3 ] = number_of_params_NN( nn3 )

%%%%%%%%%%%%%%%
%% mdl params for training
sgd_errors_hbf1 = 1; % record errors in SGS?
[ step_size_params_hbf1, nb_iterations_hbf1, batchsize_hbf1 ] = step_size_HBF1( hbf1 );
%% mdl params for training
sgd_errors_hbf2 = 1; % record errors in SGS?
[ step_size_params_hbf2, nb_iterations_hbf2, batchsize_hbf2 ] = step_size_HBF2( hbf2 );
% %% mdl params for training
% sgd_errors_nn3 = 1; % record errors in SGS?
% [ step_size_params_nn3, nb_iterations_nn3, batchsize_nn3 ] = step_size_NN3( nn3);
%% GPU
gpu_on = 0;
if gpu_on
    X_train = gpuArray(X_train);
    Y_train = gpuArray(Y_train);
    X_test = gpuArray(X_test);
    Y_test = gpuArray(Y_test);
    [hbf1, step_size_params_hbf1] = put_NN_in_GPU( hbf1 );
end
%% train 1 hidden NN model
%tic
%[ hbf1, iteration_errors_train_hbf1, iteration_errors_test_hbf1 ] = special_multilayer_learn_HBF_MiniBatchSGD( X_train, Y_train, hbf1, nb_iterations_hbf1, batchsize_hbf1, X_test,Y_test, step_size_params_hbf1, sgd_errors_hbf1);
%time_passed = toc;
%num_params_hbf1
%[secs_hbf1, minutes_hbf1, hours_hbf1, ~] = time_elapsed(nb_iterations_hbf1, time_passed )
%% train 2 hidden NN model
tic
[ hbf2, iteration_errors_train_hbf2, iteration_errors_test_hbf2 ] = special_multilayer_learn_HBF_MiniBatchSGD( X_train, Y_train, hbf2, nb_iterations_hbf2, batchsize_hbf2, X_test,Y_test, step_size_params_hbf2, sgd_errors_hbf2);
time_passed = toc;
num_params_hbf2
[secs_hbf2, minutes_hbf2, hours_hbf2, ~] = time_elapsed(nb_iterations_hbf2, time_passed )
%% train 3 hidden NN model
% tic
% %[ nn, iteration_errors_train_nn3, iteration_errors_test_nn3 ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train,Y_train, nn3, nb_iterations_nn3,batchsize_nn3, X_test,Y_test, step_size_params_nn3, sgd_errors_nn3 );
% time_passed = toc;
% num_params_nn3
% [secs_nn2, minutes_nn2, hours_nn2, ~] = time_elapsed(nb_iterations_nn2, time_passed )
%
save('current_results_nn')
%%
beep;