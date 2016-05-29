restoredefaultpath;clear;clc;clear;clc;
fprintf('NN1 vs NN2');
%% load multilayer libraries
folderName = fullfile('../../../hbf_research_ml_model_library/multilayer_HModel_multivariant_regression');
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
L=2;
nn_param = struct('eps', cell(1,L) );
%scale of init W
eps_nn = 0.01;
for l=1:L
    nn_param(l).eps =eps_nn;
end
%
nn = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
D_1 = 72;
D_2 = D_out;
nn(1).W = zeros([D, D_1]);
nn(2).W = zeros([D_1, D_2]);
%activation funcs and F
for l=1:L
    if mod(l,2) == 1 % (l mod 2) = (a mod m)
        nn(l).Act = Act;
        nn(l).dAct_ds = dAct_ds;
    else % mod(l,2) == 0 even
        nn(l).Act = Identity;
        nn(l).dAct_ds = dIdentity_ds;
    end
end
%regularization
for l=1:L
    nn(l).lambda = 0;
    nn(l).beta = 0;
end
%% initialize
for l=1:L
    [D_l_1, D_l] = size(nn(l).W);
    if mod(l,2) == 1 % (l mod 2) = (a mod m)
        nn(l).W = nn_param(l).eps * randn([D_l_1, D_l] );
        nn(l).b = nn_param(l).eps * randn([1, D_l] );
        nn(l).Wmask = 1;
        nn(l).bmask = 1;
    else
        nn(l).W = nn_param(l).eps * randn( [D_l_1, D_l] );
        nn(l).b = 0;
        nn(l).Wmask = 1;
        nn(l).bmask = 0;
    end
end
nn(1).F = @F;
nn(1).msg = 'nn1';
nn1 = nn;

%%%%%%%%%%%%%%%
%% make 2 hidden NN model
L=4;
nn_param = struct('eps', cell(1,L) );
%scale of init W
eps_nn = 0.01;
for l=1:L
    nn_param(l).eps =eps_nn;
end
%
nn = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
%dim of W and b
D_1 = 36;
D_2 = 1;
D_3 = 36;
D_4 = D_out;
nn(1).W = zeros([D, D_1]);
nn(2).W = zeros([D_1, D_2]);
nn(3).W = zeros([D_2, D_3]);
nn(4).W = zeros([D_3, D_4]);
%activation funcs and F
for l=1:L
    if mod(l,2) == 1 % (l mod 2) = (a mod m)
        nn(l).Act = Act;
        nn(l).dAct_ds = dAct_ds;
    else % mod(l,2) == 0 even
        nn(l).Act = Identity;
        nn(l).dAct_ds = dIdentity_ds;
    end
end
%regularization
for l=1:L
    nn(l).lambda = 0;
    nn(l).beta = 0;
end
%% initialize
for l=1:L
    [D_l_1, D_l] = size(nn(l).W);
    if nn(l).Act( ones(1) ) == ones(1) %if Act ==  Identity
        nn(l).W = nn_param(l).eps * randn([D_l_1, D_l] );
        nn(l).b = nn_param(l).eps * randn([1, D_l] );
        nn(l).Wmask = 1;
        nn(l).bmask = 1;
    else
        nn(l).W = nn_param(l).eps * randn([D_l_1, D_l] );
        nn(l).b = nn_param(l).eps * randn([1, D_l] );
        nn(l).Wmask = 1;
        nn(l).bmask = 0;
    end
end
nn(1).F = @F;
nn(1).msg = 'nn2';
nn2 = nn;

%% make 3 hidden NN model
L=6;
nn_param = struct('eps', cell(1,L) );
%scale of init W
eps_nn = 0.01;
for l=1:L
    nn_param(l).eps =eps_nn;
end
%
nn = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
%dim of W and b
D_1 = 24;
D_2 = 1;
D_3 = 24;
D_4 = 1;
D_5 = 5;
D_6 = D_out;
nn(1).W = zeros([D, D_1]);
nn(2).W = zeros([D_1, D_2]);
nn(3).W = zeros([D_2, D_3]);
nn(4).W = zeros([D_3, D_4]);
nn(5).W = zeros([D_2, D_3]);
nn(6).W = zeros([D_3, D_4]);
%activation funcs and F
for l=1:L
    if mod(l,2) == 1 % (l mod 2) = (a mod m)
        nn(l).Act = Act;
        nn(l).dAct_ds = dAct_ds;
    else % mod(l,2) == 0 even
        nn(l).Act = Identity;
        nn(l).dAct_ds = dIdentity_ds;
    end
end
%regularization
for l=1:L
    nn(l).lambda = 0;
    nn(l).beta = 0;
end
%% initialize
for l=1:L
    [D_l_1, D_l] = size(nn(l).W);
    if nn(l).Act( ones(1) ) == ones(1) %if Act ==  Identity
        nn(l).W = nn_param(l).eps * randn([D_l_1, D_l] );
        nn(l).b = nn_param(l).eps * randn([1, D_l] );
        nn(l).Wmask = 1;
        nn(l).bmask = 1;
    else
        nn(l).W = nn_param(l).eps * randn([D_l_1, D_l] );
        nn(l).b = 0;
        nn(l).Wmask = 1;
        nn(l).bmask = 0;
    end
end
nn(1).F = @F;
nn(1).msg = 'nn3';
nn3 = nn;

%% number of params
[ num_params_nn1 ] = number_of_params_NN( nn1 )
[ num_params_nn2 ] = number_of_params_NN( nn2 )
[ num_params_nn3 ] = number_of_params_NN( nn3 )

%%%%%%%%%%%%%%%
%% mdl params for training
sgd_errors_nn1 = 1; % record errors in SGS?
[ step_size_params_nn1, nb_iterations_nn1, batchsize_nn1 ] = step_size_NN1( nn1 );
%% mdl params for training
sgd_errors_nn2 = 1; % record errors in SGS?
[ step_size_params_nn2, nb_iterations_nn2, batchsize_nn2 ] = step_size_NN2( nn2 );
%% mdl params for training
sgd_errors_nn3 = 1; % record errors in SGS?
[ step_size_params_nn3, nb_iterations_nn3, batchsize_nn3 ] = step_size_NN3( nn3);
%% train 1 hidden NN model
tic
%[ nn1, iteration_errors_train_nn1, iteration_errors_test_nn1 ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train, Y_train, nn1, nb_iterations_nn1, batchsize_nn1, X_test,Y_test, step_size_params_nn1, sgd_errors_nn1);
time_passed = toc;
num_params_nn1
[secs_nn1, minutes_nn1, hours_nn1, ~] = time_elapsed(nb_iterations_nn1, time_passed )
%% train 2 hidden NN model
tic
[ nn, iteration_errors_train_nn2, iteration_errors_test_nn2 ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train,Y_train, nn2, nb_iterations_nn2,batchsize_nn2, X_test,Y_test, step_size_params_nn2, sgd_errors_nn2 );
time_passed = toc;
num_params_nn2
[secs_nn2, minutes_nn2, hours_nn2, ~] = time_elapsed(nb_iterations_nn2, time_passed )
%% train 3 hidden NN model
tic
%[ nn, iteration_errors_train_nn3, iteration_errors_test_nn3 ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train,Y_train, nn3, nb_iterations_nn3,batchsize_nn3, X_test,Y_test, step_size_params_nn3, sgd_errors_nn3 );
time_passed = toc;
num_params_nn3
[secs_nn2, minutes_nn2, hours_nn2, ~] = time_elapsed(nb_iterations_nn2, time_passed )
%
save('current_results_nn')
%%
beep;