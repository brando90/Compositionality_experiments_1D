restoredefaultpath;clear;clc;clear;clc;
%% load multilayer libraries
folderName = fullfile('../../../hbf_research_ml_model_library/multilayer_HModel_multivariant_regression');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../../hbf_research_ml_model_library/common');
p = genpath(folderName);
addpath(p);
%% target function
data_set = './f8D_all_data_set';
load(data_set);
D = size(X,2);
D_out = size(Y,2);
nb_samples = size(X,1);
%% make train/test data set
percentage_split = 0.8;
N_train = int64(nb_samples * percentage_split);
N_test = int64(nb_samples * (1 - percentage_split));
X_train = X(1:N_train,:);
Y_train = Y(1:N_train,:);
X_test = X(1:N_test,:);
Y_test = Y(1:N_test,:);
%% Activation funcs
run('./activation_funcs');
%Act = relu_func;
%dAct_ds = dRelu_ds;
Act = sigmoid_func;
dAct_ds = dSigmoid_ds;
lambda = 0;
%% make 1 hidden NN model
L=2; % 2 layer, 1 hidden layer
nn1_param = struct('Dim', cell(1,L), 'eps', cell(1,L) );
%dim of W and b
D_1 = 5;
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
%% mdl params for training
sgd_errors = 1;
%nb_iterations = int64(100) % nb_iterations
nb_iterations = int64(10500) % nb_iterations
batchsize = 64
step_size_params_nn1 =  struct('eta_c', cell(1), 'eta_t', cell(1), ...
    'AdaGrad', cell(1), 'Momentum', cell(1), ...
    'Decaying', cell(1), 'step_size', cell(1), 'print_error_to_screen', cell(1) );
step_size_params_nn1.print_error_to_screen = 1;
step_size_params_nn1.Decaying = 1;
step_size_params_nn1.step_size = 0.01;
step_size_params_nn1.decay_rate = 1.5; %if 1 its not decaying then
%% train 1 hidden NN model
tic
[ nn1, iteration_errors_train_nn1, iteration_errors_test_nn1 ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train, Y_train, nn1, nb_iterations, batchsize, X_test,Y_test, step_size_params_nn1, sgd_errors);
time_passed = toc;
[secs_nn1, minutes_nn1, hours_nn1, ~] = time_elapsed(nb_iterations, time_passed )

train_error_nn1 = iteration_errors_train_nn1(nb_iterations);
test_error_nn1 = iteration_errors_test_nn1(nb_iterations);
%plot( 1:nb_iterations+1, [iteration_errors_train_nn1, iteration_errors_test_nn1] );
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
eps = 0.01;
for l=1:L
    nn2_param(l).eps =eps;
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
%% mdl params for training
sgd_errors = 1;
%nb_iterations = int64(10000) % nb_iterations
batchsize = 64
step_size_params_nn2 =  struct('eta_c', cell(1), 'eta_t', cell(1), ...
    'AdaGrad', cell(1), 'Momentum', cell(1), ...
    'Decaying', cell(1), 'step_size', cell(1), 'print_error_to_screen', cell(1) );
step_size_params_nn2.print_error_to_screen = 1;
step_size_params_nn2.Decaying = 1;
step_size_params_nn2.step_size = 0.01;
step_size_params_nn2.decay_rate = 1.5; %if 1 its not decaying then
%% train 2 hidden NN model
tic
[ nn2, iteration_errors_train_nn2, iteration_errors_test_nn2 ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train,Y_train, nn2, nb_iterations,batchsize, X_test,Y_test, step_size_params_nn2, sgd_errors );
time_passed = toc;
[secs_nn2, minutes_nn2, hours_nn2, ~] = time_elapsed(nb_iterations, time_passed )
train_error_nn2 = iteration_errors_train_nn2(nb_iterations);
test_error_nn2 = iteration_errors_test_nn2(nb_iterations);
%%
train_error_nn1
test_error_nn1
train_error_nn2
test_error_nn2
%%
save('current_results')
%%
beep;