clear;clc;clear;clc;
%% load multilayer libraries
folderName = fullfile('../../../hbf_research_ml_model_library/multilayer_HModel_multivariant_regression');
p = genpath(folderName);
addpath(p);
folderName = fullfile('../../../hbf_research_ml_model_library/common');
p = genpath(folderName);
addpath(p);
%% target function
h11 = @(A) (1/10)*(1*A(1) + 2*A(2))^4; % ( x1 + x2)
h12 = @(A) (1/10)*(3*A(1) + 4*A(2))^3;
h21 = @(A) (1/100)*(5*A(1) + 6*A(2))^2;
f_target = struct('h', cell(2,2), 'f', cell(2,2));
f_target(1,1).h = h11;
f_target(1,2).h = h12;
f_target(2,1).h = h21;
f_target(1,1).f = @f_4D;
%% make data set
sigpower = 'measured';
powertype = 'linear';
snr = 8;
low_x = -2;
high_x = 2;
nb_samples = 1000;
[X,Y] = generate_data_from_function( f_target, snr, low_x,high_x, nb_samples, sigpower, powertype);
D = size(X,2);
D_out = size(Y,2);
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
mdl_param = struct('Dim', cell(1,L), 'eps', cell(1,L) );
%dim of W and b
D_1 = 1000;
mdl_param(1).Dim = [D, D_1];
mdl_param(2).Dim = [D_1, D_out];
%scale of init W
eps = 0.01;
mdl_param(1).eps = eps;
mdl_param(2).eps =eps;
%activation funcs and F
mdl_param(1).Act = Act;
mdl_param(1).dAct_ds = dAct_ds;
mdl_param(1).F = 'F_NO_activation_final_layer';
%regularization
mdl_param(1).lambda = 0;
mdl_param(2).lambda = 0;
%make NN mdl
nn1 = make_NN_model(L, mdl_param);
%% mdl params for training
sgd_errors = 1;
nb_iterations = int64(100) % nb_iterations
batchsize = 64
step_size_params =  struct('eta_c', cell(1), 'eta_t', cell(1), ...
    'AdaGrad', cell(1), 'Momentum', cell(1), ...
    'Decaying', cell(1), 'step_size', cell(1), 'print_error_to_screen', cell(1) );
step_size_params.print_error_to_screen = 1;
step_size_params.Decaying = 1;
step_size_params.step_size = 0.01;
step_size_params.decay_rate = 1; %if 1 its not decaying then
%% train 1 hidden NN model
[ nn1, iteration_errors_train, iteration_errors_test ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train, Y_train, nn1, nb_iterations, batchsize, X_test,Y_test, step_size_params, sgd_errors);
train_error_nn1 = iteration_errors_train(nb_iterations);
test_error_nn1 = iteration_errors_test(nb_iterations);
plot( 1:nb_iterations+1, [iteration_errors_train, iteration_errors_test] );
%% make 2 hidden NN model
L=3; % 3 layer, 2 hidden layer
mdl_param = struct('Dim', cell(1,L), 'eps', cell(1,L) );
%dim of W and b
D_1 = 8;
D_2 = 4;
mdl_param(1).Dim = [D, D_1];
mdl_param(2).Dim = [D_1, D_2];
mdl_param(3).Dim = [D_2, D_out];
%scale of init W
eps = 0.01;
for l=1:L
    mdl_param(l).eps =eps;
end
%activation funcs and F
for l=1:L-1
    mdl_param(l).Act = Act;
    mdl_param(l).dAct_ds = dAct_ds;
end
mdl_param(1).F = 'F_NO_activation_final_layer';
%regularization
for l=1:L
    mdl_param(l).lambda = 0;
end
%make NN mdl
nn2 = make_NN_model( L, mdl_param);
%% mdl params for training
sgd_errors = 1;
nb_iterations = int64(20) % nb_iterations
batchsize = 64
step_size_params =  struct('eta_c', cell(1), 'eta_t', cell(1), ...
    'AdaGrad', cell(1), 'Momentum', cell(1), ...
    'Decaying', cell(1), 'step_size', cell(1), 'print_error_to_screen', cell(1) );
step_size_params.print_error_to_screen = 1;
step_size_params.Decaying = 1;
step_size_params.step_size = 0.01;
step_size_params.decay_rate = 1; %if 1 its not decaying then
%% train 2 hidden NN model
[ nn2, iteration_errors_train, iteration_errors_test ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train,Y_train, nn2, nb_iterations,batchsize, X_test,Y_test, step_size_params, sgd_errors );
train_error_nn2 = iteration_errors_train(nb_iterations);
test_error_nn2 = iteration_errors_test(nb_iterations);
%%
train_error_nn1
test_error_nn1
train_error_nn2
test_error_nn2
%%
beep;