%% target function
h11 = @(A) (1*A(1) + 2*A(2))^2;
h12 = @(A) (3*A(1) + 4*A(2))^3;
h21 = @(A) (5*A(1) + 6*A(2))^4;
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
nb_samples = 10000;
[X,Y] = generate_data_from_function( f_target, snr, low_x,high_x, nb_samples, sigpower, powertype);
D = size(X,2);
D_out = size(Y,2);
%% make train/test data set
percentage_split = 0.8;
N_train = nb_samples * percentage_split;
N_test = nb_samples * (1 - percentage_split);
%% Activation funcs
run('./activation_funcs');
Act = relu_func;
dAct_ds = dRelu_ds;
lambda = 0;
%% make 1 hidden NN model
L=2;
D_1 = 8;
mdl_param = struct('Dim', cell(1,L), 'eps', cell(1,L) );
mdl_param(1).Dim = [D, D_1];
mdl_param(2).Dim = [D_1, D_out];
eps = 0.01;
mdl_param(1).eps = eps;
mdl_param(2).eps =eps;
mdl_param.Act = Act;
mdl_param.Act = dAct_ds;
mdl_param.lambda = 0;
nn1 = make_NN_model( L, mdl_param);
%% make 2 hidden NN model
L=3;
D_1 = 8;
D_2 = 4;
mdl_param = struct('D', cell(1,L));
mdl_param(1).Dim = [D, D_1];
mdl_param(2).Dim = [D_1, D_2];
mdl_param(2).Dim = [D_2, D_out];
eps = 0.01;
epsilon(1).eps = eps;
epsilon(2).eps =eps;
epsilon(3).eps =eps;
nn2 = make_NN_model( L, mdl_param, epsilon , Act, dAct_ds, lambda);