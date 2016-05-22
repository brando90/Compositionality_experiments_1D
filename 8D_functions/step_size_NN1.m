function [ step_size_params_nn1, nb_iterations_nn1, batchsize_nn1 ] = step_size_NN1( )
%% step-size
step_size_params_nn1 =  struct('eta_c', cell(1), 'eta_t', cell(1), ...
    'AdaGrad', cell(1), 'Momentum', cell(1), ...
    'Decaying', cell(1), 'step_size', cell(1), 'print_error_to_screen', cell(1) );
step_size_params_nn1.print_error_to_screen = 1;
step_size_params_nn1.Decaying = 1;
step_size_params_nn1.step_size = 0.025;
step_size_params_nn1.decay_rate = 1.25; %if 1 its not decaying then
step_size_params_nn1.mod_when = 2000;
%% nb_iterations
nb_iterations_nn1 = int64(10000);
batchsize_nn1 = 32;
end