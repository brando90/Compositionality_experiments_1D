function [ step_size_params_nn2, nb_iterations_nn2, batchsize_nn2 ] = step_size_NN2( )
%% step-size
step_size_params_nn2 =  struct('eta_c', cell(1), 'eta_t', cell(1), ...
    'AdaGrad', cell(1), 'Momentum', cell(1), ...
    'Decaying', cell(1), 'step_size', cell(1), 'print_error_to_screen', cell(1) );
step_size_params_nn2.print_error_to_screen = 1;
step_size_params_nn2.Decaying = 1;
step_size_params_nn2.step_size = 0.015;
step_size_params_nn2.decay_rate = 1.7; %if 1 its not decaying then
step_size_params_nn2.mod_when = 2000;
%% nb_iterations
nb_iterations_nn2 = int64(10000);
batchsize_nn2 = 16;
%% print iteration
factor = 100;
step_size_params_nn2.print_every_multiple = ceil(nb_iterations_nn2/factor);
end

