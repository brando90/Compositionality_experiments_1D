function [ step_size_params_hbf1, nb_iterations_hbf1, batchsize_hbf1 ] = step_size_HBF1( )
%% step-size
step_size_params_hbf1 =  struct('eta_c', cell(1), 'eta_t', cell(1), ...
    'AdaGrad', cell(1), 'Momentum', cell(1), ...
    'Decaying', cell(1), 'step_size', cell(1), 'print_error_to_screen', cell(1) );
step_size_params_hbf1.print_error_to_screen = 1;
step_size_params_hbf1.Decaying = 1;
step_size_params_hbf1.step_size = 0.025;
step_size_params_hbf1.decay_rate = 1.25; %if 1 its not decaying then
step_size_params_hbf1.mod_when = 2000;
%% nb_iterations
nb_iterations_hbf1 = int64(100000);
batchsize_hbf1 = 32;
end