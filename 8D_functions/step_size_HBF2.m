function [ step_size_params_hbf2, nb_iterations_hbf2, batchsize_hbf2 ] = step_size_HBF2( )
%% step-size
step_size_params_hbf2 =  struct('eta_c', cell(1), 'eta_t', cell(1), ...
    'AdaGrad', cell(1), 'Momentum', cell(1), ...
    'Decaying', cell(1), 'step_size', cell(1), 'print_error_to_screen', cell(1) );
step_size_params_hbf2.print_error_to_screen = 1;
step_size_params_hbf2.Decaying = 1;
step_size_params_hbf2.step_size = 0.01;
step_size_params_hbf2.decay_rate = 1.5; %if 1 its not decaying then
step_size_params_hbf2.mod_when = 2000;
%% nb_iterations
nb_iterations_hbf2 = int64(1000);
batchsize_hbf2 = 2;
end

