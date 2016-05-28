function [ step_size_params, nb_iterations, batchsize ] = step_size_NN2( )
L = 6;
%% step-size
step_size_params =  struct('eta_c', cell(1), 'eta_t', cell(1), ...
    'AdaGrad', cell(1), 'Momentum', cell(1), ...
    'Decaying', cell(1), 'step_size', cell(1), 'print_error_to_screen', cell(1) );
step_size_params.print_error_to_screen = 1;
for l=1:L
    step_size_params.W(l).eta = 0.025;
    step_size_params.W(l).decay_rate = 1.25; %if 1 its not decaying then
    step_size_params.W(l).decay_frequency = 2000;
end
for l=1:L
    step_size_params.b(l).eta = 0.025;
    step_size_params.b(l).decay_rate = 1.25; %if 1 its not decaying then
    step_size_params.b(l).decay_frequency = 2000;
end
%% nb_iterations
nb_iterations = int64(10000);
batchsize = 4;
%% print iteration
factor = 100;
step_size_params.print_every_multiple = ceil(nb_iterations/factor);
end