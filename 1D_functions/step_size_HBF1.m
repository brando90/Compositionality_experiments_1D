function [ step_size_params, nb_iterations, batchsize ] = step_size_HBF1( )
L = 2;
%% step-size
step_size_params =  struct( 'AdaGrad', cell(1,1), 'Momentum', cell(1,1), ...
    'Decaying', cell(1,1), 'step_size', cell(1,1), ...
    'print_error_to_screen', cell(1,1) );
step_size_params.print_error_to_screen = 1;
%% constant step size
for l=1:L
    step_size_params.W(l).eta = 0.025;
    step_size_params.W(l).decay_rate = 1.25; %if 1 its not decaying then
    step_size_params.W(l).decay_frequency = 2000;
end
for l=1:L
    step_size_params.Std(l).eta = 0.025;
    step_size_params.Std(l).decay_rate = 1.25; %if 1 its not decaying then
    step_size_params.Std(l).decay_frequency = 2000;
end
%% nb_iterations
nb_iterations = int64(10000);
batchsize = 4;
%% print iteration
factor = 100;
step_size_params.print_every_multiple = ceil(nb_iterations/factor);
end