function [ step, nb_iterations, batchsize ] = step_size_HBF1( hbf )
L = size(hbf,2);
%% step-size
step(1).print_error_to_screen = true;
step(1).AdaGrad = false;
step(1).Momentum = true;
%% optimization method
if step(1).Momentum
    for l=1:L
        step.W(l).alpha = 0.9;
        step.W(l).v = zeros( size(hbf(l).W) );
    end
    for l=1:L
        step.Std(l).alpha = 0.95;
        step.Std(l).v = zeros( size(hbf(l).beta) );
    end
elseif step(1).AdaGrad
    for l=1:L
        step.W(l).G_w  = zeros( size(hbf(l).W) );
    end
    for l=1:L
        step.Std(l).G_b = zeros( size(hbf(l).beta) );
    end 
else
   %error('unknown optimzation method')
end
%% decay stuff
for l=1:L
    step.W(l).eta = 0.01;
    step.W(l).decay_rate = 1.1; %if 1 its not decaying then
    step.W(l).decay_frequency = 2500;
end
for l=1:L
    step.Std(l).eta = 50;
    step.Std(l).decay_rate = 1.1; %if 1 its not decaying then
    step.Std(l).decay_frequency = 2500;
end
%% nb_iterations
nb_iterations = int64(6000);
batchsize = 100;
%% print iteration
factor = 60;
step.print_every_multiple = ceil(nb_iterations/factor);
end