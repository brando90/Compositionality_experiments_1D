function [ step, nb_iterations, batchsize ] = step_size_HBF2( hbf )
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
        step.Std(l).alpha = 0.9;
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
    %step.W(l).eta = 10000;
    step.W(l).decay_rate = 1.1; %if 1 its not decaying then
    step.W(l).decay_frequency = 2500;
    if mod(l,2) == 1
        step.W(l).eta = 0.9;
    else
        step.W(l).eta = 10;
    end
end
for l=1:L
    step.Std(l).eta = 10;
    step.Std(l).decay_rate = 1.1; %if 1 its not decaying then
    step.Std(l).decay_frequency = 2500;
end
%% nb_iterations
nb_iterations = int64(8000);
batchsize = 500;
%% print iteration
factor = 600;
step.print_every_multiple = ceil(nb_iterations/factor);
end