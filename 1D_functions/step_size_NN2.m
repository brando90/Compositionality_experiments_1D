function [ step, nb_iterations, batchsize ] = step_size_NN2( nn )
L = size(nn,2);
%% step-size
step(1).print_error_to_screen = true;
step(1).AdaGrad = false;
step(1).Momentum = true;
%% optimization method
if step(1).Momentum
    for l=1:L
        step.W(l).alpha = 0.9;
        step.W(l).v = zeros( size(nn(l).W) );
    end
    for l=1:L
        step.b(l).alpha = 0.9;
        step.b(l).v = zeros( size(nn(l).b) );
    end
elseif step(1).AdaGrad
    for l=1:L
        step.W(l).G_w  = zeros( size(nn(l).W) );
    end
    for l=1:L
        step.b(l).G_b = zeros( size(nn(l).b) );
    end 
else
   error('unknown optimzation method')
end
%% decay stuff
for l=1:L
    step.W(l).eta = 0.025;
    step.W(l).decay_rate = 1.25; %if 1 its not decaying then
    step.W(l).decay_frequency = 2000;
end
for l=1:L
    step.b(l).eta = 0.025;
    step.b(l).decay_rate = 1.25; %if 1 its not decaying then
    step.b(l).decay_frequency = 2000;
end
%% nb_iterations
nb_iterations = int64(10000);
batchsize = 4;
%% print iteration
factor = 100;
step.print_every_multiple = ceil(nb_iterations/factor);
end