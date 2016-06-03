function [ step, nb_iterations, batchsize ] = step_size_HBF2( hbf )
L = size(hbf,2);
%% step-size
step(1).print_error_to_screen = true;
step(1).AdaGrad = true;
step(1).Momentum = false;
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
        step.W(l).G_w_eta = 0.1;
        step.W(l).G_w  = zeros( size(hbf(l).W) );
    end
    for l=1:L
        step.Std(l).G_Std_eta = 0.1;
        step.Std(l).G_Std = zeros( size(hbf(l).beta) );
    end 
else
   %error('unknown optimzation method')
end
%% decay stuff
for l=1:L
    %step.W(l).eta = 5;
    step.W(l).decay_rate = 1.1; %if 1 its not decaying then
    step.W(l).decay_frequency = 3500;
    if mod(l,2) == 1
        step.W(l).eta = 0.1;
    else
        step.W(l).eta = 100;
    end
end
for l=1:L
    step.Std(l).eta = 0.1;
    step.Std(l).decay_rate = 1.1; %if 1 its not decaying then
    step.Std(l).decay_frequency = 2500;
end
%% nb_iterations
nb_iterations = int64(6000);
batchsize = 2000;
%% print iteration
factor = 600;
step.print_every_multiple = ceil(nb_iterations/factor);
end