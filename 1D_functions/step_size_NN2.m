function [ step, nb_iterations, batchsize ] = step_size_NN2( nn )
L = size(nn,2);
%% step-size
step(1).print_error_to_screen = true;
step(1).AdaGrad = false;
step(1).Momentum = true;
%% optimization method
alpha_W = [0.9,0.5,0.4,0.3];
alpha_b = [0.9,0.5,0.4,0.3];
if step(1).Momentum
    for l=1:L
        %step.W(l).alpha = 0.5;
        step.W(l).alpha = alpha_W(l);
        step.W(l).v = zeros( size(nn(l).W) );
    end
    for l=1:L
        %step.b(l).alpha = 0.5;
        step.b(l).alpha = alpha_b(l);
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
etas_W = [0.6,0.4,0.3,0.0001];
etas_b = [0.5,0.4,0.3,0.0001];
for l=1:L
    step.W(l).eta = etas_W(l);
    %step.W(l).eta = 0.0001;
    step.W(l).decay_rate = 1.1; %if 1 its not decaying then
    step.W(l).decay_frequency = 1000;
end
for l=1:L
    step.b(l).eta = etas_b(l);
    %step.b(l).eta = 0.0001;
    step.b(l).decay_rate = 1.1; %if 1 its not decaying then
    step.b(l).decay_frequency = 1000;
end
%% nb_iterations
nb_iterations = int64(13500);
batchsize = 3000;
%% print iteration
factor = 100;
step.print_every_multiple = ceil(nb_iterations/factor);
end