function [ nn, step ] = put_NN_in_GPU( nn, step )
L = size(nn,2);
for l=1:L
    nn(l).W = gpuArray(nn(l).W);
    nn(l).b = gpuArray(nn(l).b);
    nn(l).beta = gpuArray(nn(l).beta);
    nn(l).lambda = gpuArray(nn(l).lambda);
end
%%
if step(1).Momentum
    for l=1:L
        step.W(l).alpha = gpuArray( step.W(l).alpha );
        step.W(l).v = gpuArray(zeros( size(nn(l).W) ) );
    end
    for l=1:L
        step.b(l).alpha = gpuArray( step.b(l).alpha );
        step.b(l).v = gpuArray( zeros( size(nn(l).b) ) );
    end
elseif step(1).AdaGrad
    for l=1:L
        step.W(l).G_w  = gpuArray( zeros( size(nn(l).W) ) );
    end
    for l=1:L
        step.b(l).G_b = gpuArray( zeros( size(nn(l).b) ) );
    end 
else
   %error('unknown optimzation method')
end

for l=1:L
    step.W(l).eta = gpuArray(step.W(l).eta);
    step.W(l).decay_rate =  gpuArray(step.W(l).decay_rate); 
end
end