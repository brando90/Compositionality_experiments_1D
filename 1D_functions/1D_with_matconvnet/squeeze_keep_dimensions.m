function [ W ] = squeeze_keep_dimensions( W )
% always returns D1 x D2 even if D1=1
num_dim = ndims(W);
if num_dim == 2
    return;
else
    % size(W) >= 3
    dimensions_W = size(W);
    for d=1:num_dim-1
        current_dim = dimensions_W(d);
        if current_dim ~= 1
            W = squeeze(W);
            return;
        end
    end
    % all dimensions are equal to 1 
    W = squeeze(W); % (D2 x D1) = (D2 x D1) = (D2 x 1)
    W = W'; % (D1 x D2)
    return;
end
end