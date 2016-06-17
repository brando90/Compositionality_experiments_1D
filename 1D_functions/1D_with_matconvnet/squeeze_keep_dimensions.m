function [ W ] = squeeze_keep_dimensions( W )
% always returns M x D
size_array = size(W); % (a x b x D x M)
if size_array(1) == 1 && size_array(2) == 1 && size_array(3) == 1
    % case: (D x M) = (1 x D)
    W = squeeze(W)'; % (1 x D)= (D x 1)'
else
    W = squeeze(W); % (D x M)
end
end