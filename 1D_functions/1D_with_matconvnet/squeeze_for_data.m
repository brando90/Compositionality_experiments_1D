function [ X ] = squeeze_for_data( X )
% always returns M x D
size_array = size(X); % (a x b x D x M)
if size_array(1) == 1 && size_array(2) == 1 && size_array(3) == 1
    X = squeeze(X); % (M x 1)
else
    X = squeeze(X)'; % (M x D) = (D x M)'
end
end