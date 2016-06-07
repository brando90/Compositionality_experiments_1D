function conv_layer = poly_optimal_first_layer_for_sin1d5x(conv_layer)
domain = [-2*pi 2*pi];
num_feat = size(conv_layer.weights{1},4);
%conv_layer.weights{1}(:,:,:);
conv_layer.weights{1} = conv_layer.weights{1}*0 + 1;
%if num_feat == 12
interval = (domain(2)-domain(1))/(num_feat-1);
conv_layer.weights{2}(:) = domain(1):interval:domain(2);
% elseif num_feat == 6
%     conv_layer.weights{2}(:) = [-1.5*pi
% else
%     error('dim not supported');
% end

conv_layer.learningRate = [0.001 0.001];

