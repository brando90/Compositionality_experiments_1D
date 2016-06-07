function conv_layer = poly_optimal_first_layer_for_grid(conv_layer)
domain = [-2*pi 2*pi];
num_feat = size(conv_layer.weights{1},4);
%conv_layer.weights{1}(:,:,:);
conv_layer.weights{1} = conv_layer.weights{1}*0 + 1;
%if num_feat == 12
assert(sqrt(num_feat) == round(sqrt(num_feat)));
interval = (domain(2)-domain(1))/(sqrt(num_feat)-1);
pivots = domain(1):interval:domain(2);
A = allcomb(pivots,pivots);
conv_layer.weights{2}() =;
% elseif num_feat == 6
%     conv_layer.weights{2}(:) = [-1.5*pi
% else
%     error('dim not supported');
% end

conv_layer.learningRate = [0.001 0.001];

