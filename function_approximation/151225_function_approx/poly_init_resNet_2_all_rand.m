function net = poly_init_resNet_2_all_rand(varargin)
% net = poly_init_resNet('inputSize',[1 1 4],'hiddenLayers',[5])
opts.networkType = 'dagnn' ;
opts.hiddenLayers = [5];
opts.useBatchNorm = true;
opts.inputSize     = [1 1 5];
opts.nonlinearity = 'relu';
opts.learningRate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,5)];
opts.force_batchSize = [];
opts.checkPoints = 'all';
opts.disableBNParams = false;
opts.enable_residual = true;
opts = vl_argparse(opts, varargin) ;


lr = [1 1];

% Define network CIFAR10-quick
net.layers = {} ;

init_scale = 1;
init_scale_b = 1;
% Block 1
previous_featNum = opts.inputSize(3);
current_featNum = opts.hiddenLayers(1);
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{init_scale*randn(1,1,previous_featNum,current_featNum, 'single'), init_scale_b*randn(1, current_featNum, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0 ) ;
if opts.useBatchNorm
    net = insertBnorm_disableParams(net, numel(net.layers), opts.disableBNParams );
end
% if isstruct( opts.nonlinearity )
%     current_layer = opts.nonlinearity; current_layer.featNum = current_featNum; net.layers{end+1} = current_layer;  %struct('type', opts.nonlinearity{1}, 'featNum',opts.nonlinearity{2}) ;
% else
%     net.layers{end+1} = struct('type', opts.nonlinearity) ;
% end

for count = 2:numel(opts.hiddenLayers)    
    fromName = ['nonlinearity_' num2str(count)];
    if isstruct( opts.nonlinearity )
        current_layer = opts.nonlinearity;  current_layer.name = fromName; current_layer.featNum = current_featNum; net.layers{end+1} = current_layer;  %struct('type', opts.nonlinearity{1}, 'featNum',opts.nonlinearity{2}) ;
    else
        net.layers{end+1} = struct('type', opts.nonlinearity,'name',fromName) ;
    end
    
    previous_featNum = current_featNum;
    current_featNum = opts.hiddenLayers(count);
    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{init_scale*randn(1,1,previous_featNum,current_featNum, 'single'),  init_scale_b*randn(1, current_featNum, 'single')}}, ...
                               'learningRate', lr, ...
                               'stride', 1, ...
                               'pad', 0 ) ;
    if opts.useBatchNorm
        net = insertBnorm_disableParams(net, numel(net.layers),  opts.disableBNParams);
    end
    if opts.enable_residual
        net.layers{end+1} = struct('type', 'sum', 'name', ['sum_' num2str(count)], 'fromName', fromName); % bypass
    end
end

fromName = ['nonlinearity_last'];
if isstruct( opts.nonlinearity )
    current_layer = opts.nonlinearity; current_layer.name = fromName; current_layer.featNum = current_featNum; net.layers{end+1} = current_layer;    %struct('type', opts.nonlinearity{1}, 'featNum',opts.nonlinearity{2}) ;
else
    net.layers{end+1} = struct('type', opts.nonlinearity,'name',fromName) ;
end


previous_featNum = current_featNum;
current_featNum = 1;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{init_scale*randn(1,1,previous_featNum,current_featNum, 'single'), randn(1, current_featNum, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0 ) ;

% Loss layer
net.layers{end+1} = struct('type', 'squareloss') ;

% Meta parameters
net.meta.inputSize = opts.inputSize;
net.meta.trainOpts.learningRate = opts.learningRate ;
net.meta.trainOpts.weightDecay = 0 ;
net.meta.trainOpts.batchSize = 100 ;
if ~isempty(opts.force_batchSize)
    net.meta.trainOpts.batchSize = opts.force_batchSize ;
end
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    % net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
    %              {'prediction','label'}, 'error') ;
  otherwise
    assert(false) ;
end


% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{randn(ndim, 1, 'single'), randn(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;





% --------------------------------------------------------------------
function net = insertBnorm_disableParams(net, l, disableParams)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{randn(ndim, 1, 'single'), randn(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0], 'disableParams',disableParams) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;

