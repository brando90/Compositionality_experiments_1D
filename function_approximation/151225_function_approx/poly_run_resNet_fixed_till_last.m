function [net, info, imdb] = poly_run_resNet_fixed_till_last(varargin)
%% rprop
% nVar = 3; nOrder = 2; hiddenLayers = [4 4]; force_batchSize = 100; useBatchNorm = false; density = 1; poly_eval_func = @poly_eval_n3_d3_v05_extra_power2; nonlinearity = struct('type','relu');  [net, info] = poly_run_resNet('expDir',['~/vlfeat_exp/resNet/poly/poly_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,5)]*1,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0,'weightDecay',0); % --- square, the same architecture
% nVar = 3; nOrder = 2; hiddenLayers = [1 1]; force_batchSize = 200; useBatchNorm = false; density = 1; poly_eval_func = @poly_eval_n3_d3_v05_extra_power2; nonlinearity = struct('type','absSquare');  [net, info] = poly_run_resNet('expDir',['~/vlfeat_exp/resNet/poly/poly_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [0.05*ones(1,500)]*0.1,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0,'weightDecay',0,'checkPoints',[10]); % --- square, the same architecture

% numEpochs = 50000; disable_bias = true; all_t=tic; lr = 0.1; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [4 4 4]; force_batchSize = 300; useBatchNorm = true; density = 1; poly_eval_func = @poly_eval_n1_sin2; nonlinearity = struct('type','relu');  [net, info] = poly_run_resNet('expDir',['~/vlfeat_exp/resNet/poly/poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias)  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ 0.05*ones(1,numEpochs) ]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0:100:numEpochs],'rprop_p',rprop_p,'func_every_epoch',@randWeight_every_epoch,'disable_bias',disable_bias); toc(all_t); % --- square, the same architecture 

%% non-neg
% lr = 0.1; nVar = 1; nOrder = 2; rprop_p = 2; hiddenLayers = [10]; force_batchSize = 300; useBatchNorm = false; density = 1; poly_eval_func = @poly_eval_n1_sin2; nonlinearity = struct('type','relu');  [net, info] = poly_run_resNet('expDir',['~/vlfeat_exp/resNet/poly/poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [0.05*ones(1,30) 0.005*ones(1,15) 0.0005*ones(1,15)]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0],'rprop_p',rprop_p); % --- square, the same architecture 

%%


% run(fullfile(fileparts(mfilename('fullpath')), ...
%   '..', '..', 'matlab', 'vl_setupnn.m')) ;

% opts.modelType = 'lenet' ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = ''; % fullfile('data', sprintf('cifar-%s', opts.modelType)) ;
opts.gpus    = [1];
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.useBatchNorm = true;
opts.hiddenLayers = [5];
opts.nVar         = 4;
opts.nOrder       = 10;
opts.numTrain     = 60000; % 10000 
opts.numTest      = 30000; % 2000 
opts.density     = 0.5; 
opts.poly_eval_func = @poly_eval_n3_d3_v01;
opts.force_batchSize =[];
opts.momentum = 0.9;
opts.weightDecay = 0.0005;
opts.checkPoints = 'all';
opts.rprop_p = false;
opts.dataDir = fullfile('data','cifar') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.resNet_n = 3;
opts.func_every_epoch = [];
opts.networkType = 'dagnn' ;
opts.replaceFunc = [];
opts.learningRate =  [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,5)];
opts.nonlinearity = 'relu';
opts.disableBNParams = false;
opts.enable_residual = true;
opts.disable_bias = false;
opts = vl_argparse(opts, varargin) ; 


opts.train = struct() ;
opts.train.gpus  = opts.gpus;
opts.train.errorFunction =  'regression';
opts.train.momentum = opts.momentum;
opts.train.checkPoints = opts.checkPoints;
opts.train.weightDecay = opts.weightDecay;
opts.train.func_every_epoch = opts.func_every_epoch;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

nVar = opts.nVar;
nOrder = opts.nOrder;
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = setup_polynomial_imdb('nOrder',nOrder,'nVar',nVar,'numTrain',opts.numTrain,'numTest',opts.numTest,'density',opts.density,'poly_eval_func',opts.poly_eval_func);
  mkdir(opts.expDir) ; 
  save(opts.imdbPath, '-struct', 'imdb') ; 
end

%imdb = setup_polynomial_imdb(opts);

net = poly_init_resNet_2_fixed_till_last('inputSize',[1 1 nVar],'hiddenLayers',opts.hiddenLayers,'learningRate',opts.learningRate,'networkType',opts.networkType,'useBatchNorm',opts.useBatchNorm,'force_batchSize',opts.force_batchSize,'nonlinearity',opts.nonlinearity,'disableBNParams',opts.disableBNParams,'disable_bias',opts.disable_bias,'enable_residual',opts.enable_residual);

if opts.rprop_p == 1
    dagnet_to_rprop(net);
elseif opts.rprop_p == 2
    dagnet_to_4D_nonneg(net);
end

% if isDagNN(net)
%     opts.networkType = 'dagnn';
% else
%     opts.networkType = 'simplenn';
% end



if ~isempty(opts.replaceFunc) && strcmp(opts.networkType,'dagnn')
    net = cnn_net2resNet(net,opts.replaceFunc);
end

%net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------


switch opts.networkType
  case 'simplenn', trainfn = @cnn_train_beta17 ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train, ...
                      'val', find(imdb.images.set == 3)) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) poly_getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) poly_getDagNNBatch(bopts,x,y) ;
end




% % -------------------------------------------------------------------------
% function [images, labels] = getSimpleNNBatch(imdb, batch)
% % -------------------------------------------------------------------------
% images = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(1,batch) ;
% if rand > 0.5, images=fliplr(images) ; end

% % -------------------------------------------------------------------------
% function inputs = getDagNNBatch(opts, imdb, batch)
% % -------------------------------------------------------------------------
% images = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(1,batch) ;
% if rand > 0.5, images=fliplr(images) ; end
% if opts.numGpus > 0
%   images = gpuArray(images) ;
% end
% inputs = {'input', images, 'label', labels} ;
