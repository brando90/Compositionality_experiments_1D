function [net, info, imdb] = poly_run_resNet(varargin)
%% rprop
% nVar = 3; nOrder = 2; hiddenLayers = [4 4]; force_batchSize = 100; useBatchNorm = false; density = 1; poly_eval_func = @poly_eval_n3_d3_v05_extra_power2; nonlinearity = struct('type','relu');  [net, info] = poly_run_resNet('expDir',['~/vlfeat_exp/resNet/poly/poly_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,5)]*1,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0,'weightDecay',0); % --- square, the same architecture
% nVar = 3; nOrder = 2; hiddenLayers = [1 1]; force_batchSize = 200; useBatchNorm = false; density = 1; poly_eval_func = @poly_eval_n3_d3_v05_extra_power2; nonlinearity = struct('type','absSquare');  [net, info] = poly_run_resNet('expDir',['~/vlfeat_exp/resNet/poly/poly_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [0.05*ones(1,500)]*0.1,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0,'weightDecay',0,'checkPoints',[10]); % --- square, the same architecture

% numEpochs = 50000; disable_bias = false; all_t=tic; lr = 0.01; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [12]; force_batchSize = 300; useBatchNorm = false; density = 1; poly_eval_func = @poly_eval_n1_sin2; nonlinearity = struct('type','relu');  [net, info] = poly_run_resNet('expDir',['~/vlfeat_exp/resNet/poly/poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias)  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ 0.05*ones(1,numEpochs) ]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0:100:numEpochs],'rprop_p',rprop_p,'func_every_epoch',@randWeight_every_epoch,'disable_bias',disable_bias); toc(all_t); % --- square, the same architecture 
%% normal
% enable_residual = false; domain = [-2*pi 2*pi]; numEpochs = 2000; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.001; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [1]; force_batchSize = 3000; useBatchNorm = true; density = 1; poly_eval_func = @poly_eval_n1_simple_1_layer; nonlinearity = struct('type','relu');  [net, info] = poly_run_resNet('enable_residual',enable_residual,'domain',domain,'expDir',['~/vlfeat_exp/resNet/poly_4/poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func)  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ones(1,numEpochs)]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0 numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'no_plot',false); toc(all_t); % --- square, the same architecture

%% approx random
% nVar = 100; nOut = 1; domain = [-10 10]; numEpochs = 100; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.1; nOrder = 2; rprop_p = 0; hiddenLayers = [12]; force_batchSize = 300; useBatchNorm = true; density = 1; poly_eval_func = @poly_evalNet_10_10_10_10_noBN; nonlinearity = struct('type','relu');  [net, info] = poly_run_resNet('nOut',nOut,'domain',[-2 2],'expDir',['~/vlfeat_exp/resNet/poly_2/poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func)  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0:20:numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func); toc(all_t); % --- square, the same architecture % 10 10 10: obj 17,  10: obj 16,  30: obj 5

%% 2D
% numEpochs = 100; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.1; nVar = 2; nOrder = 2; rprop_p = 1; hiddenLayers = [200 200 200 200]; force_batchSize = 300; useBatchNorm = true; density = 1; poly_eval_func = @poly_eval_n2_grid_PI; nonlinearity = struct('type','relu');  [net, info] = poly_run_resNet('expDir',['~/vlfeat_exp/resNet/poly/poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func)  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0:20:numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func); toc(all_t); % --- square, the same architecture

%% binary tree
% nVar = 16; nOut = 1; domain = [-5 5]; numEpochs = 100; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.0001; nOrder = 2; rprop_p = 0; hiddenLayers = [5 5 5 5]; force_batchSize = 300; useBatchNorm = true; density = 1; poly_eval_func = @poly_eval_spacial_binary_tree_sin_v01_n16; nonlinearity = @vl_nnrelu;  [net, info] = poly_run_resNet('nOut',nOut,'domain',domain,'expDir',['~/vlfeat_exp/resNet/poly_2/poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' func2str_2(nonlinearity) '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func)  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity,'learningRate', [ ones(1,numEpochs) ]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0:20:numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func); toc(all_t);

% shallow
%  10*ones(1,log2(nVar))
%  10*log2(nVar)
%  domain = [-2*pi 2*pi];
% nVar = 16; nOut = 1;  domain = [-2*pi 2*pi]; numEpochs = 100; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.00001; nOrder = 2; rprop_p = 0; hiddenLayers = 10*ones(1,log2(nVar)); force_batchSize = 300; useBatchNorm = true; density = 1; poly_eval_func = @poly_eval_spacial_binary_tree_ReLU_v01_n16; nonlinearity = struct('type','relu');  [net, info] = poly_run_resNet('nOut',nOut,'domain',domain,'expDir',['~/vlfeat_exp/resNet/poly_2/poly_lr_' num2str(lr) '_domain_' num2str_underscore(domain)  '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '_2_'  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func); toc(all_t);

% nVar = 16; nOut = 1;  domain = [-2*pi 2*pi]; numEpochs = 100; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.00001; nOrder = 2; rprop_p = 0; hiddenLayers = 10*ones(1,log2(nVar)); force_batchSize = 300; useBatchNorm = true; density = 1; poly_eval_func = @poly_eval_spacial_binary_tree_ReLU_v01_n16; nonlinearity = struct('type','relu');  [net, info] = poly_run_resNet('nOut',nOut,'domain',domain,'expDir',['~/vlfeat_exp/resNet/poly_2/poly_lr_' num2str(lr)  '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func)  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func); toc(all_t);


% run(fullfile(fileparts(mfilename('fullpath')), ...
%   '..', '..', 'matlab', 'vl_setupnn.m')) ;

% opts.modelType = 'lenet' ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = ''; % fullfile('data', sprintf('cifar-%s', opts.modelType)) ;
opts.gpus    = [1];
opts.no_plot = false;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.useBatchNorm = true;
opts.hiddenLayers = [5];
opts.nVar         = 4;
opts.nOrder       = 10;
opts.numTrain     = 60000; % 10000 
opts.numTest      = 60000; % 2000 
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
opts.disable_bias = false;
opts.first_layer_func = [];
opts.enable_residual = false;
opts.no_last_bnorm = true; 
opts.domain =  [-2*pi 2*pi];
opts.nOut = 1;
opts.weightInitMethod = 'gaussian';
opts = vl_argparse(opts, varargin) ; 


opts.train = struct() ;
opts.train.gpus  = opts.gpus;
opts.train.errorFunction =  'regression';
opts.train.momentum = opts.momentum;
opts.train.checkPoints = opts.checkPoints;
opts.train.weightDecay = opts.weightDecay;
opts.train.func_every_epoch = opts.func_every_epoch;
opts.train.no_plot = opts.no_plot;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

nVar = opts.nVar;
nOrder = opts.nOrder;
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = setup_polynomial_imdb('nOrder',nOrder,'nVar',nVar,'numTrain',opts.numTrain,'numTest',opts.numTest,'density',opts.density,'poly_eval_func',opts.poly_eval_func,'domain',opts.domain,'nOut',opts.nOut);
    mkdir_if_not_exist(opts.expDir) ; 
    save(opts.imdbPath, '-struct', 'imdb') ; 
end

%imdb = setup_polynomial_imdb(opts);

if  ~isempty(regexp(func2str(opts.poly_eval_func),'_eval_spacial'))
    net = poly_init_binary_tree_convNet('inputSize',[nVar 1 1],'hiddenLayers',opts.hiddenLayers,'learningRate',opts.learningRate,'networkType',opts.networkType,'useBatchNorm',opts.useBatchNorm,'force_batchSize',opts.force_batchSize,'nonlinearity',opts.nonlinearity,'disableBNParams',opts.disableBNParams,'disable_bias',opts.disable_bias,'first_layer_func',opts.first_layer_func,'enable_residual',opts.enable_residual,'weightInitMethod',opts.weightInitMethod,'no_last_bnorm',opts.no_last_bnorm); 
else
    net = poly_init_resNet_2_no_last_bnorm_02('inputSize',[1 1 nVar],'hiddenLayers',opts.hiddenLayers,'learningRate',opts.learningRate,'networkType',opts.networkType,'useBatchNorm',opts.useBatchNorm,'force_batchSize',opts.force_batchSize,'nonlinearity',opts.nonlinearity,'disableBNParams',opts.disableBNParams,'disable_bias',opts.disable_bias,'first_layer_func',opts.first_layer_func,'enable_residual',opts.enable_residual,'weightInitMethod',opts.weightInitMethod,'no_last_bnorm',opts.no_last_bnorm); 
end

if opts.rprop_p == 1
    dagnet_to_rprop(net);
elseif opts.rprop_p == 2
    dagnet_to_4D_nonneg(net);
end

if imdb.opts.nVar==1
    plot_resNet_1Dfit_func(net,imdb);
    pause(2);
    %    waitforbuttonpress 
elseif imdb.opts.nVar==2
    plot_resNet_2Dfit_func(net,imdb);
    pause(2);
    %    waitforbuttonpress 
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
