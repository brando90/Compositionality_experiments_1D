setup() ;
% setup('useGpu', true); % Uncomment to initialise with a GPU support
%% Part 3.1: Prepare the data
% Load a database of blurred images to train from
imdb = load('data/text_imdb.mat') ;

%% Part 3.2: Create a network architecture

net = initializeSmallCNN() ;
%net = initializeLargeCNN() ;
% Display network
vl_simplenn_display(net) ;

%% Part 3.3: learn the model
% Add a loss (using a custom layer)
net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;

% Train
trainOpts.expDir = 'data/text-small' ;
trainOpts.gpus = [] ;
% Uncomment for GPU training:
%trainOpts.expDir = 'data/text-small-gpu' ;
%trainOpts.gpus = [1] ;
trainOpts.batchSize = 16 ;
trainOpts.learningRate = 0.02 ;
trainOpts.plotDiagnostics = false ;
%trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
trainOpts.numEpochs = 20 ;
trainOpts.errorFunction = 'none' ;

net = cnn_train(net, imdb, @getBatch, trainOpts) ;