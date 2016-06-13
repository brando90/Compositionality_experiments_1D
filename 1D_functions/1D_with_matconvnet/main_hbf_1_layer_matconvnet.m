clc;clear;clc;clear;
%% prepare Data
data_set = 'f1D_cos_snr_Inf';
fprintf('DATA SET: %s \n', data_set);
load(data_set);
%% make train/test data set
N_train = 60000;
N_test = 60000;
X_train = X_train(1:N_train,:);
Y_train = Y_train(1:N_train,:);
X_test = X_test(1:N_test,:);
Y_test = Y_test(1:N_test,:);
%split = ones(1,M);
%split(floor(M*0.75):end) = 2;
%imdb.images.set = split;
% load image dadabase (imgdb)
imdb.images.data = X_train;
imdb.images.label = Y_test;
%% prepare parameters
L1=3;

w1 = randn(1,1,1,L1); %1st layer weights
s1 = 0.5; %1st layer std

w2 = randn(1,1,1,L1); %2nd layer weights
b2 = randn(1,1,1,L1); %2nd layer biases

G1 = ones(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1) BN scale, one per  dimension
B1 = zeros(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1) BN shift, one per  dimension
bn_eps = 1e-4;
%% make CNN layers: conv, BN, relu, conv, pdist, l2-loss
net.layers = {} ;
addCustom_hbf_norm_layer(net, @cutom_hbf_norm_forward, @cutom_hbf_norm_backward);
net.layers{end+1} = struct('type', 'custom', ...
                           'name', 'hbf_norm1',...
                           'forward', add_custom_hbf_norm_forward(@cutom_hbf_norm_forward), ...
                           'backward', add_custom_hbf_norm_backward(@cutom_hbf_norm_backward), ...
                           'weights', {w1,s1}, ... %TODO
                           'learningRate', [0.9 0.9], ... %TODO
                           'weightDecay', [1 1]) ; %TODO
net.layers{end+1} = struct('type', 'bnorm', ...
                           'name', 'bnorm1',...
                           'weights', {{g1, b1}}, ...
                           'EPSILON', bn_eps, ...
                           'learningRate', [1 1 0.05], ...
                           'weightDecay', [0 0]) ;  
net.layers{end+1} = struct('type', 'custom', ... %TODO
                           'name', 'exp1' ) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv2', ...
                           'weights', {{w2, b2}}, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pdist', ...
                           'name', 'averageing1', ...
                           'class', 0, ...
                           'p', 1) ;
net.layers{end+1} = struct('type', 'custom', ...
                           'name', 'L2_loss', ...
                           'forward', get_custom_l2_loss_forward(@l2LossForward), ...
                           'backward', get_custom_l2_loss_backward(@l2LossBackward), ...
                           'class', imdb.images.label,
                           ) ;
%% add L2-loss                   
net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;
net.layers{end}.class = Y_test; % its the test set
net = vl_simplenn_tidy(net) ;
%% prepare train options
trainOpts.expDir = 'results/' ; %save results/trained cnn
trainOpts.gpus = [] ;
trainOpts.batchSize = 2 ;
trainOpts.learningRate = 0.02 ; %TODO: why is this learning rate here?
trainOpts.plotDiagnostics = false ;
%trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
trainOpts.numEpochs = 50 ; % number of training epochs
trainOpts.errorFunction = 'none' ;
%% CNN TRAIN
res = vl_simplenn(net, X_train, 1);
for epoch=1:trainOpts.numEpochs
    %% forward pass and compute derivatives
    projection = 1;
    res = vl_simplenn(net, X_train, projection); % check these derivatives numerically?
    %% SGD
    num_layers = size(net, 2);
    for l=num_layers:-1:1
        for j=1:numel(res(l).dzdw)
            net.layers{l}.weights{j} = net.layers{l}.weights{j} - net.layers{l}.learningRate(j)*res(l).dzdw{j}; 
        end    
    end
end
%%
disp('END')
beep;