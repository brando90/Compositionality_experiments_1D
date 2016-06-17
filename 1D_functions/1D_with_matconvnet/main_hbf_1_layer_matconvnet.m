clc;clear;clc;clear;
%% prepare Data
data_set = 'f1D_cos_snr_Inf';
fprintf('DATA SET: %s \n', data_set);
load(data_set);
%% make train/test data set
N_train = 60000;
N_test = 60000;
imbd = make_1D_data_matconvnet_format( X_train, X_train, X_test, Y_test );
%% prepare parameters
L1=3;
D1=1;
w1 = randn(1,1, D1,L1); % D
s1 = 0.5; %1st layer std
c1 = randn(1, L1); %1st layer coeffs
G1 = ones(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1) BN scale, one per  dimension
B1 = zeros(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1) BN shift, one per  dimension
bn_eps = 1e-4;
%% learning rate
eta_w1 = 0.9;
eta_s1 = 0.9;
eta_G1 = 0.9;
eta_B1 = 0.9;
eta_G1 = 0.9;
%% weight decay
decay_w1 = 1; %TODO what is this?
decay_s1 = 1; %TODO what is this?
%% make HBF
net.layers = {} ;
net.layers{end+1} = struct('type', 'custom', ...
                           'name', 'hbf_norm1',...
                           'forward', get_custom_hbf_norm_forward(@cutom_hbf_norm_forward), ...
                           'backward', get_custom_hbf_norm_backward(@cutom_hbf_norm_backward), ...
                           'weights', {{w1,s1}}, ...
                           'learningRate', [eta_w1 eta_s1], ...
                           'weightDecay', [decay_w1 decay_s1]) ;
net.layers{end+1} = struct('type', 'bnorm', ...net.layers{1, 1}.weights
                           'name', 'bnorm1',...
                           'weights', {{G1, B1}}, ...
                           'EPSILON', bn_eps, ...
                           'learningRate', [1 1 0.05], ...
                           'weightDecay', [0 0]) ;  
net.layers{end+1} = struct('type', 'custom', ...
                           'name', 'exp1', ...                           
                           'forward', get_custom_hbf_norm_forward(@cutom_hbf_norm_forward), ...
                           'backward', get_custom_hbf_norm_backward(@cutom_hbf_norm_backward) ) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv2', ...
                           'weights', {{c1, []}}, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pdist', ...
                           'name', 'averageing1', ...
                           'class', 0, ...
                           'p', 1) ;
net.layers{end+1} = struct('type', 'custom', ...
                           'name', 'L2_loss', ...
                           'forward', get_custom_l2_loss_forward(@l2LossForward), ...
                           'backward', get_custom_l2_loss_backward(@l2LossBackward), ...
                           'class', imdb.images.label) ;             
net = vl_simplenn_tidy(net) ;
%% prepare train options
trainOpts.expDir = 'results/' ; % TODO %save results/trained cnn
trainOpts.gpus = [] ; % TODO 
trainOpts.batchSize = 2 ;
trainOpts.learningRate = 0.02 ; %TODO: why is this learning rate here?
trainOpts.plotDiagnostics = false ; % TODO 
%trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
trainOpts.numEpochs = 50 ; % number of training epochs
trainOpts.errorFunction = 'none' ; % TODO 
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