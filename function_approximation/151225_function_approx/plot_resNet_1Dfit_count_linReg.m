% [net imdb]= load_net_imdb('~/vlfeat_exp/resNet/poly/poly_hidden_1_nVar_1_nOrder_2_density_1_poly_eval_n1_power2_noBias_batchSize_200_bn_0_relu/net-epoch-10.mat');
% clear;
% % [net imdb]= load_net_imdb('~/vlfeat_exp/resNet/poly/poly_hidden_3_nVar_1_nOrder_2_density_1_poly_eval_n1_power2_noBias_batchSize_100_bn_0_relu/net-epoch-10.mat');
% % [net imdb]= load_net_imdb('~/vlfeat_exp/resNet/poly/poly_hidden_1_nVar_1_nOrder_2_density_1_poly_eval_n1_power4_v01_batchSize_100_bn_0_relu/net-epoch-10.mat');
% [net imdb]= load_net_imdb('~/vlfeat_exp/resNet/poly/poly_lr_1_rprop_1_hidden_120_120_nVar_1_nOrder_2_density_1_poly_eval_n1_sin_batchSize_100_bn_1_relu/net-epoch-4.mat');

nVar = 1;
opts.hiddenLayers = ones(1,10)*50;
opts.learningRate = [0.05*ones(1,500)]*0.1;
opts.networkType  = 'dagnn';
opts.useBatchNorm = false;
opts.disableBNParams = true;
opts.force_batchSize = 300;
%opts.nonlinearity = 'relu'; % struct('type','relu');
opts.nonlinearity = struct('type','abs');
net = poly_init_resNet_2_all_rand('inputSize',[1 1 nVar],'hiddenLayers',opts.hiddenLayers,'learningRate',opts.learningRate,'networkType',opts.networkType,'useBatchNorm',opts.useBatchNorm,'force_batchSize',opts.force_batchSize,'nonlinearity',opts.nonlinearity,'disableBNParams',opts.disableBNParams);

net_ = dagnn.DagNN.loadobj(net) ;
%points = gpuArray(single(reshape(-1000:0.1:1000,1,1,1,[])));  % imdb.images.data(:,:,:,batch);
points = -100:0.1:100;  % imdb.images.data(:,:,:,batch);
imdb = struct;
imdb.images.data = single(reshape(points,1,1,1,[]));
imdb.images.labels = imdb.images.data;
inputs = poly_getDagNNBatch(struct('numGpus',[]), imdb, 1:numel(points));
%move(net_, 'gpu');
net_.vars(end-2).precious = 1
net_.mode = 'test';
net_.eval(inputs);

outputs = linearized( net_.vars(end-2).value );
%targets = linearized( imdb.images.labels(:,:,:,batch) );

%figure; hold all;
%plot(points,targets(sorted_idx),'-.or');
plot(points,outputs,'-b');
slopes = poly_get_1D_slopes(points,outputs,0.01)
numel(slopes)
%legend({'Ground truth','Approximated'});



