
% load ~/vlfeat_exp/resNet/poly/poly_hidden_1_1_nVar_1_nOrder_2_density_1_poly_eval_n1_power4_noBias_batchSize_200_bn_0_absSquare/net-epoch-10.mat
% imdb = load('~/vlfeat_exp/resNet/poly/poly_hidden_1_1_nVar_1_nOrder_2_density_1_poly_eval_n1_power4_noBias_batchSize_200_bn_0_absSquare/imdb.mat');

% load ~/vlfeat_exp/resNet/poly/poly_hidden_1_1_1_nVar_1_nOrder_2_density_1_poly_eval_n1_power6_noBias_batchSize_200_bn_0_absSquare/net-epoch-10.mat
% imdb = load('~/vlfeat_exp/resNet/poly/poly_hidden_1_1_1_nVar_1_nOrder_2_density_1_poly_eval_n1_power6_noBias_batchSize_200_bn_0_absSquare/imdb.mat');

% load ~/vlfeat_exp/resNet/poly/poly_hidden_4_4_4_nVar_1_nOrder_2_density_1_poly_eval_n1_power6_noBias_batchSize_200_bn_0_absSquare/net-epoch-10.mat
% imdb = load('~/vlfeat_exp/resNet/poly/poly_hidden_4_4_4_nVar_1_nOrder_2_density_1_poly_eval_n1_power6_noBias_batchSize_200_bn_0_absSquare/imdb.mat');

% load ~/vlfeat_exp/resNet/poly/poly_hidden_1_nVar_1_nOrder_2_density_1_poly_eval_n1_power6_noBias_batchSize_200_bn_0_absSquare/net-epoch-10.mat
% imdb = load('~/vlfeat_exp/resNet/poly/poly_hidden_1_nVar_1_nOrder_2_density_1_poly_eval_n1_power6_noBias_batchSize_200_bn_0_absSquare/imdb.mat');

% relu 1
% [net imdb]= load_net_imdb('~/vlfeat_exp/resNet/poly/poly_hidden_1_nVar_1_nOrder_2_density_1_poly_eval_n1_power2_noBias_batchSize_200_bn_0_relu/net-epoch-10.mat');


%clear;
% [net imdb]= load_net_imdb('~/vlfeat_exp/resNet/poly/poly_hidden_3_nVar_1_nOrder_2_density_1_poly_eval_n1_power2_noBias_batchSize_100_bn_0_relu/net-epoch-10.mat');
% [net imdb]= load_net_imdb('~/vlfeat_exp/resNet/poly/poly_hidden_1_nVar_1_nOrder_2_density_1_poly_eval_n1_power4_v01_batchSize_100_bn_0_relu/net-epoch-10.mat');

%

function  plot_resNet_1Dfit_func(net,imdb,newFig)
if ~exist('newFig','var') || newFig
    close all;
end
% [net imdb]= load_net_imdb(filePath);

net_ = dagnn.DagNN.loadobj(net) ;
train_idx = find(imdb.images.set==1);
test_idx = find(imdb.images.set==3);
batch = test_idx(1:30000);

points = imdb.images.data(:,:,:,batch);
points = linearized(points);
[points sorted_idx] = sort(points);
inputs = poly_getDagNNBatch(struct('numGpus',[]), imdb, batch);
net_.vars(end-2).precious = 1;
net_.eval(inputs);

outputs = linearized( net_.vars(end-2).value );
targets = linearized( imdb.images.labels(:,:,:,batch) );

if ~exist('newFig','var') || newFig
    figure;hold all;
else
    hold all;
end
%outputs(outputs<0) = 0;
plot(points,targets(sorted_idx),'-.or');
plot(points,outputs(sorted_idx),'-b');
% if ndims(net_.params(2).value) == 2
%     gridxy(net_.params(2).value(:)','Color','y','linewidth',1) ;
% end
legend({'Ground truth','Approximated'});

