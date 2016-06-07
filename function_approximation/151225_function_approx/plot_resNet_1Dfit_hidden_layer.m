[net imdb stats]= load_net_imdb_best_params('~/vlfeat_exp/resNet/poly/poly_lr_1_rprop_0_hidden_6_6_nVar_1_nOrder_2_density_1_poly_eval_n1_sin2_batchSize_300_bn_1_relu/net-epoch-1500.mat');

net_ = dagnn.DagNN.loadobj(net) ;
train_idx = find(imdb.images.set==1);
test_idx = find(imdb.images.set==3);
batch = test_idx(1:30000);

points = imdb.images.data(:,:,:,batch);
points = linearized(points);
[points sorted_idx] = sort(points);
inputs = poly_getDagNNBatch(struct('numGpus',[]), imdb, batch);
for k = 1:numel(net_.vars)
    net_.vars(k).precious = 1;
end
net_.params(2).value = net_.params(2).value*0;
net_.params(5).value(:,1) = net_.params(5).value(:,1)*0; % mu
%net_.params(5).value(:,2) = net_.params(5).value(:,2)*0 + 1; % sigma
net_.params(7).value = net_.params(7).value*0;
%net_.params(10).value(:,1) = net_.params(10).value(:,1)*0; % mu
%net_.params(10).value(:,2) = net_.params(10).value(:,2)*0 + 1; % sigma
net_.params(12).value = net_.params(12).value*0;
net_.mode = 'test';
net_.eval(inputs);

outputs = linearized( net_.vars(end-2).value );
targets = linearized( imdb.images.labels(:,:,:,batch) );
first_relu = linearized( net_.vars(4).value(:,:,6,:) );
second_relu = linearized( net_.vars(8).value(:,:,6,:) );

figure; hold all;
plot(points,targets(sorted_idx),'-.or');
plot(points,outputs(sorted_idx),'-b');
plot(points,first_relu(sorted_idx),'-y');
plot(points,second_relu(sorted_idx),'-k');
legend({'Ground truth','Approximated','1st ReLU','2nd ReLU'});



figure; hold all;
plot(points,targets(sorted_idx),'-.or');
plot(points,outputs(sorted_idx),'-b');
for k = 1:6
    tmp = linearized( net_.vars(8).value(:,:,k,:) );
    plot(points,tmp(sorted_idx));
end
legend({'Ground truth','Approximated','1','2','3','4','5','6'});


%% after first conv
figure; hold all;
plot(points,targets(sorted_idx),'-.or');
plot(points,outputs(sorted_idx),'-b');
for k = 3:4
    tmp = linearized( net_.vars(3).value(:,:,k,:) );
    plot(points,tmp(sorted_idx));
end
legend({'Ground truth','Approximated','1','2','3','4','5','6'});



figure; hold all;
plot(points,targets(sorted_idx),'-.or');
plot(points,outputs(sorted_idx),'-b');
for k = 1:6
    tmp = linearized( net_.vars(4).value(:,:,k,:) );
    plot(points,tmp(sorted_idx));
end
legend({'Ground truth','Approximated','1','2','3','4','5','6'});

