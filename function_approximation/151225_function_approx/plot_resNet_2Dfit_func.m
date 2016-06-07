function  plot_resNet_2Dfit_func(net,imdb,newFig)
if ~exist('newFig','var') || newFig
    close all;
end


net_ = dagnn.DagNN.loadobj(net) ;
train_idx = find(imdb.images.set==1);
test_idx = find(imdb.images.set==3);
batch = test_idx(1:30000);

points = imdb.images.data(:,:,:,batch);
if size(points,1) ~= 1
    x = linearized(points(1,:,:,:)); % 1st dim of input
    y = linearized(points(2,:,:,:)); % 2nd dim of input
else
    x = linearized(points(:,:,1,:));
    y = linearized(points(:,:,2,:));
end
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


subplot(1,2,1);
surf_from_scatter_auto(x,y,outputs);
set(gca,'CameraPosition',[0 0 20]);
subplot(1,2,2);
surf_from_scatter_auto(x,y,targets);
set(gca,'CameraPosition',[0 0 20]);
%camorbit(43,0);

%surf_from_scatter_auto(x,y,outputs);
%stem3(x, y, targets, 'k', 'fill')             % Original Data
