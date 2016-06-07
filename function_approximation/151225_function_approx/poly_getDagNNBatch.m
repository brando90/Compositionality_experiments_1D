% -------------------------------------------------------------------------
function inputs = poly_getDagNNBatch(opts, imdb, batch)
% inputs = poly_getDagNNBatch(struct('numGpus',[1]), imdb, 1:500);
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(:,:,:,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;
