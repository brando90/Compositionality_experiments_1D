% -------------------------------------------------------------------------
function [images, labels] = poly_getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(:,:,:,batch) ;
images = gpuArray(images) ;

%inputs = {'input', images, 'label', labels} ;

