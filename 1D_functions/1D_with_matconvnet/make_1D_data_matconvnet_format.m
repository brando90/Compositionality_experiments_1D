function [ imdb, X_train, Y_train, X_test, Y_test ] = make_1D_data_matconvnet_format( N_train, N_test, X_train, Y_train, X_test, Y_test )
%%
X_train = X_train(1:N_train,:); % (N_train  x 1)
Y_train = Y_train(1:N_train,:); % (N_train  x 1)
X_test = X_test(1:N_test,:); % (N_test  x 1)
Y_test = Y_test(1:N_test,:); % (N_test  x 1)
%%
imdb = struct();
imdb.images.data = zeros(1,1,1,N_train+N_test);
imdb.images.label = zeros(1,1,1,N_train+N_test);
imdb.images.data(1,1,1,:) = [X_train; X_test]';
imdb.images.label(1,1,1,:) = [Y_train; Y_test]';
%% a 1×n1×n vector containing a 1 for training images and an 2 for validation images.
split = 2*ones(1,N_train+N_test); % 2 for validation images.
split(1:N_train) = 1; % 1 for training images
imdb.images.set = split;
%%
%%
X_train = zeros(1,1,1,N_train); % (N_train  x 1)
Y_train = zeros(1,1,1,N_train); % (N_train  x 1)
X_test = zeros(1,1,1,N_test); % (N_test  x 1)
Y_test = zeros(1,1,1,N_test); % (N_test  x 1)
%
X_train(1,1,1,:) = X_train(1:N_train,:); % (N_train  x 1)
Y_train(1,1,1,:) = Y_train(1:N_train,:); % (N_train  x 1)
X_test(1,1,1,:) = X_test(1:N_test,:); % (N_test  x 1)
Y_test(1,1,1,:) = Y_test(1:N_test,:); % (N_test  x 1)
end