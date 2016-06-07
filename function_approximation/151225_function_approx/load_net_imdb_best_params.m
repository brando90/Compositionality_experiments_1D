function [net imdb stats]= load_net_imdb_best_params(net_file)
load(net_file);
a = fileparts(net_file);
imdb = load(fullfile(a,'imdb.mat'));
net.params = stats.params_best;
