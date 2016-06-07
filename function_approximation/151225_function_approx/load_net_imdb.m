function [net imdb stats]= load_net_imdb(net_file)
load(net_file);
a = fileparts(net_file);
imdb = load(fullfile(a,'imdb.mat'));
