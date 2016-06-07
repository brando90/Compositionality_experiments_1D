%% 1 layer
addpath ~/Dropbox/loop-share/matlab/matconvnet-1.0-beta17_polestar/examples/mnist/
all_info = {};
%candidates = [1 2:2:16 20:4:32 40:8:64 80:16:256 288:32:384]
candidates = [10 20 40 60 80 100 120]
%candidates = [384]
gpus = [1];
for i = 1:numel(candidates)
    for r = 1
        hiddenLayers = [candidates(i)]; useBatchNorm = true;
        expDir =  ['~/vlfeat_exp/poly_param_vs_perf/mnist/hidden_' num2str(numel(hiddenLayers)) '/' num2str_underscore(hiddenLayers)];
        working_file = fullfile(expDir,'lock.mat');
        result_file  = fullfile(expDir,'result.mat');
        if ~exist(working_file,'file') && ~exist(result_file,'file')
            mkdir_if_not_exist_for_a_file(working_file);
            lock = 1;
            save(working_file,'lock');
             [net, info] = cnn_mnist_FC(struct('expDir',expDir,'dataDir','~/vlfeat_exp/matconvnet_data/mnist/','hiddenLayers',hiddenLayers,'useBatchNorm',useBatchNorm,'gpus',gpus));
            save(result_file,'net','info');
            delete(working_file);
            all_info_1{i,r} = info;
        elseif exist(result_file,'file')
            load(result_file);
            info.hiddenLayers = hiddenLayers;
            info.numParams = dagnn_count_params_of_conv(net);
            all_info_1{i,r} = info;
        end
    end
end



% 2 layer
addpath ~/Dropbox/loop-share/matlab/matconvnet-1.0-beta17_polestar/examples/mnist/
all_info = {};
candidates = [10 20 40 60 80 100]
gpus = [1];
for i = 1:numel(candidates)
    for r = 1
        hiddenLayers = [candidates(i) candidates(i)]; useBatchNorm = true;
        expDir =  ['~/vlfeat_exp/poly_param_vs_perf/mnist/hidden_' num2str(numel(hiddenLayers)) '/' num2str_underscore(hiddenLayers)];
        working_file = fullfile(expDir,'lock.mat');
        result_file  = fullfile(expDir,'result.mat');
        if ~exist(working_file,'file') && ~exist(result_file,'file')
            mkdir_if_not_exist_for_a_file(working_file);
            lock = 1;
            save(working_file,'lock');
             [net, info] = cnn_mnist_FC(struct('expDir',expDir,'dataDir','~/vlfeat_exp/matconvnet_data/mnist/','hiddenLayers',hiddenLayers,'useBatchNorm',useBatchNorm,'gpus',gpus));
            save(result_file,'net','info');
            delete(working_file);
            all_info_2{i,r} = info;
        elseif exist(result_file,'file')
            load(result_file);
            info.hiddenLayers = hiddenLayers;
            info.numParams = dagnn_count_params_of_conv(net);
            all_info_2{i,r} = info;
        end
    end
end





% 3 layer
addpath ~/Dropbox/loop-share/matlab/matconvnet-1.0-beta17_polestar/examples/mnist/
all_info = {};
gpus = [1];
candidates = [10 20 40 60 80]
%candidates = [26]
for i = 1:numel(candidates)
    for r = 1
        hiddenLayers = [candidates(i) candidates(i) candidates(i)]; useBatchNorm = true;
        expDir =  ['~/vlfeat_exp/poly_param_vs_perf/mnist/hidden_' num2str(numel(hiddenLayers)) '/' num2str_underscore(hiddenLayers)];
        working_file = fullfile(expDir,'lock.mat');
        result_file  = fullfile(expDir,'result.mat');
        if ~exist(working_file,'file') && ~exist(result_file,'file')
            mkdir_if_not_exist_for_a_file(working_file);
            lock = 1;
            save(working_file,'lock');
             [net, info] = cnn_mnist_FC(struct('expDir',expDir,'dataDir','~/vlfeat_exp/matconvnet_data/mnist/','hiddenLayers',hiddenLayers,'useBatchNorm',useBatchNorm,'gpus',gpus));
            save(result_file,'net','info');
            delete(working_file);
            all_info_3{i,r} = info;
        elseif exist(result_file,'file')
            load(result_file);
            info.hiddenLayers = hiddenLayers;
            info.numParams = dagnn_count_params_of_conv(net);
            all_info_3{i,r} = info;
        end
    end
end

