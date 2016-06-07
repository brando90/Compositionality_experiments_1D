%% 1 layer
addpath ~/Dropbox/loop-share/matlab/y2016spring/151225_function_approx/
all_info = {};
nVar = 16;
%candidates = [1 2 4 6 10];
candidates = [2 6 10]*log2(nVar);
gpus = [1];
for i = 1:numel(candidates)
    for r = 1:100
        numEpochs = 200; enable_residual = false; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.00001; nOrder = 2; rprop_p = 0; hiddenLayers = [candidates(i)]; force_batchSize = 300; useBatchNorm = true; poly_eval_func = @poly_eval_spacial_binary_tree_abs_v01_n16; nonlinearity = struct('type','relu');
        nOut = 1; domain = [-2*pi 2*pi];
        expDir = ['~/vlfeat_exp/poly_param_vs_perf/binary_tree_2/numEpochs_' num2str(numEpochs) '_bs' num2str(force_batchSize) '/numHidLayer_' num2str(numel(hiddenLayers)) '/resLearn_' num2str(enable_residual) '_poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '/' num2str(r) '_domain_' num2str_underscore(domain) ];
        working_file = fullfile(expDir,'lock.mat');
        result_file  = fullfile(expDir,'result.mat');
        if ~exist(working_file,'file') && ~exist(result_file,'file')
            mkdir_if_not_exist_for_a_file(working_file);
            lock = 1;
            save(working_file,'lock');
            [net, info] = poly_run_resNet('domain',domain,'expDir', expDir,'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',gpus,'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0 numEpochs/2 numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'enable_residual',enable_residual,'no_plot',true); toc(all_t); % --- square, the same architecture
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





%% 4 layer
addpath ~/Dropbox/loop-share/matlab/y2016spring/151225_function_approx/
all_info = {};
nVar = 16;
candidates = [2 6 10];
gpus = [1];
for i = 1:numel(candidates)
    for r = 1:100
        numEpochs = 200; enable_residual = false; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.00001; nOrder = 2; rprop_p = 0; hiddenLayers = candidates(i)*ones(1,log2(nVar)); force_batchSize = 300; useBatchNorm = true; poly_eval_func = @poly_eval_spacial_binary_tree_abs_v01_n16; nonlinearity = struct('type','relu');
        nOut = 1; domain = [-2*pi 2*pi];
        expDir = ['~/vlfeat_exp/poly_param_vs_perf/binary_tree_2/numEpochs_' num2str(numEpochs) '_bs' num2str(force_batchSize) '/numHidLayer_' num2str(numel(hiddenLayers)) '/resLearn_' num2str(enable_residual) '_poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '/' num2str(r) '_domain_' num2str_underscore(domain) ];
        working_file = fullfile(expDir,'lock.mat');
        result_file  = fullfile(expDir,'result.mat');
        if ~exist(working_file,'file') && ~exist(result_file,'file')
            mkdir_if_not_exist_for_a_file(working_file);
            lock = 1;
            save(working_file,'lock');
            [net, info] = poly_run_resNet('domain',domain,'expDir', expDir,'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',gpus,'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0 numEpochs/2 numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'enable_residual',enable_residual,'no_plot',true); toc(all_t); % --- square, the same architecture
            save(result_file,'net','info');
            delete(working_file);
            all_info_4{i,r} = info;
        elseif exist(result_file,'file')
            load(result_file);
            info.hiddenLayers = hiddenLayers;
            info.numParams = dagnn_count_params_of_conv(net);
            all_info_4{i,r} = info;
        end
    end
end



