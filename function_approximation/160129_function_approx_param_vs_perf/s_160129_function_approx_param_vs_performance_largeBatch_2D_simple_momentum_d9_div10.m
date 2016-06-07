%% 1 layer
addpath ~/Dropbox/loop-share/matlab/y2016spring/151225_function_approx/
all_info = {};
%candidates = [1 2:2:16 20:4:32 40:8:64 80:16:256 288:32:384]
%candidates = [1 2:2:16 20:4:32 48:16:128 160:32:384]
candidates = [24 48 72 128 256]
%candidates = [384]
gpus = [1];
for i = 1:numel(candidates)
    for r = 1:5
        numEpochs = 200; enable_residual = false; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.001; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [candidates(i)]; force_batchSize = 3000; useBatchNorm = true; poly_eval_func = @poly_eval_n1_cos_pow4; nonlinearity = struct('type','relu');
        lr = lr/10;
        numEpochs = numEpochs*10;
        expDir = ['~/vlfeat_exp/poly_param_vs_perf/uai_01/div10_1D_bs' num2str(force_batchSize) '/numHidLayer_' num2str(numel(hiddenLayers)) '/resLearn_' num2str(enable_residual) '_poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '/' num2str(r) ];
        working_file = fullfile(expDir,'lock.mat');
        result_file  = fullfile(expDir,'result.mat');
        if ~exist(working_file,'file') && ~exist(result_file,'file')
            mkdir_if_not_exist_for_a_file(working_file);
            lock = 1;
            save(working_file,'lock');
            [net, info] = poly_run_resNet('expDir', expDir,'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',gpus,'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0 numEpochs/2 numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'enable_residual',enable_residual,'no_plot',true); toc(all_t); % --- square, the same architecture
            save(result_file,'net','info');
            delete(working_file);
            all_info_1{i,r} = info;
        elseif exist(result_file,'file')
            load(result_file);
            info.hiddenLayers = hiddenLayers;
            info.numParams = dagnn_count_params_of_conv(net);
            all_info_1{i,r} = info;
            all_net_1{i,r} = net;
        end
    end
end


%all_info_train_obj = cellfun_with_same_param (all_info_1,@get_train_min_obj,struct);
%all_info_val_obj   = cellfun_with_same_param (all_info_1,@get_val_min_obj,struct);
%all_info_train_obj = cellfun_with_same_param (all_info_3,@get_train_min_obj,struct)

% 2 layer
addpath ~/Dropbox/loop-share/matlab/y2016spring/151225_function_approx/
all_info = {};
candidates = [24 48 72]/2
gpus = [1];
for i = 1:numel(candidates)
    for r = 1:5
        numEpochs = 200; enable_residual = false; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.001; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [candidates(i) candidates(i)]; force_batchSize = 3000; useBatchNorm = true; poly_eval_func = @poly_eval_n1_cos_pow4; nonlinearity = struct('type','relu');
        lr = lr/10;
        numEpochs = numEpochs*10;
        expDir = ['~/vlfeat_exp/poly_param_vs_perf/uai_01/div10_1D_bs' num2str(force_batchSize) '/numHidLayer_' num2str(numel(hiddenLayers)) '/resLearn_' num2str(enable_residual) '_poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '/' num2str(r) ];
        working_file = fullfile(expDir,'lock.mat');
        result_file  = fullfile(expDir,'result.mat');
        if ~exist(working_file,'file') && ~exist(result_file,'file')
            mkdir_if_not_exist_for_a_file(working_file);
            lock = 1;
            save(working_file,'lock');
            [net, info] = poly_run_resNet('expDir', expDir,'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',gpus,'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0 numEpochs/2 numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'enable_residual',enable_residual,'no_plot',true); toc(all_t); % --- square, the same architecture
            save(result_file,'net','info');
            delete(working_file);
            all_info_2{i,r} = info;
        elseif exist(result_file,'file')
            load(result_file);
            info.hiddenLayers = hiddenLayers;
            info.numParams = dagnn_count_params_of_conv(net);
            all_info_2{i,r} = info;
            all_net_2{i,r} = net;
        end
    end
end







% 3 layer
addpath ~/Dropbox/loop-share/matlab/y2016spring/151225_function_approx/
all_info = {};
gpus = [1];
candidates = [24 48 72]/3
%candidates = [26]
all_net_3 = {};
for i = 1:numel(candidates)
    for r = 1:5
        numEpochs = 200; enable_residual = false; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.001; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [candidates(i) candidates(i) candidates(i)]; force_batchSize = 3000; useBatchNorm = true; poly_eval_func = @poly_eval_n1_cos_pow4; nonlinearity = struct('type','relu');
        lr = lr/10;
        numEpochs = numEpochs*10;
        expDir = ['~/vlfeat_exp/poly_param_vs_perf/uai_01/div10_1D_bs' num2str(force_batchSize) '/numHidLayer_' num2str(numel(hiddenLayers)) '/resLearn_' num2str(enable_residual) '_poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '/' num2str(r) ];
        working_file = fullfile(expDir,'lock.mat');
        result_file  = fullfile(expDir,'result.mat');
        if ~exist(working_file,'file') && ~exist(result_file,'file')
            mkdir_if_not_exist_for_a_file(working_file);
            lock = 1;
            save(working_file,'lock');
            [net, info] = poly_run_resNet('expDir', expDir,'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',gpus,'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0 numEpochs/2 numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'enable_residual',enable_residual,'no_plot',true); toc(all_t); % --- square, the same architecture
            save(result_file,'net','info');
            delete(working_file);
            all_info_3{i,r} = info;
        elseif exist(result_file,'file')
            load(result_file);
            info.hiddenLayers = hiddenLayers;
            info.numParams = dagnn_count_params_of_conv(net);
            all_info_3{i,r} = info;
            all_net_3{i,r} = net;
        end
    end
end

% imdb = load(strrep(result_file,'result.mat','imdb.mat'));
%  plot_resNet_1Dfit_func(all_net_3{3,5},imdb,true)

%% best model
%  all_net_3{3,5}.params
%  all_info_3{3,5}.val(end)


%% net = copy(  all_net_3{3,5}   );
%  imdb = load(strrep(result_file,'result.mat','imdb.mat'));
% net = copy(  all_net_3{3,5} );
% my_dagMergeBatchNorm(net)
% plot_resNet_1Dfit_func(net,imdb,true)
%  plot_resNet_1Dfit_func(all_net_3{3,5},imdb,true)