%% 1 layer
addpath ~/Dropbox/loop-share/matlab/y2016spring/151225_function_approx/
all_info = {};
candidates = [1 2:2:16 20:4:32 40:8:64 80:16:256 288:32:384]
gpus = [3];
for i = 1:numel(candidates)
    for r = 1
        numEpochs = 200; enable_residual = false; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.001; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [candidates(i) ]; force_batchSize = 3000; useBatchNorm = true; poly_eval_func = @poly_eval_n1_16x4_neg48x2_12; domain = [-2 2]; nonlinearity = struct('type','relu');
        if candidates(i) > 32
            lr = 5*lr./candidates(i);
            numEpochs = numEpochs*10;
        end
        expDir = ['~/vlfeat_exp/poly_param_vs_perf/bs' num2str(force_batchSize) '/numHidLayer_' num2str(numel(hiddenLayers)) '/resLearn_' num2str(enable_residual) '_poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '/' num2str(r) ];
        working_file = fullfile(expDir,'lock.mat');
        result_file  = fullfile(expDir,'result.mat');
        if ~exist(working_file,'file') && ~exist(result_file,'file')
            mkdir_if_not_exist_for_a_file(working_file);
            lock = 1;
            save(working_file,'lock');
            [net, info] = poly_run_resNet('domain',domain,'expDir', expDir,'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',gpus,'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.1,'weightDecay',0,'checkPoints',[0 numEpochs/2 numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'enable_residual',enable_residual,'no_plot',true); toc(all_t); % --- square, the same architecture
            save(result_file,'net','info');
            delete(working_file);
            all_info{i,r} = info;
        elseif exist(result_file,'file')
            load(result_file);
            info.hiddenLayers = hiddenLayers;
            info.numParams = dagnn_count_params_of_conv(net);
            all_info_1{i,r} = info;
        end
    end
end


%all_info_train_obj = cellfun_with_same_param (all_info_1,@get_train_min_obj,struct);
%all_info_val_obj   = cellfun_with_same_param (all_info_1,@get_val_min_obj,struct);
%all_info_train_obj = cellfun_with_same_param (all_info_3,@get_train_min_obj,struct)

% 2 layer
addpath ~/Dropbox/loop-share/matlab/y2016spring/151225_function_approx/
all_info = {};
candidates = [1 2 4 8 12 16 20 24 28 32]
gpus = [2];
for i = 1:numel(candidates)
    for j = 1:numel(candidates)
        for r = 1:2
            numEpochs = 200; enable_residual = false; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.001; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [candidates(i) candidates(j)]; force_batchSize = 3000; useBatchNorm = true; poly_eval_func = @poly_eval_n1_16x4_neg48x2_12; domain = [-2 2]; nonlinearity = struct('type','relu');
            expDir = ['~/vlfeat_exp/poly_param_vs_perf/bs' num2str(force_batchSize) '/numHidLayer_' num2str(numel(hiddenLayers)) '/resLearn_' num2str(enable_residual) '_poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '/' num2str(r) ];
            working_file = fullfile(expDir,'lock.mat');
            result_file  = fullfile(expDir,'result.mat');
            if ~exist(working_file,'file') && ~exist(result_file,'file')
                mkdir_if_not_exist_for_a_file(working_file);
                lock = 1;
                save(working_file,'lock');
                [net, info] = poly_run_resNet('domain',domain,'expDir', expDir,'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',gpus,'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.1,'weightDecay',0,'checkPoints',[0 numEpochs/2 numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'enable_residual',enable_residual,'no_plot',true); toc(all_t); % --- square, the same architecture
                save(result_file,'net','info');
                delete(working_file);
                all_info{i,j,r} = info;
            elseif exist(result_file,'file')
                load(result_file);
                info.hiddenLayers = hiddenLayers;
                info.numParams = dagnn_count_params_of_conv(net);
                all_info_2{i,j,r} = info;
            end
        end
    end
end







% 3 layer
addpath ~/Dropbox/loop-share/matlab/y2016spring/151225_function_approx/
all_info = {};
gpus = [4];
candidates = [2 4 8 12 16 24]
for i = 1:numel(candidates)
    for j = 1:numel(candidates)
        for k = 1:numel(candidates)
            for r = 1:2
                numEpochs = 200; enable_residual = false; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.001; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [candidates(i) candidates(j) candidates(k)]; force_batchSize = 3000; useBatchNorm = true; poly_eval_func = @poly_eval_n1_16x4_neg48x2_12; domain = [-2 2]; nonlinearity = struct('type','relu');
                expDir = ['~/vlfeat_exp/poly_param_vs_perf/bs' num2str(force_batchSize) '/numHidLayer_' num2str(numel(hiddenLayers)) '/resLearn_' num2str(enable_residual) '_poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '/' num2str(r) ];
                working_file = fullfile(expDir,'lock.mat');
                result_file  = fullfile(expDir,'result.mat');
                if ~exist(working_file,'file') && ~exist(result_file,'file')
                    mkdir_if_not_exist_for_a_file(working_file);
                    lock = 1;
                    save(working_file,'lock');
                    [net, info] = poly_run_resNet('domain',domain,'expDir', expDir,'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',gpus,'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity.type,'learningRate', [ ones(1,numEpochs) ]*lr,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.1,'weightDecay',0,'checkPoints',[0 numEpochs/2 numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'enable_residual',enable_residual,'no_plot',true); toc(all_t); % --- square, the same architecture
                    save(result_file,'net','info');
                    delete(working_file);
                    all_info{i,j,k,r} = info;
                elseif exist(result_file,'file')
                    load(result_file);
                    info.hiddenLayers = hiddenLayers;
                    info.numParams = dagnn_count_params_of_conv(net);
                    all_info_3{i,j,k,r} = info;
                end
            end
        end
    end
end


