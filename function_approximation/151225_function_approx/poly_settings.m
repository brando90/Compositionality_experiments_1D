% binary tree,  relu sq
addpath ~/Dropbox/loop-share/matlab/y2016spring/151225_function_approx/ 
no_last_bnorm = false; nVar = 16; nOut = 1; domain = [-pi pi]; numEpochs = 100; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.0001; nOrder = 2; rprop_p = 0; thickness = 2; hiddenLayers = [thickness thickness thickness thickness]; force_batchSize = 300; useBatchNorm = true; density = 1; poly_eval_func = @poly_eval_spacial_binary_tree_sin_v02_n16_pi; nonlinearity = @vl_nnrelu_square_orig;  [net, info] = poly_run_resNet('nOut',nOut,'domain',domain,'expDir',['~/vlfeat_exp/resNet/poly_2/poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' func2str_2(nonlinearity) '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '_noLastBN_' num2str(no_last_bnorm)  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity,'learningRate', [ ones(1,numEpochs) ]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0:20:numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'no_last_bnorm',no_last_bnorm); toc(all_t); 



no_last_bnorm = false; nVar = 16; nOut = 1; domain = [-pi pi]; numEpochs = 100; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.0001; nOrder = 2; rprop_p = 0; thickness = 10*4; hiddenLayers = [thickness]; force_batchSize = 300; useBatchNorm = true; density = 1; poly_eval_func = @poly_eval_spacial_binary_tree_sin_v02_n16_pi; nonlinearity = @vl_nnrelu_square_orig;  [net, info] = poly_run_resNet('nOut',nOut,'domain',domain,'expDir',['~/vlfeat_exp/resNet/poly_2/poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' func2str_2(nonlinearity) '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '_noLastBN_' num2str(no_last_bnorm)  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity,'learningRate', [ ones(1,numEpochs) ]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0:20:numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'no_last_bnorm',no_last_bnorm); toc(all_t); 
























% cosine
no_last_bnorm = false; nVar = 16; nOut = 1; domain = [-5 5]; numEpochs = 100; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.0001; nOrder = 2; rprop_p = 0; thickness = 10; hiddenLayers = [thickness thickness thickness thickness]; force_batchSize = 300; useBatchNorm = true; density = 1; poly_eval_func = @poly_eval_spacial_binary_tree_cos_v02_n16; nonlinearity = @vl_nnrelu;  [net, info] = poly_run_resNet('nOut',nOut,'domain',domain,'expDir',['~/vlfeat_exp/resNet/poly_2/poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder)  '_density_' num2str(density) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' func2str_2(nonlinearity) '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '_noLastBN_' num2str(no_last_bnorm)  ],'dataDir','~/vlfeat_exp/matconvnet_data/cifar/','replaceFunc',[],'gpus',[1],'nVar',nVar,'nOrder',nOrder,'hiddenLayers',hiddenLayers,'nonlinearity', nonlinearity,'learningRate', [ ones(1,numEpochs) ]*lr,'density',density,'networkType','dagnn','useBatchNorm',useBatchNorm,'poly_eval_func',poly_eval_func,'force_batchSize',force_batchSize,'momentum',0.9,'weightDecay',0,'checkPoints',[0:20:numEpochs],'rprop_p',rprop_p,'func_every_epoch',[],'disable_bias',disable_bias,'first_layer_func',first_layer_func,'no_last_bnorm',no_last_bnorm); toc(all_t); 