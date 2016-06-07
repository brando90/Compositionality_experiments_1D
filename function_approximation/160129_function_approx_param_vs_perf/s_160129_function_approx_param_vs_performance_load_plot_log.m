%%% 1 layer
addpath ~/Dropbox/loop-share/matlab/y2016spring/151225_function_approx/
all_info_1 = {};
candidates = [1 2:2:16 20:4:32 40:8:64 80:16:256 288:32:384]
%candidates = [3];
gpus = [4];
for i = 1:numel(candidates)
    for r = 1:2
        numEpochs = 20; enable_residual = false; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.001; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [candidates(i) ]; force_batchSize = 300; useBatchNorm = true; poly_eval_func = @poly_eval_n1_sin2; nonlinearity = struct('type','relu');
        if candidates(i) > 32
            lr = 5*lr./candidates(i);
            numEpochs = numEpochs*10;
        end
        expDir = ['~/vlfeat_exp/poly_param_vs_perf/numHidLayer_' num2str(numel(hiddenLayers)) '/resLearn_' num2str(enable_residual) '_poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '/' num2str(r) ];
        working_file = fullfile(expDir,'lock.mat');
        result_file  = fullfile(expDir,'result.mat');
        if ~exist(working_file,'file') && ~exist(result_file,'file')
            %
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

% 2 layer
addpath ~/Dropbox/loop-share/matlab/y2016spring/151225_function_approx/
all_info_2 = {};
candidates = [1 2 4 8 12 16 20 24 28 32]
gpus = [4];
for i = 1:numel(candidates)
    for j = 1:numel(candidates)
        for r = 1:2
            numEpochs = 20; enable_residual = false; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.001; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [candidates(i) candidates(j)]; force_batchSize = 300; useBatchNorm = true; poly_eval_func = @poly_eval_n1_sin2; nonlinearity = struct('type','relu');
            expDir = ['~/vlfeat_exp/poly_param_vs_perf/numHidLayer_' num2str(numel(hiddenLayers)) '/resLearn_' num2str(enable_residual) '_poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '/' num2str(r) ];
            working_file = fullfile(expDir,'lock.mat');
            result_file  = fullfile(expDir,'result.mat');
            if ~exist(working_file,'file') && ~exist(result_file,'file')
                %
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
all_info_3 = {};
gpus = [2];
candidates = [2 4 8 12 16 24]
for i = 1:numel(candidates)
    for j = 1:numel(candidates)
        for k = 1:numel(candidates)
            for r = 1:5
                numEpochs = 20; enable_residual = false; first_layer_func = @return_1st_arg; disable_bias = false; all_t=tic; lr = 0.001; nVar = 1; nOrder = 2; rprop_p = 0; hiddenLayers = [candidates(i) candidates(j) candidates(k)]; force_batchSize = 300; useBatchNorm = true; poly_eval_func = @poly_eval_n1_sin2; nonlinearity = struct('type','relu');
                expDir = ['~/vlfeat_exp/poly_param_vs_perf/numHidLayer_' num2str(numel(hiddenLayers)) '/resLearn_' num2str(enable_residual) '_poly_lr_' num2str(lr) '_rprop_' num2str(rprop_p)  '_hidden_' num_array_to_string_separated_by_underscore(hiddenLayers) '_nVar_' num2str(nVar) '_nOrder_' num2str(nOrder) '_' func2str(poly_eval_func) '_batchSize_' num2str(force_batchSize) '_bn_' num2str(useBatchNorm) '_' nonlinearity.type '_NOBias_' num2str(disable_bias) '_1stLyer_' func2str(first_layer_func) '/' num2str(r) ];
                working_file = fullfile(expDir,'lock.mat');
                result_file  = fullfile(expDir,'result.mat');
                if ~exist(working_file,'file') && ~exist(result_file,'file')
                    %
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






%all_info_train_obj = cellfun_with_same_param (all_info_1,@get_train_min_obj,struct);
%all_info_val_obj   = cellfun_with_same_param (all_info_1,@get_val_min_obj,struct);


%% plot num units vs best train
val_or_train = 'val';
close all; figure;
subplot(1,3,1);hold all;
x = [];y = [];s = [];c = [];
current_info = all_info_1;
for i = 1:numel(current_info)
    x(end+1) = sum(current_info{i}.hiddenLayers(:));
    y(end+1) = log( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 40;
    c(end+1) = 0;
end
colr = 'k';
scatter(x,y,s,'p','MarkerEdgeColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
fx1 = polyval(p,x1);
fx2 = f2(x1); 
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_2;
for i = 1:numel(current_info)
    if abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2))> 0
        continue;
    end
    x(end+1) = sum(current_info{i}.hiddenLayers(:));
    y(end+1) = log( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 10;
    c(end+1) = 0.1;
end
colr = [0 0.5 1];
scatter(x,y,s,'o','MarkerEdgeColor','b','MarkerFaceColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_3;
for i = 1:numel(current_info)
    if (abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2)) > 0) || (abs(current_info{i}.hiddenLayers(2) ~= current_info{i}.hiddenLayers(3)) > 0)
        continue;
    end
    x(end+1) = sum(current_info{i}.hiddenLayers(:));
    y(end+1) = log( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 5;
    c(end+1) = 0.5;
end
colr = [0.2 0.5 0];
scatter(x,y,s,'s','MarkerEdgeColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

legend({'1 hidden pts','1 hidden fit','2 hidden pts','2 hidden fit','3 hidden pts','3 hidden fit'});
if strcmp(val_or_train,'val')
    ylabel('Minimum validation objective of approx. sin(2x)');
else
    ylabel('Minimum training objective of approx. sin(2x)');
end
xlabel('Number of units');


%% plot num params vs best train
%close all; figure; hold all;
subplot(1,3,2);hold all;
x = [];y = [];s = [];c = [];
current_info = all_info_1;
for i = 1:numel(current_info)
    x(end+1) = current_info{i}.numParams;
    y(end+1) = log( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 40;
    c(end+1) = 0;
end
colr = 'k';
scatter(x,y,s,'p','MarkerEdgeColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_2;
for i = 1:numel(current_info)
    if abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2))> 0
        continue;
    end
    x(end+1) = current_info{i}.numParams;
    y(end+1) = log( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 15;
    c(end+1) = 0.1;
end
colr = [0 0.5 1];
scatter(x,y,s,'o','MarkerEdgeColor','b','MarkerFaceColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_3;
for i = 1:numel(current_info)
    if (abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2)) > 0) || (abs(current_info{i}.hiddenLayers(2) ~= current_info{i}.hiddenLayers(3)) > 0)
        continue;
    end
    x(end+1) = current_info{i}.numParams;
    y(end+1) = log( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 7;
    c(end+1) = 0.5;
end
colr = [0.2 0.5 0];
scatter(x,y,s,'s','MarkerEdgeColor',[0.2 0.5 0]);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

%legend({'1 hidden','2 hidden','3 hidden'});
legend({'1 hidden pts','1 hidden fit','2 hidden pts','2 hidden fit','3 hidden pts','3 hidden fit'});
if strcmp(val_or_train,'val')
    ylabel('Minimum validation objective of approx. sin(2x)');
else
    ylabel('Minimum training objective of approx. sin(2x)');
end
xlabel('Number of parameters');



%% plot num params vs best train
%close all; figure; hold all;
subplot(1,3,3);hold all;
x = [];y = [];s = [];c = [];
current_info = all_info_1;
for i = 1:numel(current_info)
    x(end+1) = current_info{i}.numParams*sum(current_info{i}.hiddenLayers(:));
    y(end+1) = log( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 40;
    c(end+1) = 0;
end
colr = 'k';
scatter(x,y,s,'p','MarkerEdgeColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_2;
for i = 1:numel(current_info)
    if abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2))> 0
        continue;
    end
    x(end+1) = current_info{i}.numParams*sum(current_info{i}.hiddenLayers(:));
    y(end+1) = log( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 15;
    c(end+1) = 0.1;
end
colr = [0 0.5 1];
scatter(x,y,s,'o','MarkerEdgeColor','b','MarkerFaceColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_3;
for i = 1:numel(current_info)
    if (abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2)) > 0) || (abs(current_info{i}.hiddenLayers(2) ~= current_info{i}.hiddenLayers(3)) > 0)
        continue;
    end
    x(end+1) = current_info{i}.numParams*sum(current_info{i}.hiddenLayers(:));
    y(end+1) = log( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 7;
    c(end+1) = 0.5;
end
colr = [0.2 0.5 0];
scatter(x,y,s,'s','MarkerEdgeColor',[0.2 0.5 0]);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

%legend({'1 hidden','2 hidden','3 hidden'});
legend({'1 hidden pts','1 hidden fit','2 hidden pts','2 hidden fit','3 hidden pts','3 hidden fit'});
if strcmp(val_or_train,'val')
    ylabel('Minimum validation objective of approx. sin(2x)');
else
    ylabel('Minimum training objective of approx. sin(2x)');
end
xlabel('#parameters *  #units');

