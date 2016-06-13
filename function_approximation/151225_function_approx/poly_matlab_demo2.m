% Solve an Input-Output Fitting problem with a Neural Network
% Script generated by NFTOOL
%
% This script assumes these variables are defined:
%
%   houseInputs - input data.
%   houseTargets - target data.
 
rehash toolbox;

close all;
inputs = randn(2,100000);
targets = gt_func(inputs);

gt_func = @(x)  x(1,:).^4 + 1.5*x(2,:).^4;

 
% Create a Fitting Network
hiddenLayerSize = [7 7];
net = fitnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%net.layers{1}.transferFcn = 'tansig';
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'poslin';
% for i = 1:numel(net.layers)-1
%     net.layers{i}.transferFcn = 'hardlim';
% end
% net.layers{3}.transferFcn = 'poslin';
% Train the Network
%net.trainFcn = 'traingd';
%net.trainFcn = 'trainscg'; % reasonable
%net.trainFcn = 'traincgf'; % reasonable to good
%net.trainFcn = 'traincgp'; % reasonable
%clear trainmybm
%net.trainFcn = 'trainmybm'; % reasonable to good!
net.trainFcn = 'trainlm'; % best
%net.trainFcn = 'trainbfg'; % poor to reasonable
%net.trainFcn = 'trainrp'; % 
[net,tr] = train(net,inputs,targets,'useGPU','yes','showResources','yes');
% [net,tr] = train(net,inputs,targets);

[X,Y] = meshgrid(min(inputs(1,:)):0.1:max(inputs(1,:)),min(inputs(2,:)):0.1:max(inputs(2,:)));
surf_in = [X(:)'; Y(:)'];
Z = net(surf_in);
Z = reshape(Z,size(X));
Z_gt = gt_func(surf_in);
Z_gt = reshape(Z_gt,size(X));
figure;surf(X,Y,Z);
figure;surf(X,Y,Z_gt);


% % Test the Network
% outputs = net(inputs);
% errors = gsubtract(outputs,targets);
% performance = perform(net,targets,outputs)
 
% % View the Network
% view(net)
 
% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotfit(targets,outputs)
% figure, plotregression(targets,outputs)
% figure, ploterrhist(errors)