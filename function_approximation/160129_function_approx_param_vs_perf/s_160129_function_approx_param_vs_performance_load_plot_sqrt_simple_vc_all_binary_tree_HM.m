%% plot num units vs best train
%function_name  = 'sin(x+2y)+cos(3x+2y)';
%function_name  = 'sin(2x)'
%function_name  = 'MNIST';
%function_name  = '16x^4-48x^2+12';
%function_name  = 'binary_tree, |left - right|+';
%function_name  = 'binary_tree, |left - right|';
%function_name  = 'binary_tree, cosine';
function_name  = 'Function suggested by Mhaskar';
%function_name  = '-48x^2+12';
%function_name  = 'sin(4Dcomb)';
%function_name  = '(x1+x2)^2  + 1.5*(x3+x4)^2';
%function_name  = 'Degree 2 polynomial, all monomials, uniform weights, 1/10 learning rates'
%function_name  = 'Degree 2 polynomial, all monomials, random weights'
%function_name  = 'simple 1 layer'
%function_name  = 'simple 2 layers'

val_or_train = 'val';
close all; figure;
suptitle(function_name);
subplot(1,2,1);hold all;
set(gcf,'position',get(0,'screensize'))
x = [];y = [];s = [];c = [];
current_info = all_info_1;
for i = 1:numel(current_info)
    try
        x(end+1) = sum(current_info{i}.hiddenLayers(:));
        y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
        s(end+1) = 40;
        c(end+1) = 0;
    end
end
colr = 'k';
scatter(x,y,s,'p','MarkerEdgeColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
%fx1 = polyval(p,x1);
fx2 = f2(x1); 
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_4;
for i = 1:numel(current_info)
    % if (abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2)) > 0) || (abs(current_info{i}.hiddenLayers(2) ~= current_info{i}.hiddenLayers(3)) > 0)
    %     continue;
    % end
    try
        x(end+1) = sum(current_info{i}.hiddenLayers(:));
        y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
        s(end+1) = 14.99;
        c(end+1) = 0.5;
    end
end
colr = [0.2 0.5 0];
scatter(x,y,s,'s','MarkerEdgeColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
%fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

legend({'1 hidden pts','1 hidden fit','4 hidden pts','4 hidden fit'});
if strcmp(val_or_train,'val')
    ylabel(['Minimum validation (sqrt) objective  ']);
else
    ylabel(['Minimum training (sqrt) objective  ']);
end
xlabel('Number of units');











val_or_train = 'val';
subplot(1,2,2);hold all;
set(gcf,'position',get(0,'screensize'))
x = [];y = [];s = [];c = [];
current_info = all_info_1;
for i = 1:numel(current_info)
    try
        x(end+1) = current_info{i}.numParams;
        y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
        s(end+1) = 40;
        c(end+1) = 0;
    end
end
colr = 'k';
scatter(x,y,s,'p','MarkerEdgeColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
%fx1 = polyval(p,x1);
fx2 = f2(x1); 
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_4;
for i = 1:numel(current_info)
    % if (abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2)) > 0) || (abs(current_info{i}.hiddenLayers(2) ~= current_info{i}.hiddenLayers(3)) > 0)
    %     continue;
    % end
    try
        x(end+1) = current_info{i}.numParams;
        y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
        s(end+1) = 14.99;
        c(end+1) = 0.5;
    end
end
colr = [0.2 0.5 0];
scatter(x,y,s,'s','MarkerEdgeColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
%fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

legend({'1 hidden pts','1 hidden fit','4 hidden pts','4 hidden fit'});
if strcmp(val_or_train,'val')
    ylabel(['Minimum validation (sqrt) objective  ']);
else
    ylabel(['Minimum training (sqrt) objective  ']);
end
xlabel('Number of parameters');
