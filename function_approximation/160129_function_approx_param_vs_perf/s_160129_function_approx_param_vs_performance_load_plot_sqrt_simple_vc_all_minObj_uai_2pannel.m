%% plot num units vs best train

%function_name  = 'sin(x+2y)+cos(3x+2y)';
%function_name  = 'sin(2x)'
%function_name  = 'MNIST';
%function_name  = '16x^4-48x^2+12';
function_name  = 'f(x) = 2(2cos^2(x) - 1)^2-1';
%function_name  = '-48x^2+12';
%function_name  = 'sin(4Dcomb)';
%function_name  = '(x1+x2)^2  + 1.5*(x3+x4)^2';
%function_name  = 'Degree 2 polynomial, all monomials, uniform weights, 1/10 learning rates'
%function_name  = 'Degree 2 polynomial, all monomials, random weights'
%function_name  = 'simple 1 layer'
%function_name  = 'simple 2 layers'
sqrt = @return_1st_arg;

val_or_train = 'val';
close all; figure;
suptitle(function_name);
subplot(1,2,1);hold all;
set(gcf,'position',get(0,'screensize'))
x = [];y = [];s = [];c = [];
current_info = all_info_1;
for i = 1:numel(current_info)
    x(end+1) = sum(current_info{i}.hiddenLayers(:));
    y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 40;
    c(end+1) = 0;
end
colr = 'k';

ux = []; uy = []; unique_x = unique(x);
for i = 1:numel(unique_x)
    idx = find( x == unique_x(i));
    ux(i) = unique_x(i);
    uy(i) = min(y(idx));
end
plot(ux,uy,'LineWidth',3,'Color',colr);%ylim([1.5 2.5]);

% scatter(x,y,s,'p','MarkerEdgeColor',colr);
% f1 = polyfit(x,y,1);
% f2 = fit(x',y','exp1');
% x1 = linspace(0,max(x(:)),100);
% %fx1 = polyval(p,x1);
% fx2 = f2(x1); 
% [fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_2;
for i = 1:numel(current_info)
    % if abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2))> 0
    %     continue;
    % end
    x(end+1) = sum(current_info{i}.hiddenLayers(:));
    y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 15.01;
    c(end+1) = 0.1;
end

colr = [0 0.5 1];
ux = []; uy = []; unique_x = unique(x);
for i = 1:numel(unique_x)
    idx = find( x == unique_x(i));
    ux(i) = unique_x(i);
    uy(i) = min(y(idx));
end
plot(ux,uy,'LineWidth',3,'Color',colr);%ylim([1.5 2.5]);

% scatter(x,y,s,'o','MarkerEdgeColor','b','MarkerFaceColor',colr);
% f1 = polyfit(x,y,1);
% f2 = fit(x',y','exp1');
% x1 = linspace(0,max(x(:)),100);
% %fx1 = polyval(p,x1);
% fx2 = f2(x1);
% [fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_3;
for i = 1:numel(current_info)
    % if (abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2)) > 0) || (abs(current_info{i}.hiddenLayers(2) ~= current_info{i}.hiddenLayers(3)) > 0)
    %     continue;
    % end
    x(end+1) = sum(current_info{i}.hiddenLayers(:));
    y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 14.99;
    c(end+1) = 0.5;
end
colr = [0.2 0.5 0];

ux = []; uy = []; unique_x = unique(x);
for i = 1:numel(unique_x)
    idx = find( x == unique_x(i));
    ux(i) = unique_x(i);
    uy(i) = min(y(idx));
end
plot(ux,uy,'LineWidth',3,'Color',colr);%ylim([1.5 2.5]);



% scatter(x,y,s,'s','MarkerEdgeColor',colr);
% f1 = polyfit(x,y,1);
% f2 = fit(x',y','exp1');
% x1 = linspace(0,max(x(:)),100);
% %fx1 = polyval(p,x1);
% fx2 = f2(x1);
% [fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

legend({'1 hidden','2 hidden','3 hidden'});%legend({'1 hidden pts','1 hidden fit','2 hidden pts','2 hidden fit','3 hidden pts','3 hidden fit'});
xlim([20 90]);
if strcmp(val_or_train,'val')
    ylabel(['Test error']);
else
    ylabel(['Training error']);
end
xlabel('#units');


%% plot num params vs best train
%close all; figure; hold all;
subplot(1,2,2);hold all;
x = [];y = [];s = [];c = [];
current_info = all_info_1;
for i = 1:numel(current_info)
    x(end+1) = current_info{i}.numParams;
    y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 40;
    c(end+1) = 0;
end
colr = 'k';

ux = []; uy = []; unique_x = unique(x);
for i = 1:numel(unique_x)
    idx = find( x == unique_x(i));
    ux(i) = unique_x(i);
    uy(i) = min(y(idx));
end
plot(ux,uy,'LineWidth',3,'Color',colr);%ylim([1.5 2.5]);

% scatter(x,y,s,'p','MarkerEdgeColor',colr);
% f1 = polyfit(x,y,1);
% f2 = fit(x',y','exp1');
% x1 = linspace(0,max(x(:)),100);
% %fx1 = polyval(p,x1);
% fx2 = f2(x1);
% [fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_2;
for i = 1:numel(current_info)
    % if abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2))> 0
    %     continue;
    % end
    x(end+1) = current_info{i}.numParams;
    y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 15.01;
    c(end+1) = 0.1;
end
colr = [0 0.5 1];

ux = []; uy = []; unique_x = unique(x);
for i = 1:numel(unique_x)
    idx = find( x == unique_x(i));
    ux(i) = unique_x(i);
    uy(i) = min(y(idx));
end
plot(ux,uy,'LineWidth',3,'Color',colr);%ylim([1.5 2.5]);

% scatter(x,y,s,'o','MarkerEdgeColor','b','MarkerFaceColor',colr);
% f1 = polyfit(x,y,1);
% f2 = fit(x',y','exp1');
% x1 = linspace(0,max(x(:)),100);
% %fx1 = polyval(p,x1);
% fx2 = f2(x1);
% [fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_3;
for i = 1:numel(current_info)
    % if (abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2)) > 0) || (abs(current_info{i}.hiddenLayers(2) ~= current_info{i}.hiddenLayers(3)) > 0)
    %     continue;
    % end
    x(end+1) = current_info{i}.numParams;
    y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 14.99;
    c(end+1) = 0.5;
end
colr = [0.2 0.5 0];

ux = []; uy = []; unique_x = unique(x);
for i = 1:numel(unique_x)
    idx = find( x == unique_x(i));
    ux(i) = unique_x(i);
    uy(i) = min(y(idx));
end
plot(ux,uy,'LineWidth',3,'Color',colr);%ylim([1.5 2.5]);

% scatter(x,y,s,'s','MarkerEdgeColor',[0.2 0.5 0]);
% f1 = polyfit(x,y,1);
% f2 = fit(x',y','exp1');
% x1 = linspace(0,max(x(:)),100);
% %fx1 = polyval(p,x1);
% fx2 = f2(x1);
% [fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

%legend({'1 hidden','2 hidden','3 hidden'});
legend({'1 hidden','2 hidden','3 hidden'});%legend({'1 hidden pts','1 hidden fit','2 hidden pts','2 hidden fit','3 hidden pts','3 hidden fit'});
xlim([0 900]);
% if strcmp(val_or_train,'val')
%     ylabel(['Minimum validation (sqrt) objective over 5 trials  ']);
% else
%     ylabel(['Minimum training (sqrt) objective over 5 trials  ']);
% end
xlabel('#parameters');

