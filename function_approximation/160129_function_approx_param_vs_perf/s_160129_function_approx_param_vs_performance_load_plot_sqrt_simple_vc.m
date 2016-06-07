%% plot num units vs best train

%function_name  = 'sin(x+2y)+cos(3x+2y)';
%function_name  = 'sin(2x)'
%function_name  = 'MNIST';
function_name  = '16x^4-48x^2+12';
val_or_train = 'val';
close all; figure;
subplot(1,3,1);hold all;
x = [];y = [];s = [];c = [];
current_info = all_info_1;
for i = 1:numel(current_info)
    x(end+1) = sum(current_info{i}.hiddenLayers(:));
    y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 40;
    c(end+1) = 0;
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
scatter(x,y,s,'o','MarkerEdgeColor','b','MarkerFaceColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
%fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

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
scatter(x,y,s,'s','MarkerEdgeColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
%fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

legend({'1 hidden pts','1 hidden fit','2 hidden pts','2 hidden fit','3 hidden pts','3 hidden fit'});
if strcmp(val_or_train,'val')
    ylabel(['Minimum validation (sqrt) objective  of approx. ' function_name '']);
else
    ylabel(['Minimum training (sqrt) objective  of approx. ' function_name '']);
end
xlabel('Number of units');


%% plot num params vs best train
%close all; figure; hold all;
subplot(1,3,2);hold all;
x = [];y = [];s = [];c = [];
current_info = all_info_1;
for i = 1:numel(current_info)
    x(end+1) = current_info{i}.numParams;
    y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 40;
    c(end+1) = 0;
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
scatter(x,y,s,'o','MarkerEdgeColor','b','MarkerFaceColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
%fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

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
scatter(x,y,s,'s','MarkerEdgeColor',[0.2 0.5 0]);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
%fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

%legend({'1 hidden','2 hidden','3 hidden'});
legend({'1 hidden pts','1 hidden fit','2 hidden pts','2 hidden fit','3 hidden pts','3 hidden fit'});
if strcmp(val_or_train,'val')
    ylabel(['Minimum validation (sqrt) objective  of approx. ' function_name '']);
else
    ylabel(['Minimum training (sqrt) objective  of approx. ' function_name '']);
end
xlabel('Number of parameters');



%% plot num params vs best train
%close all; figure; hold all;
subplot(1,3,3);hold all;
x = [];y = [];s = [];c = [];
current_info = all_info_1;
for i = 1:numel(current_info)
    x(end+1) = current_info{i}.numParams*sum(current_info{i}.hiddenLayers(:));
    y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 40;
    c(end+1) = 0;
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
current_info = all_info_2;
for i = 1:numel(current_info)
    % if abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2))> 0
    %     continue;
    % end
    x(end+1) = current_info{i}.numParams*sum(current_info{i}.hiddenLayers(:));
    y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 15.01;
    c(end+1) = 0.1;
end
colr = [0 0.5 1];
scatter(x,y,s,'o','MarkerEdgeColor','b','MarkerFaceColor',colr);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
%fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

x = [];y = [];s = [];c = [];
current_info = all_info_3;
for i = 1:numel(current_info)
    % if (abs(current_info{i}.hiddenLayers(1) - current_info{i}.hiddenLayers(2)) > 0) || (abs(current_info{i}.hiddenLayers(2) ~= current_info{i}.hiddenLayers(3)) > 0)
    %     continue;
    % end
    x(end+1) = current_info{i}.numParams*sum(current_info{i}.hiddenLayers(:));
    y(end+1) = sqrt( nanmin([current_info{i}.(val_or_train).objective]) );
    s(end+1) = 14.99;
    c(end+1) = 0.5;
end
colr = [0.2 0.5 0];
scatter(x,y,s,'s','MarkerEdgeColor',[0.2 0.5 0]);
f1 = polyfit(x,y,1);
f2 = fit(x',y','exp1');
x1 = linspace(0,max(x(:)),100);
%fx1 = polyval(p,x1);
fx2 = f2(x1);
[fx2 x1] = nonparametric_fit_by_binning_overlap(x,y,0.4,0.01); plot(x1,fx2,'LineWidth',3,'Color',colr);

%legend({'1 hidden','2 hidden','3 hidden'});
legend({'1 hidden pts','1 hidden fit','2 hidden pts','2 hidden fit','3 hidden pts','3 hidden fit'});
if strcmp(val_or_train,'val')
    ylabel(['Minimum validation (sqrt) objective  of approx. ' function_name '']);
else
    ylabel(['Minimum training (sqrt) objective  of approx. ' function_name '']);
end
xlabel('#parameters * #units');



