restoredefaultpath;clear;clc;clear;clc;
disp('making data set');
%% target function
f_target = struct('h', cell(1,3), 'f', cell(1,3));

h1 = @(A) cos(A);
h2 = @(A) 2*A^2 - 1;
h3 = @(A) 2*A^2 - 1;
f_target(1).h = h1;
f_target(2).h = h2;
f_target(3).h = h3;

f_target(1,1).f = @f_1D_hard_code;
%% make data set
sigpower = 'measured';
powertype = 'linear';
snr = 2;
low_x = -2*pi;
high_x = 2*pi;
nb_samples = 100000; %100,000
D = 1;
[X_train,Y_train] = generate_data_from_function( f_target, snr, low_x,high_x, nb_samples, sigpower, powertype, D);
[X_cv,Y_cv] = generate_data_from_function( f_target, snr, low_x,high_x, nb_samples, sigpower, powertype, D);
[X_test,Y_test] = generate_data_from_function( f_target, snr, low_x,high_x, nb_samples, sigpower, powertype, D);
D = size(X_train,2);
D_out = size(Y_train,2);
%%
%data_file_name = 'f8D_all_data_set_Id_interval';
%data_file_name = 'f1D_cos_NO_NOISE';
data_file_name = sprintf('f1D_cos_snr_%d', snr)
save(data_file_name)
beep;