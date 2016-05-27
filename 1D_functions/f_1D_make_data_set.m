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
low_x = -1;
high_x = 1;
nb_samples = 100000; %100,000
D = 1;
[X,Y] = generate_data_from_function( f_target, snr, low_x,high_x, nb_samples, sigpower, powertype, D);
D = size(X,2);
D_out = size(Y,2);
%%
save('f8D_all_data_set_Id_interval')
beep;