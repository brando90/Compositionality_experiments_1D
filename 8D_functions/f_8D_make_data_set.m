restoredefaultpath;clear;clc;clear;clc;
%% target function
f_target = struct('h', cell(2,2), 'f', cell(2,2));

h11 = @(A) (1/20)*(1*A(1) + 2*A(2))^4; % ( x1 + x2)
h12 = @(A) (1/10)*(3*A(1) + 4*A(2))^3;
h21 = @(A) (1/100)*(5*A(1) + 6*A(2))^2;
f_target(1,1).h = h11;
f_target(1,2).h = h12;
f_target(2,1).h = h21;

h13 = @(A) (1/20)*(1*A(1) + 2*A(2))^4; % ( x1 + x2)
h14 = @(A) (1/10)*(3*A(1) + 4*A(2))^3;
h22 = @(A) (1/100)*(5*A(1) + 6*A(2))^2;
f_target(1,3).h = h13;
f_target(1,4).h = h14;
f_target(2,2).h = h22;

h31 = @(A) (1/10)*( (1/100)*A(1) + (1/100)*A(2) )^5;
f_target(3,1).h = h31;

% f_target(1,1).f_4D = @f_4D;
% f_target(1,1).f_8D = @f_8D;
% f_target(1,1).f_8D_hard_code = @f_8D_hard_code;
f_target(1,1).f = @f_8D_hard_code;
%% make data set
sigpower = 'measured';
powertype = 'linear';
snr = 8;
low_x = -2;
high_x = 2;
nb_samples = 100000; %100,000
D = 8;
[X,Y] = generate_data_from_function( f_target, snr, low_x,high_x, nb_samples, sigpower, powertype, D);
D = size(X,2);
D_out = size(Y,2);
%%
save('f8D_all_data_set')
beep;