clc;clear;
%%
folderName = fullfile('..');
p = genpath(folderName);
addpath(p);
%% create f_target
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

h31 = @(A) (1/100)*(5*A(1) + 6*A(2))^2;
f_target(3,1).h = h31;

f_target(1,1).f_4D = @f_4D;
f_target(1,1).f_8D = @f_8D;
f_target(1,1).f_8D_hard_code = @f_8D_hard_code;
%% unit test
x = 1:8;
f_val_hard_coded = f_target(1,1).f_8D_hard_code( x, f_target )
f_val_comp = f_target(1,1).f_8D( x, f_target )