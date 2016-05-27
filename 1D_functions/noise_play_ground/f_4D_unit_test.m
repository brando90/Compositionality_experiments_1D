%% target function
h11 = @(A) (1*A(1) + 2*A(2))^2;
h12 = @(A) (3*A(1) + 4*A(2))^3;
h21 = @(A) (5*A(1) + 6*A(2))^4;
f_target = struct('h', cell(2,2), 'f', cell(2,2));
f_target(1,1).h = h11;
f_target(1,2).h = h12;
f_target(2,1).h = h21;
f_target(1,1).f = @f_4D;
%% test unit
x = [1,-2,3,-4];
f_target_val = f_target(1,1).f(x, f_target)
f_hard_coded_val = ( 5*(1*x(1) + 2*x(2))^2 + 6*(3*x(3) + 4*x(4))^3 )^4