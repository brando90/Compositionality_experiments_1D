%% target function
h11 = @(A) (1*A(1) + 2*A(2))^2;
h12 = @(A) (3*A(1) + 4*A(2))^3;
h21 = @(A) (5*A(1) + 6*A(2))^4;
f_target = struct('h', cell(2,2), 'f', cell(2,2));
f_target(1,1).h = h11;
f_target(1,2).h = h12;
f_target(2,1).h = h21;
f_target(1,1).f = @f_4D;
%%
D = 4;
x_max = 2;
x_min = -2;
X = x_min + (x_max - x_min).*rand(N,D);
eps = 0.1;
epsilon = randn(N,1);
for n = 1:N
    xn = X(n,:);
    f_xn = f_target(1,1).f_4D( x, f_target );
end
