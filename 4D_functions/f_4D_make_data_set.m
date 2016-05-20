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
sigpower = 'measured';
powertype = 'linear';
snr = 8;
low_x = -2;
high_x = 2;
nb_samples = 10000;
generate_data_from_function( f_target, snr, low_x,high_x, nb_samples, sigpower, powertype)