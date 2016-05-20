function [ X, Y ] = generate_data_from_function( f_target, snr, low_x,high_x, nb_samples, sigpower, powertype)
% sigpower = usually 'measured', powertype = usually 'linear'
X = linspace(low_x,high_x,nb_samples); % (N x D)
Y = zeros(N,1); % (N x 1)
for n = 1:nb_samples
    xn = X(n,:);
    fx = f_target(1,1).f_4D( xn, f_target );
    yn = awgn(fx,snr,sigpower, powertype);
    Y(n,1) = yn;
end
end