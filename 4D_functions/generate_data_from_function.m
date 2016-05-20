function [ X, Y ] = generate_data_from_function( f_target, snr, low_x,high_x, nb_samples, sigpower, powertype)
% sigpower = usually 'measured', powertype = usually 'linear'
X = low_x + (high_x - low_x) * rand(nb_samples,4);
Y = zeros(nb_samples,1); % (N x 1)
for n = 1:nb_samples
    xn = X(n,:);
    fx = f_target(1,1).f( xn, f_target );
    yn = awgn(fx,snr,sigpower, powertype);
    Y(n,:) = yn;
end
end