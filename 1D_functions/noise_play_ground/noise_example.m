% To cause awgn to measure the power of X and add noise to
% produce a linear SNR of 4, use:
close all;
X = linspace(-2,2,100);
Y = 5*X.^5 + 4*X.^4 + 3*X.^3 + 2*X.^2 + 1*X.^1;
snr = 8;
Y_awgn_linear_measured = awgn(Y,snr,'measured','linear');
Y_awgn_measured = awgn(Y,snr,'measured');
figure;
plot(X, [Y; Y_awgn_linear_measured; Y_awgn_measured ])
legend('Original Y', 'Y linear and measured', 'Y measured')