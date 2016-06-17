function [ dzdx ] = custom_exp_backward( A, P )
dzdx = A .* P;
end