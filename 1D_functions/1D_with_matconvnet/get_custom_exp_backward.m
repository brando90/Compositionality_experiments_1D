function [ dzdx ] = get_custom_exp_backward( A, P )
dzdx = A .* P;
end