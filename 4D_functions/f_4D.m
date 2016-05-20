function [ h_21_val ] = f_4D( x, f_target )
%
h11_val = f_target(1,1).h(x(1:2));
h_12_val = f_target(1,2).h(x(3:4));
h_21_val = f_target(2,1).h( [h11_val, h_12_val] );
end