function [ f_val ] = f_8D_hard_code( x, f_target )
%compute left
h(1,1).val = f_target(1,1).h(x(1:2));
h(1,2).val = f_target(1,2).h(x(3:4));
h(2,1).val = f_target(2,1).h( [h(1,1).val, h(1,2).val] );
%compute right
h(1,3).val = f_target(1,3).h(x(5:6));
h(1,4).val = f_target(1,4).h(x(7:8));
h(2,2).val = f_target(2,2).h( [h(1,3).val, h(1,4).val] );
%compute all
h(3,1).val = f_target(3,1).h( [h(2,1).val, h(2,2).val] );
f_val = h(3,1).val;
end