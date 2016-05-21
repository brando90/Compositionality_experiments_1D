function [ h_31_val ] = f_8D( x, f_target )
%compute left
f_left = f_target(1,1:2);
f_left(2,1).h = f_target(2,1).h;
f_left_val = f_left(1,1).f_4D(x(1:4), f_left); % h_21_val
%compute right
f_right = f_target(1,3:4);
f_right(2,1).h = f_target(2,2).h;
f_right(1,1).f_4D = @f_4D;
f_right_val = f_right(1,1).f_4D(x(5:8), f_right);
%compute all
h_31_val = f_target(3,1).h( [f_left_val, f_right_val] );
end