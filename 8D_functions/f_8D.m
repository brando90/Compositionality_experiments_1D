function [ h_31_val ] = f_8D( x, f_target )
%compute left
f_left = f_target(1:4);
f_left_val = f_left(1,1).f_4D(x(4:8)); % h_21_val
%compute right
f_right = f_target(4:8);
f_right_val = f_right(1,1).f_4D(x(4:8));
%compute all
h_31_val = f_target(3,1).h( [f_left_val, f_right_val] );
end