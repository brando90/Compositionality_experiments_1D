function val = poly_eval_n3_d3_v04(x,dummy1)
% cannot learn well
val = 0.32*x(1)^2*x(3) + 1.2*x(1)^2 + 0.12*x(1) - 1.45*x(1)*x(2)  + 3.25*x(2)^3 +  2.7*x(2)^3*x(3)^3 + -0.44*x(3)^2 + -1.22*x(3)*x(1)  + 2.88*x(3) + 1;

