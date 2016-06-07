function val = poly_eval_n3_d3_v02(x,dummy1)
%  val = poly_eval([1 2 3 4],randn(3,3,3,3));
%  val = poly_eval([1 2 3 4 5],randn(10,10,10,10,10));
% 
val = 1*x(1)^2*x(3) - 2*x(1)*x(2) + 2*x(2)^3 + 5*x(3)^2 + -3*x(3)*x(1)  + 3*x(3) + 1;

