function val = poly_eval_n3_d3_v03(x,dummy1)
%  val = poly_eval([1 2 3 4],randn(3,3,3,3));
%  val = poly_eval([1 2 3 4 5],randn(10,10,10,10,10));
% 
val = 0.32*x(1)^2*x(3) - 1.45*x(1)*x(2) + 2.25*x(2)^3 + -0.44*x(3)^2 + -1.22*x(3)*x(1)  + 2.88*x(3) + 1;

