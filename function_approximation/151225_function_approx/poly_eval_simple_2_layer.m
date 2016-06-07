function A = poly_eval_simple_2_layer(x,dummy1)
a1 = max([x(1)-x(2) , 0]);
a2 = max([x(3)-x(4) , 0]);

A = max([a1 - a2, 0]);
