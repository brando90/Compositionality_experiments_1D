function val = poly_eval_n2_grid_PI(x,dummy1)
blockSize = pi;
r = round(x(1)/blockSize) +  round(x(2)/blockSize);
rd2 = r/2;
val = abs(rd2 - round(rd2))*4 - 1;


