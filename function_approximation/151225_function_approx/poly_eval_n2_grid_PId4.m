function val = poly_eval_n2_grid_PId4(x,dummy1)
blockSize = pi/4;
r = round(x(1)/blockSize) +  round(x(2)/blockSize);
rd2 = r/2;
val = abs(rd2 - round(rd2))*4 - 1;
