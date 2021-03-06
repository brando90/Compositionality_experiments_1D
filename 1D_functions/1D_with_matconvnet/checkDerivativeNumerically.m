function [err, dx_numerical, dx] = checkDerivativeNumerically(f, x, dx, print_err)
%CHECKDERIVATIVENUMERICALLY  Check a layer's deriviative numerically
%   ERR = CHECKDERIVATIVENUMERICALLY(F, X, DX) takes the scalar function F,
%   its tensor input X and its derivative DX at X and compares DX to
%   a numerical approximation of the derivative returing their difference
%   ERR.

y = f(x) ;
dx_numerical = zeros(size(dx), 'single') ;
delta = 0.0001 ;
% size_dx_numerical_1 = size(dx_numerical)
% size_dx_1 = size(dx)
% size_x_1 = size(x)

for n = 1:size(x,4)
  for k = 1:size(x,3)
    for j = 1:size(x,2)
      for i = 1:size(x,1)
        xp = x ;
        xp(i,j,k,n) = xp(i,j,k,n) + delta ;
        yp = f(xp) ;
        dx_numerical(i,j,k,n) =  (yp - y) / delta ; %compute Numerical Derivative
      end
    end
  end
end
% size_dx_numerical = size(dx_numerical)
% size_dx = size(dx)
err = dx_numerical - dx ;

if print_err
    range = max(abs(dx(:))) * [-1 1] ;
    T = size(x,4) ;
    for t = 1:size(x,4)
      subplot(T,3,1+(t-1)*3) ; bar3(dx(:,:,1,t)) ; zlim(range) ;
      title(sprintf('dx(:,:,1,%d) (given)',t)) ;
      subplot(T,3,2+(t-1)*3) ; bar3(dx_numerical(:,:,1,t)) ; zlim(range) ;
      title(sprintf('dx(:,:,1,%d) (numerical)',t)) ;
      subplot(T,3,3+(t-1)*3) ; bar3(abs(err(:,:,1,t))) ; zlim(range) ;
      title('absolute difference') ;
    end
end
end