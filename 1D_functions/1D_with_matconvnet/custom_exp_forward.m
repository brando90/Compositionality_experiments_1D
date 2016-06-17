function [ forward_function ] = custom_exp_forward( fwfun )
forward_function = @forward;
%res(i+1) = layer.forward(layer, res(i), res(i+1))
  function res_ =  forward(layer, res, res_)
    A = fwfun(res.x) ;
    res_.x = A;
  end
end