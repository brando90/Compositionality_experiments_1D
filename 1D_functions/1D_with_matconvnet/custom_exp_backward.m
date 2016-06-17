function [ backward_function ] = custom_exp_backward( bwfun )
backward_function = @backward;
%res(i) = layer.backward(layer, res(i), res(i+1))
  function res = backward(layer, res, res_)
    dzdx = bwfun(res_.x,res_.dzdx) ;
    res.dzdx = dzdx;
  end
end