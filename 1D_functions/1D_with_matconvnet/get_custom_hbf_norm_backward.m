function [ backward_function ] = get_custom_hbf_norm_backward( bwfun )
backward_function = @backward;
%res(i) = layer.backward(layer, res(i), res(i+1))
  function res = backward(layer, res, res_)
    W = layer.weights{1};
    S = layer.weights{2};
    Delta_tilde = res_.aux;
    [ dzdx, dzdw, dzds ] = bwfun(res.x,W,S,Delta_tilde, res_.dzdx) ;
    res.dzdx = dzdx;
    res.dzdw = {dzdw,dzds};
  end
end