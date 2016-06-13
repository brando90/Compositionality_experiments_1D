function [ backward_function ] = get_custom_hbf_norm_backward( bwfun )
backward_function = @backward;
%res(i) = layer.backward(layer, res(i), res(i+1))
  function res = backward(layer, res, res_)
    [ dzdx, dzdw, dzds ] = bwfun(res.x, layer.class, res_.dzdx) ;
    res.dzdx = dzdx;
    res.dzdw = dzdw;
    res.dzdx = dzds;
  end
end