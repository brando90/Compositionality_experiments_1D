function [ backward_function ] = get_custom_l2_loss_backward( bwfun )
backward_function = @backward;
%res(i) = layer.backward(layer, res(i), res(i+1))
  function res = backward(layer, res, res_)
    res.dzdx = bwfun(res.x, layer.class, res_.dzdx) ;
  end
end

