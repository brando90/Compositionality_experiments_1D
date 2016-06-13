function [ backward_function ] = get_custom_l2_loss_backward( bwfun )
backward_function = @backward;
  function res = backward(layer, res, res_)
    res.dzdx = bwfun(res.x, layer.class, res_.dzdx) ;
  end
end

