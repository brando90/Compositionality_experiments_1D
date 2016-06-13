function [ forward_function ] = get_custom_l2_loss_forward( fwfun )
forward_function = @forward;
  function res_ =  forward(layer, res, res_)
    res_.x = fwfun(res.x, layer.class) ;
  end
end