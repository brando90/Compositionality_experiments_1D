function [ forward_function ] = get_custom_l2_loss_forward( fwfun )
forward_function = @forward;
%res(i+1) = layer.forward(layer, res(i), res(i+1))
  function res_ =  forward(layer, res, res_)
    res_.x = fwfun(res.x, layer.class) ;
  end
end