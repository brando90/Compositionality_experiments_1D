function [ forward_function ] = get_custom_hbf_norm_forward( fwfun )
forward_function = @forward;
  function res_ =  forward(layer, res, res_)
    res_.x = fwfun(res.x, layer.class) ;
  end
end