function [ forward_function ] = get_custom_hbf_norm_forward( fwfun )
forward_function = @forward;
%res(i+1) = layer.forward(layer, res(i), res(i+1))
  function res_ =  forward(layer, res, res_)
    % cutom_hbf_norm_forward( X,W,S )
    W = layer.weights{1};
    S = layer.weights{2};
    res_.x = fwfun(res.x, W,S ) ;
  end
end