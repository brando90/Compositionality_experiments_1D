function net = addCustom_hbf_norm_layer(net, fwfun, bwfun)
%addCustom_hbf_norm_layer Add a custom hbf_norm layer to a network
%   NET = ADDCUSTOMLOSSLAYER(NET, FWDFUN, BWDFUN) adds a custom hbf_norm
%   layer to the network NET using FWDFUN for forward pass and BWDFUN for
%   a backward pass.

layer.name = 'hbf_norm' ;
layer.type = 'custom' ;
layer.forward = @forward ;
layer.backward = @backward ;
[W, S] = init_hbf_weights(); %TODO inilization
layer.weights = {{W, S}};
layer.learningRate = [0.9 0.9];
layer.weightDecay = [0 0];

net.layers{end+1} = layer;

  function res_ =  forward(layer, res, res_)
    res_.x = fwfun(res.x, layer.class) ;
  end

  function res = backward(layer, res, res_)
    [ dzdx, dzdw, dzds ] = bwfun(res.x, layer.class, res_.dzdx) ;
    res.dzdx = dzdx;
    res.dzdw = dzdw;
    res.dzdx = dzds;
  end
end


