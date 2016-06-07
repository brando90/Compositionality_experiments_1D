function y = poly_evalNet_10_10_10_10_noBN(x,opts)
nVar = opts.nVar;
nOut = opts.nOut;
hiddenLayers = [10 10 10 10];

random_seed = 1001;
%% set random seed %%%%%%%%%%
old_stream = RandStream.getGlobalStream();
s1 = RandStream.create('mrg32k3a','seed', random_seed );
try
    RandStream.setDefaultStream(s1);
catch
    RandStream.setGlobalStream(s1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


net = poly_init_resNet_2_no_first_bnorm('inputSize',[1 1 nVar],'hiddenLayers',hiddenLayers,'useBatchNorm',false,'enable_residual',false,'weightInitMethod','xavierimproved');
if ndims(x) ~=4
    assert(ndims(x) == 2);
    x = permute(x,[4 3 2 1]);
end
images = single( x);
label  = single( randn(1,1,nOut,size(x,4)));
inputs = {'input', images, 'label', label} ;
pred_idx =  net.getVarIndex('prediction');
net.vars(pred_idx).precious = 1;
%net = dagnet_to_single(net);
net.eval(inputs);

y = net.vars(pred_idx).value;
if nOut == 1
   y = permute(y,[4 3 2 1]);
else
   % 4D 
end

%%
RandStream.setGlobalStream(old_stream); % do not interfere with the old stream


