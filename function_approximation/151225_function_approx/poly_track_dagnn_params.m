function [values epoch legs] = poly_track_dagnn_params(root,params_indices)
%  [values epoch legs] = poly_track_dagnn_params(root,{ 1 [1]; 2 [1]; 3 [1]; 4 [1]; 5 [1]; 6 [1] } );
%  [values epoch legs] = poly_track_dagnn_params(root2,{ 1 [1]; 2 [1]; 3 [1]; 4 [1]; } );
%  [values epoch legs] = poly_track_dagnn_params(root2,{ 1 [1 1 1 1]; 1 [1 1 1 2]; 2 [1 1];  3 [1 1 1 1];  3 [1 1 2 1];  4 [1 1]; } );

[target_name epoch target] = meval_get_net_list_from_dir(root);
values = [];
legs = {};
for i = 1:numel(target_name)
    load(target_name{i});
    if ~isDagNN(net)
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ; 
    end
    for p = 1:size(params_indices,1)
        param_idx = params_indices{p,1};
        param_location = params_indices{p,2};
        theParam = net.params(param_idx);
        theParamValue = theParam.value;
        param_location_cell = num2cell(param_location);
        values(p,i) = theParamValue( sub2ind(size(theParamValue), param_location_cell{:} ) );   
        if numel(legs) < p || isempty(legs{p})
            legs{p} = [theParam.name ' : '  num_array_to_string_separated_by_underscore(param_location)];
        end
    end
end
