function [ nn ] = make_NN_model( L, mdl_params)
run('./activation_funcs');
nn = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
%% set activation funcs
for l=1:L
    if l==L
        switch F_func_name
        case 'F_NO_activation_final_layer'
            nn(l).Act = Identity;
            nn(l).dAct_ds = dIdentity_ds;
        case 'F_activation_final_layer'
            nn(l).Act = Act;
            nn(l).dAct_ds = dAct_ds;
            nn(l).beta = 0;
            nn(l).lambda = lambda;
        end
    else
        nn(l).Act = Act;
        nn(l).dAct_ds = dAct_ds;
        nn(l).beta = 0;
        nn(l).lambda = lambda;
    end
end
%% initialize
for l=1:L
    [D_l_1, D_l] = size(dim_layer(l).Dim);
    nn(l).W = normrnd(0, epsilon(l).eps, [D_l_1, D_l] );
    nn(l).b = normrnd(0, epsilon(l).eps, [1, D_l] );
end
end