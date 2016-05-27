function [ nn ] = make_NN_model( L, mdl_param)
run('./activation_funcs');
nn = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
%% set activation funcs
F_func_name = mdl_param(1).F;
nn(1).F = @F;
for l=1:L
    if l==L
        switch F_func_name
        case 'F_NO_activation_final_layer'
            nn(l).Act = Identity;
            nn(l).dAct_ds = dIdentity_ds;
            nn(l).beta = 0;
            nn(l).lambda = mdl_param(l).lambda;
        case 'F_activation_final_layer'
            nn(l).Act = mdl_params(1).Act;
            nn(l).dAct_ds = mdl_params(1).dAct_ds;
            nn(l).beta = 0;
            nn(l).lambda = mdl_param(l).lambda;
        end
    else
        nn(l).Act = mdl_param(l).Act;
        nn(l).dAct_ds = mdl_param(l).dAct_ds;
        nn(l).beta = 0;
        nn(l).lambda = mdl_param(l).lambda;
    end
end
%% initialize
for l=1:L
    D_l_1 = mdl_param(l).Dim(1);
    D_l = mdl_param(l).Dim(2);
    nn(l).W = mdl_param(l).eps * randn([D_l_1, D_l] );
    nn(l).b = mdl_param(l).eps * randn([1, D_l] );
end
end