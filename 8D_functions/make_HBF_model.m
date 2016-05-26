function [ mdl ] = make_HBF_model(L, mdl_param)
run('./activation_funcs');
mdl = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
%% set activation funcs
F_func_name = mdl_param(1).F;
mdl(1).F = @F;
for l=1:L
    if l==L
        switch F_func_name
        case 'F_NO_activation_final_layer'
            mdl(l).Act = Identity;
            mdl(l).dAct_ds = dIdentity_ds;
            mdl(l).beta = mdl_param(l).beta;
            mdl(l).lambda = mdl_param(l).lambda;
        case 'F_activation_final_layer'
            mdl(l).Act = mdl_param(1).Act;
            mdl(l).dAct_ds = mdl_param(1).dAct_ds;
            mdl(l).beta = mdl_param(l).beta;
            mdl(l).lambda = mdl_param(l).lambda;
        end
    else
        mdl(l).Act = mdl_param(l).Act;
        mdl(l).dAct_ds = mdl_param(l).dAct_ds;
        mdl(l).beta = mdl_param(l).beta;
        mdl(l).lambda = mdl_param(l).lambda;
    end
end
%% initialize
switch mdl_param(1).init_method
case 't_zeros_plus_eps'
    for l=1:L
        D_l_1 = mdl_param(l).Dim(1);
        D_l = mdl_param(l).Dim(2);
        mdl(l).W = mdl_param(l).eps * randn([D_l_1, D_l] );
    end
case 't_random_data_points'
    X_train = mdl_param(1).X_train;
    D_1 = size(mdl(1).W, 2);
    mdl(1).W = datasample(X_train, D_1, 'Replace', false)'; % (D^(1) x D^(0)) = 
    N = size(X_train,1);
    L = size(mdl,2);
    for l = 1:L
        D_l = size(mdl(l).W,2);
        for d_l = 1:D_l
            x_dl = X_train(rand(1,N),:); % (1 x D)
            %% forward pass up to l
            fp = struct('A', cell(1,L), 'Z', cell(1,L), 'Delta_tilde', cell(1,L));
            A = x_dl; % ( M x D) = (M x D^(0))
            for l_p = 1:l
                WW = sum(mdl(l_p).W.^2, 1); % ( 1 x D^(l-1)= sum( (M x D^(l)), 1 )
                XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
                Delta_tilde = 2*(A*mdl(l_p).W) - bsxfun(@plus, WW, XX) ;
                Z = mdl(l_p).beta*( Delta_tilde ); % (M x D^(l-1)) - (M x D^(l-1))
                A = mdl(l_p).Act(Z); % (M x D^(l))
                fp(l_p).Delta_tilde = Delta_tilde;
                fp(l_p).Z = Z;
                fp(l_p).A = A; % (M x D^(l))
            end
            %%
            mdl(l).W(:,d_l) = fp(l).A;
        end
    end
otherwise
    error('Initi method not existent')
end

end