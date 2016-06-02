function [ hbf ] = make_hbf( hbf_params)
L = size(hbf_params,2);
run('./activation_funcs');
%%
hbf = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
%activation funcs and F
Act = hbf_params(1).Act;
dAct_ds = hbf_params(1).dAct_ds;
for l=1:L
    if mod(l,2) == 1 % (l mod 2) = (a mod m)
        hbf(l).Act = Act;
        hbf(l).dAct_ds = dAct_ds;
    else % mod(l,2) == 0 even
        hbf(l).Act = Identity;
        hbf(l).dAct_ds = dIdentity_ds;
    end
end
%regularization
for l=1:L
    hbf(l).lambda = hbf_params(l).lambda;
    hbf(l).beta = hbf_params(l).beta;
end
hbf(1).F = @F;
%% initialize
switch hbf_params(1).init_method
case 't_zeros_plus_eps'
    for l=1:L
        [D_l_1, D_l] = size(hbf_params(l).W);
        if mod(l,2) == 1
            hbf(l).W = hbf_params(l).eps * randn([D_l_1, D_l] );
            hbf(l).b = hbf_params(l).eps * randn([1, D_l] );
            hbf(l).Wmask = 1;
            hbf(l).Stdmask = 1;
        else
            hbf(l).W = hbf_params(l).eps * randn([D_l_1, D_l] );
            hbf(l).b = 0;
            hbf(l).Wmask = 1;
            hbf(l).Stdmask = 0;
        end
    end
case 't_random_data_points'
    X_train = mdl_param(1).X_train;
    D_1 = mdl_param(1).Dim(2);
    % If DATA is a matrix, then Y is a matrix containing K rows selected from DATA
    mdl(1).W = datasample(X_train, D_1, 'Replace', false)'; % (D^(0) x D^(1)) = (D^(1) x D^(0))' 
    N = size(X_train,1);
    L = size(mdl,2);
    fp = struct('A', cell(1,L), 'Z', cell(1,L), 'Delta_tilde', cell(1,L));
    %% initliaze mdl.W layers 2 to L
    for l = 2:L
        D_l = mdl_param(l).Dim(2);
        current_data_index = ceil(rand(D_l,1) * N); % (D^(l) x D^(0)) = (M x D^(0))
        X_l = X_train(current_data_index,:); % (1 x D)
        %% forward pass up to l - 1
        A = X_l; % ( M x D) = (M x D^(0))
        for l_p = 1:l-1
            WW = sum(mdl(l_p).W.^2, 1); % ( 1 x D^(l-1)= sum( (M x D^(l)), 1 )
            XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
            Delta_tilde = 2*(A*mdl(l_p).W) - bsxfun(@plus, WW, XX) ;
            Z = mdl(l_p).beta*( Delta_tilde ); % (M x D^(l-1)) - (M x D^(l-1))
            A = mdl(l_p).Act(Z); % (M x D^(l))
            fp(l_p).Delta_tilde = Delta_tilde;
            fp(l_p).Z = Z;
            fp(l_p).A = A; % (M x D^(l))
        end
        %% use X^(l-1) to init -> W^(l)
        mdl(l).W = fp(l-1).A'; % (D^(l-1) x D^(l)) = (D^(l) x D^(l-1))'
    end
otherwise
    error('Initi method not existent')
end

end