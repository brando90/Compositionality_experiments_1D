function [ mdl ] = make_hbf( mdl_param)
L = size(mdl_param,2);
run('./activation_funcs');
%%
mdl = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
%activation funcs and F
Act = mdl_param(1).Act;
dAct_ds = mdl_param(1).dAct_ds;
for l=1:L
    if mod(l,2) == 1 % (l mod 2) = (a mod m)
        mdl(l).Act = Act;
        mdl(l).dAct_ds = dAct_ds;
    else % mod(l,2) == 0 even
        mdl(l).Act = Identity;
        mdl(l).dAct_ds = dIdentity_ds;
    end
end
%regularization
for l=1:L
    mdl(l).lambda = mdl_param(l).lambda;
    mdl(l).beta = mdl_param(l).beta;
end
mdl(1).F = @F;
%% initialize
switch mdl_param(1).init_method
case 't_zeros_plus_eps'
    for l=1:L
        [D_l_1, D_l] = size(mdl_param(l).W);
        if mod(l,2) == 1
            mdl(l).W = mdl_param(l).eps * randn([D_l_1, D_l] );
            mdl(l).b = mdl_param(l).eps * randn([1, D_l] );
            mdl(l).Wmask = 1;
            mdl(l).Stdmask = 1;
        else
            mdl(l).W = mdl_param(l).eps * randn([D_l_1, D_l] );
            mdl(l).b = 0;
            mdl(l).Wmask = 1;
            mdl(l).Stdmask = 0;
        end
    end
case 't_random_data_points'
    X_train = mdl_param(1).X_train;
    %D_1 = mdl_param(1).Dim(2);
    D_1 = size(mdl_param(1).W,2);
    % If DATA is a matrix, then Y is a matrix containing K rows selected from DATA
    mdl(1).W = datasample(X_train, D_1, 'Replace', false)'; % (D^(0) x D^(1)) = (D^(1) x D^(0))' 
    N = size(X_train,1);
    L = size(mdl,2);
    fp = struct('A', cell(1,L), 'Z', cell(1,L), 'Delta_tilde', cell(1,L));
    %% initliaze mdl.W layers 2 to L
    for l = 2:L
        %D_l = mdl_param(l).Dim(2);
        D_l = size(mdl_param(l).W,2);
        current_data_index = ceil(rand(D_l,1) * N); % (D^(l) x D^(0)) = (M x D^(0))
        X_l = X_train(current_data_index,:); % (1 x D)
        %% forward pass up to l - 1
        A = X_l; % ( M x D) = (M x D^(0))
        for l_p=1:l-1
            if mod(l_p,2) == 1
                WW = sum(mdl(l_p).W.^2, 1); % ( 1 x D^(l-1)= sum( (M x D^(l)), 1 )
                XX = sum(A.^2, 2); % (M x 1) = sum( (M x D^(l-1)), 2 )
                Delta_tilde = 2*(A*mdl(l_p).W) - bsxfun(@plus, WW, XX) ; % 2<x,w> - (|x|^2 + |w|^2) = - (|x|^2 + |w|^2) - 2<x,w>) = -|x-w|^2
                %Z = mdl(l).beta*( 2*(A*mdl(l).W) - bsxfun(@plus, WW, XX)); % (M x D^(l-1)) - (M x D^(l-1))
                Z = mdl(l_p).beta*( Delta_tilde ); % (M x D^(l-1)) - (M x D^(l-1))
                A = mdl(l_p).Act(Z); % (M x D^(l))
                fp(l_p).Delta_tilde = Delta_tilde;
                fp(l_p).Z = Z;
                fp(l_p).A = A; % (M x D^(l))
            else
                A = mdl(l_p).Act( A * mdl(l_p).W ); % (M x D^(l)) = (M x D^(l-1)+1) x (D^(l-1)+1 x D^(l))
                fp(l_p).A = A; % (M x D^(l))
            end
        end
        %% use X^(l-1) to init -> W^(l)
        mdl(l).W = fp(l-1).A'; % (D^(l-1) x D^(l)) = (D^(l) x D^(l-1))'     
    end
    for l=1:L
       if mod(l,2) == 1
            mdl(l).Wmask = 1;
            mdl(l).Stdmask = 1;
        else
            mdl(l).Wmask = 1;
            mdl(l).Stdmask = 0;
        end   
    end
otherwise
    error('Initi method not existent')
end

end