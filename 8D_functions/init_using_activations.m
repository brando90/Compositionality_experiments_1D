function [ output_args ] = init_using_activations( X, mdl_param )
N = size(X,1);
L = size(mdl,2);
for l= 1:L
    D_l = size(mdl(l).W,2);
    for d_l = 1:D_l
        x_dl = X(rand(1,N),:); % 
        %% forward pass up to l
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
        params(l).W(:,d_l) = fp(l_p).A;
    end
end

end