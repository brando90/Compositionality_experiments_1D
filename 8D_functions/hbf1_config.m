%% hbf1 config
%% activation func
gauss_func = @(S) exp(S);
dGauss_ds = @(A) A;
%%
hbf1 = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
for l =1:L-1
    h_mdl(l).Act = Act;
    h_mdl(l).dAct_ds = dAct_ds;
    h_mdl(l).beta = gau_precision;
    h_mdl(l).lambda = lambda;
end
kernel_mdl(1).F = @F;
h_mdl(1).F = @F;
switch F_func_name
    case 'F_NO_activation_final_layer'
        h_mdl(L).Act = Identity;
        h_mdl(L).dAct_ds = dIdentity_ds;
        kernel_mdl(L).Act = Identity;
        kernel_mdl(L).dAct_ds = dIdentity_ds;
        kernel_mdl(l).beta = gau_precision;
        kernel_mdl(l).lambda = lambda;
    case 'F_activation_final_layer'
        h_mdl(L).Act = Act;
        h_mdl(L).dAct_ds = dAct_ds;
        kernel_mdl(L).Act = Identity;
        kernel_mdl(L).dAct_ds = dIdentity_ds;
        h_mdl(l).beta = gau_precision;
        h_mdl(l).lambda = lambda;     
end