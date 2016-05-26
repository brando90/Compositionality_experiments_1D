function [ num_params ] = number_of_params_NN( nn )
L = size(nn,2);
num_params = 0;
for l=1:L-1
    [D_l_1, D_l] = size( nn(l).W );
    %[~, D_l] = nn(l).b = size(nn(l).b );
    num_params = num_params + (D_l_1*D_l) + D_l;
end
[D_l_1, D_l] = size(nn(L).W);
num_params = num_params + (D_l*D_l_1);
end