function [ h_L ] = f_1D_hard_code( x, f_target )
L = size(f_target,2);
h_l_1 = x;
h = struct('val', cell(1,L));
for l=1:L
    h(l).val = f_target(l).h(h_l_1);
    h_l_1 = h(l).val;
end
h_L = h(L).val;
end