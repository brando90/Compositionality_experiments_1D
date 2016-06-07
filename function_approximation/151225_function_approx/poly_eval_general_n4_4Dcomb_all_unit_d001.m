function A = poly_eval_general_n4_4Dcomb_all_unit_d001(x,coeffMat)

coeffMat = coeffMat*0 + 0.001; % all unit value

nVar = numel(x);
if nVar >= 2
    assert(nVar == ndims(coeffMat));
    nOrder = size(coeffMat,1) - 1;
    for i = 2:nVar
        assert(nOrder == size(coeffMat,i) - 1 );
    end
elseif nVar == 1
    assert(ndims(coeffMat)  == 2);
    assert(size(coeffMat,2) == 1); 
    nOrder = size(coeffMat,1) - 1;
else
    error('not supported nVar');
end

for i = 1:nVar
    varsPow{i} = realpow(repmat(x(i),[nOrder+1 1]),(0:nOrder)');
end

A=allcomb(varsPow{:});
A=prod(A,2);
A=A'*coeffMat(:);
A=sum(A);








