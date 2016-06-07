function val = poly_eval_general(x,coeffMat)
%  val = poly_eval([1 2 3 4],randn(3,3,3,3));
%  val = poly_eval([1 2 3 4 5],randn(10,10,10,10,10));
% 
nVar = numel(x);
assert(nVar == ndims(coeffMat));
nOrder = size(coeffMat,1) - 1;
for i = 2:nVar
    assert(nOrder == size(coeffMat,i) - 1 );
end

for i = 1:nVar
    varsPow{i} = realpow(repmat(x(i),[nOrder+1 1]),(0:nOrder)');
    rep = (nOrder+1)*ones(1,nVar);
    rep(i) = 1;
    perm = 1:nVar;
    perm(i) = 1; perm(setdiff(1:nVar,i)) = 2:nVar;
    varsPow{i} = permute(varsPow{i},perm);
    varsPow{i} = repmat(varsPow{i},rep);
end

varsPow_prod = varsPow{1};
for i = 2:numel(varsPow)
    varsPow_prod = varsPow_prod.*varsPow{i};
end

varsPow_prod = varsPow_prod .* coeffMat;

val = sum(varsPow_prod(:),1);




