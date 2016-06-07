function A = poly_eval_general_2(x,coeffMat)
%  val = poly_eval([1 2 3 4],randn(3,3,3,3));
%  val = poly_eval([1 2 3 4 5],randn(10,10,10,10,10));
% opts.extra_power = 1;
% [opts, varargin] = vl_argparse(opts, varargin) ;

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

% B=cartprod(varsPow{:});
% B=prod(B,2);
% B=B'*coeffMat(:);
% B=sum(B);

