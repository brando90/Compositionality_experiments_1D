function imdb = setup_polynomial_imdb(varargin)
%  imdb = setup_polynomial_imdb('nOrder',10,'nVar',4,'numTrain',10000,'numTest',2000);
%datasetDir = abs_path(datasetDir);
opts.numTrain = 500;
opts.numTest = 200;
%opts.seed = 1 ;
opts.random_seed = 1000;
opts.nOrder = 10;
opts.nVar = 5;
opts.density = 1;
opts.poly_eval_func = @poly_eval_n3_d3_v01;
opts.domain = [-2*pi 2*pi];
opts.nOut   = 1;
opts = vl_argparse(opts, varargin);

%% set random seed %%%%%%%%%%
old_stream = RandStream.getGlobalStream();
s1 = RandStream.create('mrg32k3a','seed', opts.random_seed );
try
    RandStream.setDefaultStream(s1);
catch
    RandStream.setGlobalStream(s1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

poly_eval_func = opts.poly_eval_func;

%%%%
code = DataHash_sha_384(opts);
datafile = fullfile('~/tmp/setup_polynomial_imdb/',code,'imdb.mat');
if ~exist(datafile,'file')
    
    %    [x y opts] = poly_eval_func(opts);
    if strcmp( func2str(poly_eval_func) ,'poly_eval_general' ) ||  strcmp( func2str(poly_eval_func) ,'poly_eval_general_2' )  ||  ~isempty(regexp(func2str(poly_eval_func),'_eval_general_'))
        genDim = (opts.nOrder+1)*ones(1,opts.nVar);
        coeffMat = randn(genDim);
        mult = single(rand(size(coeffMat)) <= opts.density);
        coeffMat = coeffMat .* mult;
    else
        coeffMat = [];
    end
    x = (opts.domain(2) - opts.domain(1))*rand(opts.numTrain+opts.numTest,opts.nVar) + opts.domain(1);
    if isempty(regexp(func2str(poly_eval_func),'_evalNet_'))
        for i = 1:size(x,1)
            if mod(i,1000) == 0
                disp(['preparing points...  i: ' num2str(i)]);
            end
            y(i) = poly_eval_func(x(i,:),coeffMat); 
        end
    else
        disp(['preparing points...  batch: ']);
        y = poly_eval_func(x,opts);
    end
    
    opts.coeffMat = coeffMat;

    imdb.images = struct;
    if ndims(y) ~= 4
        imdb.images.labels = single(permute(y(:),[2 3 4 1]));
    else
        imdb.images.labels = single(y);
    end
    imdb.images.data   = single(permute(x,[3 4 2 1]));
    if  ~isempty(regexp(func2str(poly_eval_func),'_eval_spacial'))
        imdb.images.data = permute(imdb.images.data,[3 1 2 4]);
    end

    imdb.images.set    = [ones(1,opts.numTrain) 3*ones(1,opts.numTest)];

    imdb.meta.sets = {'train'  'val'  'test'};
    imdb.opts = opts;
    
    mkdir_if_not_exist_for_a_file(datafile);
    save(datafile,'imdb');
else
    load(datafile);
end


RandStream.setGlobalStream(old_stream); % do not interfere with the old stream
