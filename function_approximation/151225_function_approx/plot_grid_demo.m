

domain =  [-2*pi 2*pi];
numTrain = 10000;
x = (domain(2) - domain(1))*rand(numTrain,2) + domain(1);
for i = 1:size(x,1)
    z(i) =  poly_eval_n2_grid_PI(x(i,:));
end

surf_from_scatter_auto(x(:,1),x(:,2),z);
set(gca,'CameraPosition',[0 0 7])
camorbit(45,0);

