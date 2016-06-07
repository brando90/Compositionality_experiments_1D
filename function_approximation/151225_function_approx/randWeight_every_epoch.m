function [net, stats, opts] = randWeight_every_epoch(net, stats, opts, epoch)
if ~isfield(stats,'params_best')
    stats.params_best = net.params;
    stats.params_best_stats_train = stats.train(epoch);
    stats.params_mutated_epoch = epoch;
else
    if epoch <= stats.params_mutated_epoch + 10
        disp('Skip Mutation. Grace period after mutation................');
        % skip one epoch, because the first epoch after the mutation has high training error
        return;
    else
        if stats.params_best_stats_train.objective <=  stats.train(epoch).objective
            disp('Mutation!!!!!!!!!!!!!!!!!!!!!!!!!!');
            net.params = stats.params_best;
            rand_percent = 0.01;
            for i = 1:numel(net.params)
                num_w = numel(net.params(i).value);
                rand_idx_1 = sort(random_elements(1:num_w,ceil(num_w*rand_percent)));
                rand_idx_2 = sort(random_elements(1:num_w,ceil(num_w*rand_percent)));
                tmp = net.params(i).value(rand_idx_1);
                net.params(i).value(rand_idx_1) = net.params(i).value(rand_idx_2);
                net.params(i).value(rand_idx_2) = tmp;
            end
            stats.params_mutated_epoch = epoch;
        else
            stats.params_best = net.params;
            stats.params_best_stats_train = stats.train(epoch);
            stats.params_mutated_epoch = epoch;
        end
    end
end



