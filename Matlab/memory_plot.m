% memory

clear, clc

main_dir = '/home/dtc/MyGit/dtc-scannet-sparseconvnet/Result/alienware/scannet_m8_rep1_residualFalse-000000470/Backup/';
f_random = strcat(main_dir, 'random_result_memory.csv');
f_grid = strcat(main_dir, 'grid_result_memory.csv');
f_hierarchy = strcat(main_dir, 'hierarchy_result_memory.csv');

titleName = split(main_dir, '/');
titleName = titleName{8};

data_random = csvread(f_random, 1, 0);
keep_ratio_random = data_random(:, 1);
time_random = data_random(:, 2);
flop_random = data_random(:, 3);
memory_random = data_random(:, 4);

data_grid = csvread(f_grid, 1, 0);
keep_ratio_grid = data_grid(:, 1);
time_grid = data_grid(:, 2);
flop_grid = data_grid(:, 3);
memory_grid = data_grid(:, 4);

data_hierarchy = csvread(f_hierarchy, 1, 0);
keep_ratio_hierarchy = data_hierarchy(:, 1);
time_hierarchy = data_hierarchy(:, 2);
flop_hierarchy = data_hierarchy(:, 3);
memory_hierarchy = data_hierarchy(:, 4);


figure(1), clf, hold on
set(gcf, 'position', [200, 800, 1200, 800])

plot(keep_ratio_random, time_random, '-*', 'linewidth', 3)
plot(keep_ratio_grid, time_grid, '-o', 'linewidth', 3)
plot(keep_ratio_hierarchy, time_hierarchy, '-s', 'linewidth', 3)
% plot(keep_ratio_random, time_fit, '-*', 'linewidth', 4)
set(gca, 'fontsize', 22)
ylabel('Time')
legend('random', 'grid', 'hierarchy', 'location', 'southeast')
title(titleName, 'Interpreter', 'none')
grid on

figure(2), clf, hold on
set(gcf, 'position', [200, 800, 1200, 800])
fitobject = fit(keep_ratio_random, memory_random, 'poly2');
memory_fit = feval(fitobject, keep_ratio_random);
fprintf('Memory (MB): %.2f\n', memory_fit(end))

plot(keep_ratio_random, memory_random, '-*', 'linewidth', 3)
plot(keep_ratio_grid, memory_grid, '-o', 'linewidth', 3)
plot(keep_ratio_hierarchy, memory_hierarchy, '-s', 'linewidth', 3)
plot(keep_ratio_random, memory_fit, '-*', 'linewidth', 4)
set(gca, 'fontsize', 22)
ylabel('Memory')
legend('random', 'grid', 'hierarchy', 'Fitted Line', 'location', 'southeast')
title(titleName, 'Interpreter', 'none')
grid on