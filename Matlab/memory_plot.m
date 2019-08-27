% memory plot

clear, clc

model_name = 'scannet_m16_rep2_residualTrue-000000650';

% avg with memory from pmserver
f_mem = {
    'result_memory_1.csv';
    'result_memory_2.csv';
    'result_memory_3.csv';
    'result_memory_4.csv';
};

main_dir = strcat('../Result/alienware/', model_name, '/Backup/');
f_random = strcat(main_dir, 'random_result_memory.csv');
f_grid = strcat(main_dir, 'grid_result_memory.csv');
f_hierarchy = strcat(main_dir, 'hierarchy_result_memory.csv');

titleName = split(main_dir, '/');
titleName = titleName{4};

data_random = csvread(f_random, 1, 0);
keep_ratio_random = data_random(:, 1);
time_random = data_random(:, 2);
flop_random = data_random(:, 3);
memory_random = data_random(:, 4);
for i = 1:length(f_mem)
    f_mem_file_name = strcat('../Result/pmserver/', model_name, '/Backup/random_', f_mem{i});
    data_memory = csvread(f_mem_file_name, 1, 0);
    memory_random = memory_random + data_memory(:, 4);
end
memory_random = memory_random / (1+length(f_mem));


data_grid = csvread(f_grid, 1, 0);
keep_ratio_grid = data_grid(:, 1);
time_grid = data_grid(:, 2);
flop_grid = data_grid(:, 3);
memory_grid = data_grid(:, 4);
for i = 1:length(f_mem)
    f_mem_file_name = strcat('../Result/pmserver/', model_name, '/Backup/grid_', f_mem{i});
    data_memory = csvread(f_mem_file_name, 1, 0);
    memory_grid = memory_grid + data_memory(:, 4);
end
memory_grid = memory_grid / (1+length(f_mem));

data_hierarchy = csvread(f_hierarchy, 1, 0);
keep_ratio_hierarchy = data_hierarchy(:, 1);
time_hierarchy = data_hierarchy(:, 2);
flop_hierarchy = data_hierarchy(:, 3);
memory_hierarchy = data_hierarchy(:, 4);
for i = 1:length(f_mem)
    f_mem_file_name = strcat('../Result/pmserver/', model_name, '/Backup/hierarchy_', f_mem{i});
    data_memory = csvread(f_mem_file_name, 1, 0);
    memory_hierarchy = memory_hierarchy + data_memory(:, 4);
end
memory_hierarchy = memory_hierarchy / (1+length(f_mem));

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
% fitobject = fit(keep_ratio_random, memory_random, 'poly2');
% memory_fit = feval(fitobject, keep_ratio_random);
fprintf('Random Memory (MB): %.2f\n', memory_random(end))

plot(keep_ratio_random, memory_random, '-*', 'linewidth', 3)
plot(keep_ratio_grid, memory_grid, '-o', 'linewidth', 3)
plot(keep_ratio_hierarchy, memory_hierarchy, '-s', 'linewidth', 3)
% plot(keep_ratio_random, memory_fit, '-*', 'linewidth', 4)
set(gca, 'fontsize', 22)
ylabel('Memory')
legend('random', 'grid', 'hierarchy', 'location', 'southeast')
title(titleName, 'Interpreter', 'none')
grid on