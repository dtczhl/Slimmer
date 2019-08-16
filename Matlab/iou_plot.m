% iou plot

clear, clc

main_dir = '/home/dtc/MyGit/dtc-scannet-sparseconvnet/Result/pmserver/scannet_m8_rep1_residualTrue-000000650/Backup/';
f_random = strcat(main_dir, 'random_result_main.csv');
f_grid = strcat(main_dir, 'grid_result_main.csv');
f_hierarchy = strcat(main_dir, 'hierarchy_result_main.csv');

titleName = split(main_dir, '/');
titleName = titleName{8};

data_random = csvread(f_random, 1, 0);
keep_ratio_random = data_random(:, 1);
iou_random = data_random(:, 3);

data_grid = csvread(f_grid, 1, 0);
keep_ratio_grid = data_grid(:, 1);
iou_grid = data_grid(:, 3);

data_hierarchy = csvread(f_hierarchy, 1, 0);
keep_ratio_hierarchy = data_hierarchy(:, 1);
iou_hierarchy = data_hierarchy(:, 3);


figure(1), clf, hold on
set(gcf, 'position', [200, 800, 1200, 800])
plot(keep_ratio_random, iou_random, '-*', 'linewidth', 3)
plot(keep_ratio_grid, iou_grid, '-o', 'linewidth', 3)
plot(keep_ratio_hierarchy, iou_hierarchy, '-s', 'linewidth', 3)
set(gca, 'fontsize', 22)
yticks(0:5:75)
ylim([0, 71])
ylabel('iou(%)')
legend('random', 'grid', 'hierarchy', 'location', 'southeast')
title(titleName, 'Interpreter', 'none')
grid on