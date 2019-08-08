% random, grid, hierarchy compare

clear, clc

f_random = '/home/dtc/Data/ScanNet/Accuracy/scannet_m16_rep2_residualTrue-000000530/result_random.csv';
f_grid = '/home/dtc/Data/ScanNet/Accuracy/scannet_m16_rep2_residualTrue-000000530/result_grid.csv';
f_hierarchy = '/home/dtc/Data/ScanNet/Accuracy/scannet_m16_rep2_residualTrue-000000530/result_hierarchy.csv';

data_random = csvread(f_random, 1, 0);
keep_ratio_random = data_random(:, 1);
number_points_random = data_random(:, 2);
iou_random = data_random(:, 3);
inference_time_random = data_random(:, 4);
flop_random = data_random(:, 5);
memory_random = data_random(:, 6);

data_grid = csvread(f_grid, 1, 0);
keep_ratio_grid = data_grid(:, 1);
number_points_grid = data_grid(:, 2);
iou_grid = data_grid(:, 3);
inference_time_grid = data_grid(:, 4);
flop_grid = data_grid(:, 5);
memory_grid = data_grid(:, 6);

data_hierarchy = csvread(f_hierarchy, 1, 0);
keep_ratio_hierarchy = data_hierarchy(:, 1);
number_points_hierarchy = data_hierarchy(:, 2);
iou_hierarchy = data_hierarchy(:, 3);
inference_time_hierarchy = data_hierarchy(:, 4);
flop_hierarchy = data_hierarchy(:, 5);
memory_hierarchy = data_hierarchy(:, 6);


iou_random_for_grid = interp1(keep_ratio_random, iou_random, keep_ratio_grid);
iou_random_for_hierarchy = interp1(keep_ratio_random, iou_random, keep_ratio_hierarchy);

iou_grid_improve = (iou_grid - iou_random_for_grid) ./ iou_random_for_grid * 100;
iou_hierarchy_improve = (iou_hierarchy - iou_random_for_hierarchy) ./ iou_random_for_hierarchy * 100;

figure(1), clf
set(gcf, 'Position', [200, 800, 900, 500]), hold on
plot(keep_ratio_grid, iou_grid_improve, '-*', 'linewidth', 3, 'markersize', 8)
plot(keep_ratio_hierarchy, iou_hierarchy_improve, '-o', 'linewidth', 3, 'markersize', 8)
set(gca, 'fontsize', 22)
legend('Grid Simplification', 'Hierarchy Simplification')
xlabel('Point Cloud Size (%)')
ylabel('IOU Improv. Ratio (%)')
xlim([7, 60])
box on
grid on
hold off
 

% iou_max = iou_random(end);
% iou_random_normalized = iou_random / iou_max;
% iou_grid_normalized = iou_grid / iou_max;
% iou_hierarchy_normalized = iou_hierarchy / iou_max;
% 
% 
% % IOU percentage 
% figure(1),clf
% set(gcf, 'Position', [200, 800, 900, 500]), hold on
% plot(keep_ratio_random(keep_ratio_random <= 60), iou_random_normalized(keep_ratio_random <= 60) * 100, '-*', 'linewidth', 3, 'markersize', 8)
% plot(keep_ratio_grid(keep_ratio_grid <= 60), iou_grid_normalized(keep_ratio_grid <= 60) * 100, '-s', 'linewidth', 3, 'markersize', 8)
% plot(keep_ratio_hierarchy(keep_ratio_hierarchy <= 60), iou_hierarchy_normalized(keep_ratio_hierarchy <= 60) * 100, '-o', 'linewidth', 3, 'markersize', 8)
% xlabel('Point cloud size (%)')
% ylabel('IOU vs original (%)')
% set(gca, 'fontsize', 22)
% legend('Random simplification', 'Grid simplification', 'Hierarchy simplification', 'Location', 'southeast')
% xticks(10:10:100)
% yticks(0:20:100)
% xlim([0, 60])
% ylim([0, 105])
% grid on
% box on
% 
% % absolute IOU
% figure(2),clf
% set(gcf, 'Position', [200, 800, 800, 500]), hold on
% plot(keep_ratio_random, iou_random, '-*', 'linewidth', 3, 'markersize', 8)
% plot(keep_ratio_grid, iou_grid, '-s', 'linewidth', 3, 'markersize', 8)
% plot(keep_ratio_hierarchy, iou_hierarchy, '-o', 'linewidth', 3, 'markersize', 8)
% xlabel('Point Keep Ratio (%)')
% ylabel('Average IOU (%)')
% set(gca, 'fontsize', 22)
% legend('Random simplification', 'Grid simplification', 'Hierarchy simplification', 'Location', 'southeast')
% xticks(10:10:100)
% yticks(0:10:75)
% xlim([0, 100])
% grid on
% box on
% hold off
% 
