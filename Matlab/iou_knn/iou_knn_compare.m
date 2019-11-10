% random, grid, hierarchy compare

clear, clc

git_dir = '/home/dtc/MyGit/dtc-scannet-sparseconvnet';

device = 'pmserver';

model_dir = 'scannet_m16_rep2_residualTrue-000000650';

data_dir = fullfile(git_dir, 'Result', device, model_dir, 'Backup/iou_knn');

f_random_k_1 = fullfile(data_dir, 'Random_iou_knn_1.csv');
data_random_k_1 = csvread(f_random_k_1, 1, 0);
random_keep_ratio_k_1 = data_random_k_1(:, 1);
random_iou_k_1 = data_random_k_1(:, 3);

f_grid_k_1 = fullfile(data_dir, 'Grid_iou_knn_1.csv');
data_grid_k_1 = csvread(f_grid_k_1, 1, 0);
grid_keep_ratio_k_1 = data_grid_k_1(:, 1);
grid_iou_k_1 = data_grid_k_1(:, 3);

f_hierarchy_k_1 = fullfile(data_dir, 'Hierarchy_iou_knn_1.csv');
data_hierarchy_k_1 = csvread(f_hierarchy_k_1, 1, 0);
hierarchy_keep_ratio_k_1 = data_hierarchy_k_1(:, 1);
hierarchy_iou_k_1 = data_hierarchy_k_1(:, 3);

figure(1), hold on
plot(random_keep_ratio_k_1, random_iou_k_1, 'linewidth', 2)
plot(grid_keep_ratio_k_1, grid_iou_k_1, 'linewidth', 2)
plot(hierarchy_keep_ratio_k_1, hierarchy_iou_k_1, 'linewidth', 2)
grid on 
xlabel('Keep Ratio (%)')
ylabel('IOU')
legend('Random', 'Grid', 'Hierarchy')
set(gca, 'fontsize', 16)
hold off