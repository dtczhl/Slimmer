% Plot IOU KNN

clear, clc

git_dir = '/home/dtc/MyGit/dtc-scannet-sparseconvnet';

device = 'pmserver';

model_dir = 'scannet_m16_rep2_residualTrue-000000650';

data_dir = fullfile(git_dir, 'Result', device, model_dir, 'Backup/iou_knn');

f_random_k_1 = fullfile(data_dir, 'Random_iou_knn_1.csv');
data_random_k_1 = csvread(f_random_k_1, 1, 0);
keep_ratio_k_1 = data_random_k_1(:, 1);
iou_k_1 = data_random_k_1(:, 3);

f_random_k_2 = fullfile(data_dir, 'Random_iou_knn_2.csv');
data_random_k_2 = csvread(f_random_k_2, 1, 0);
keep_ratio_k_2 = data_random_k_2(:, 1);
iou_k_2 = data_random_k_2(:, 3);

f_random_k_3 = fullfile(data_dir, 'Random_iou_knn_3.csv');
data_random_k_3 = csvread(f_random_k_3, 1, 0);
keep_ratio_k_3 = data_random_k_3(:, 1);
iou_k_3 = data_random_k_3(:, 3);

f_random_k_4 = fullfile(data_dir, 'Random_iou_knn_4.csv');
data_random_k_4 = csvread(f_random_k_4, 1, 0);
keep_ratio_k_4 = data_random_k_4(:, 1);
iou_k_4 = data_random_k_4(:, 3);

f_random_k_5 = fullfile(data_dir, 'Random_iou_knn_5.csv');
data_random_k_5 = csvread(f_random_k_5, 1, 0);
keep_ratio_k_5 = data_random_k_5(:, 1);
iou_k_5 = data_random_k_5(:, 3);


% original 
main_dir = '/home/dtc/MyGit/dtc-scannet-sparseconvnet/Result/pmserver/scannet_m16_rep2_residualTrue-000000650/Backup/';
f_random = strcat(main_dir, 'random_result_main.csv');
data_random = csvread(f_random, 1, 0);
keep_ratio_random = data_random(:, 1);
iou_random = data_random(:, 3);

figure(1), clf, hold on
plot(keep_ratio_random, iou_random, 'linewidth', 2)
plot(keep_ratio_k_1, iou_k_1, 'linewidth', 2)
legend('Original', 'KNN=1')
set(gca, 'fontsize', 16)
xlabel('Keep Ratio (%)')
ylabel('IOU')
grid on
hold off

figure(2), clf, hold on
plot(keep_ratio_k_1, iou_k_2 - iou_k_1, 'linewidth', 2)
plot(keep_ratio_k_1, iou_k_3 - iou_k_1, 'linewidth', 2)
plot(keep_ratio_k_1, iou_k_4 - iou_k_1, 'linewidth', 2)
plot(keep_ratio_k_1, iou_k_5 - iou_k_1, 'linewidth', 2)
plot([0, 100], [0, 0], 'k-.', 'linewidth', 2)
legend('KNN 2 - KNN 1', 'KNN 3 - KNN 1', 'KNN 4 - KNN 1', 'KNN 5 - KNN 1', '0 baseline')
set(gca, 'fontsize', 16)
grid on
xlabel('Keep Ratio (%)')
ylabel('IOU Absoluate Difference')
hold off