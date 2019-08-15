% show performance of random sampling
clear, clc

f_main = '../Result/pmserver/scannet_m16_rep2_residualTrue-000000650/Backup/result_main.csv';
f_memory = '../Result/alienware/scannet_m16_rep2_residualTrue-000000650/Backup/result_memory.csv';


data_main = csvread(f_main, 1, 0);
keep_ratio_main = data_main(:, 1);
num_points_main = data_main(:, 2);
iou_main = data_main(:, 3);
inference_time_main = data_main(:, 4);
flop_main = data_main(:, 5);
memory_main = data_main(:, 6);

% data_memory = csvread(f_memory, 1, 0);
% keep_ratio_memory = data_memory(:, 1);
% inference_time_memory = data_memory(:, 2);
% flop_memory = data_memory(:, 3);
% memory_memory = data_memory(:, 4);

% average IOU
figure(1),clf
set(gcf, 'Position', [200, 800, 800, 400]), hold on
plot(keep_ratio_main, iou_main, 'k-*', 'linewidth', 3, 'markersize', 10)
xlabel('Point Cloud Size (%)')
ylabel('Average IOU (%)')
set(gca, 'fontsize', 22)
xticks(10:10:100)
yticks(0:10:70)
xlim([0, 100])
ylim([0, 75])
grid on
box on
hold off

% flops
figure(2),clf
set(gcf, 'Position', [200, 800, 800, 400]), hold on
plot(keep_ratio_main, flop_main*1e6, 'k-*', 'linewidth', 3, 'markersize', 10)
xlabel('Point Cloud Size (%)')
ylabel('FLOPs')
set(gca, 'fontsize', 22)
xticks(10:10:100)
yticks([0:0.5:3]*1e10)
xlim([0, 100])
grid on
box on
hold off

% memory
figure(3),clf
set(gcf, 'Position', [200, 800, 800, 400]), hold on
plot(keep_ratio_main, memory_main/1e3, 'k-*', 'linewidth', 3, 'markersize', 10)
xlabel('Point Cloud Size (%)')
ylabel('Memory (Gbytes)')
set(gca, 'fontsize', 22)
xticks(10:10:100)
xlim([0, 100])
grid on
box on
hold off

% inference time
figure(4),clf
set(gcf, 'Position', [200, 800, 800, 400]), hold on
plot(keep_ratio_main, inference_time_main, 'k-*', 'linewidth', 3, 'markersize', 10)
xlabel('Point Cloud Size (%)')
ylabel('Inference Time (s)')
set(gca, 'fontsize', 22)
xlim([0, 100])
grid on
box on
hold off

