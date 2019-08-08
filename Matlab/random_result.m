% show performance of random sampling
clear, clc

f_data = '/home/dtc/Data/ScanNet/Accuracy/scannet_m16_rep2_residualTrue-000000530/result_random.csv';



data = csvread(f_data, 1, 0);
keep_ratio = data(:, 1);
num_points = data(:, 2);
iou = data(:, 3);
inference_time = data(:, 4);
flop = data(:, 5);
memory = data(:, 6);


% average IOU
figure(1),clf
set(gcf, 'Position', [200, 800, 800, 400]), hold on
plot(keep_ratio, iou, 'k-*', 'linewidth', 3, 'markersize', 10)
xlabel('Point Cloud Size (%)')
ylabel('Average IOU (%)')
set(gca, 'fontsize', 22)
xticks(10:10:100)
yticks(0:10:75)
xlim([0, 100])
grid on
box on
hold off

% flops
figure(2),clf
set(gcf, 'Position', [200, 800, 800, 400]), hold on
plot(keep_ratio, flop, 'k-*', 'linewidth', 3, 'markersize', 10)
xlabel('Point Cloud Size (%)')
ylabel('FLOPs (M)')
set(gca, 'fontsize', 22)
xticks(10:10:100)
yticks(0:20:100)
xlim([0, 100])
grid on
box on
hold off

% memory
figure(3),clf
set(gcf, 'Position', [200, 800, 800, 400]), hold on
plot(keep_ratio, memory/1e3, 'k-*', 'linewidth', 3, 'markersize', 10)
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
plot(keep_ratio, inference_time, 'k-*', 'linewidth', 3, 'markersize', 10)
xlabel('Point Cloud Size (%)')
ylabel('Inference Time (s)')
set(gca, 'fontsize', 22)
xlim([0, 100])
grid on
box on
hold off

