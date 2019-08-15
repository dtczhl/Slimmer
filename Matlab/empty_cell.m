% empty cells

clear,clc

f_kitti = '../Result/EmptyCell/kitti_empty_cell.csv';
f_scannet = '../Result/EmptyCell/scannet_empty_cell.csv';

data_kitti = csvread(f_kitti, 1, 0);
data_kitti(:, 1) = data_kitti(:, 1) * 100; % cm

data_scannet = csvread(f_scannet, 1, 0);
data_scannet(:, 1) = data_scannet(:, 1) * 100;

figure(1),clf, hold on
set(gcf, 'Position', [200, 800, 600, 400])
plot(data_kitti(data_kitti(:, 1)<12, 1), data_kitti(data_kitti(:, 1)<12, 2), '-*', 'markersize', 8, 'linewidth', 3)
plot(data_scannet(data_scannet(:, 1)<12, 1), data_scannet(data_scannet(:, 1)<12, 2), '-s', 'markersize', 8, 'linewidth', 3)
set(gca, 'ygrid', 'on', 'fontsize', 22)
xticks([3:1:11])
%yticks([50:10:100])
legend('KITTI', 'ScanNet', 'location', 'southwest')
xlabel('Cell Size (cm)')
ylabel('Empty Cells (%)')
xlim([3, 11])
ylim([60, 105])
box on
hold off