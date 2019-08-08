% empty cells

clear,clc

f = '/home/dtc/Data/ScanNet/Result/backup/empty.csv';
data = csvread(f, 1, 0);
data(:, 1) = data(:, 1) * 100; % cm

figure(1),clf
set(gcf, 'Position', [200, 800, 800, 400])
plot(data(:, 1), data(:, 2), 'k-*', 'markersize', 8, 'linewidth', 3)
set(gca, 'ygrid', 'on', 'fontsize', 22)
xticks([1:1:10])
yticks([50:10:100])
xlabel('Cell Length (cm)')
ylabel('Empty Cells (%)')
xlim([1, 10])
ylim([50, 105])
