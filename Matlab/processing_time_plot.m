% sampling overhead (time)

clear, clc

main_dir = '/home/dtc/Data/ScanNet/Accuracy/processing_time';
device = 'laptop';

f_random = strcat(main_dir, '/', device, '/random.time.txt');
f_grid = strcat(main_dir, '/', device, '/grid.time.txt');

data_random = dlmread(f_random);
data_grid = dlmread(f_grid);


figure(1), clf, hold on
set(gcf, 'position', [300, 600, 800, 500])
plot(data_random(:, 1), data_random(:, 2), '-*', 'linewidth', 3, 'markersize', 8)
plot(data_grid(:, 1), data_grid(:, 2), '-s', 'linewidth', 3, 'markersize', 8)
set(gca, 'fontsize', 22)
xlabel('Point Cloud Size (%)')
ylabel('Processing Time (ms)')
box on 
grid on
hold off