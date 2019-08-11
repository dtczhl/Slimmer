% sampling overhead (time)

clear, clc

main_dir = '../Result/ProcessingTime';
device = 'pmserver';

f_random = strcat(main_dir, '/', device, '/random.time.txt');
f_grid = strcat(main_dir, '/', device, '/grid.time.txt');
f_hierarchy = strcat(main_dir, '/', device, '/hierarchy.time.txt');

data_random = dlmread(f_random);
data_grid = dlmread(f_grid);
data_hierarchy = dlmread(f_hierarchy);


figure(1), clf, hold on
set(gcf, 'position', [200, 800, 900, 500])
plot(data_random(:, 1), data_random(:, 2), '-*', 'linewidth', 3, 'markersize', 8)
plot(data_grid(:, 1), data_grid(:, 2), '-s', 'linewidth', 3, 'markersize', 8)
plot(data_hierarchy(:, 1), data_hierarchy(:, 2), '-o', 'linewidth', 3, 'markersize', 8)
set(gca, 'fontsize', 22)
xlabel('Point Cloud Size (%)')
ylabel('Time (ms)')
set(gca, 'yscale', 'log')
ylim([0, 4000])
legend('Random Simplification', 'Grid Simplification', 'Hierarchy Simplification')
box on 
grid on
hold off