% sampling overhead (time)


keep_ratio = [0, 100];

random_time = linspace(10, 1, length(keep_ratio));
grid_time = linspace(20, 2, length(keep_ratio));
hierarchy_time = linspace(30, 3, length(keep_ratio));

figure(1), clf, hold on
plot(keep_ratio, random_time)
plot(keep_ratio, grid_time)
plot(keep_ratio, hierarchy_time)
xlabel('Point Cloud Size (%)')
ylabel('Processing Time (ms)')
grid on 
box on
legend('Random Simplification', 'Grid Simplification', 'Hierarchy Simplification')
hold off