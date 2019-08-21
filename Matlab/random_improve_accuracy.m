% grid random accuracy

clear, clc

iou_result = [];

models = {
    'scannet_m8_rep1_residualFalse-000000470';
    'scannet_m8_rep2_residualFalse-000000560';
    'scannet_m8_rep2_residualTrue-000000530';
    'scannet_m16_rep1_residualFalse-000000530';
    'scannet_m16_rep2_residualFalse-000000570';
    'scannet_m16_rep2_residualTrue-000000650';
};

for i = 1:length(models)
    main_dir = strcat('/home/dtc/MyGit/dtc-scannet-sparseconvnet/Result/pmserver/', models{i}, '/Backup/');
    f_grid = strcat(main_dir, 'random_result_main.csv');
    f_random = strcat(main_dir, 'random_result_main.csv');
    data_random = csvread(f_random, 1, 0);
    random_iou = data_random(end, 3);
    data_grid = csvread(f_grid, 1, 0);
    keep_ratio = data_grid(:, 1);
    iou = data_grid(:, 3);
    iou_result = [iou_result; [iou > random_iou]'];
end

iou_result = logical(iou_result);

figure(1), clf, hold on
set(gcf, 'position', [400, 600, 1100, 500])
for i = 1:length(models)
    keep_ratio_larger = keep_ratio(iou_result(i, :));
    keep_ratio_else = keep_ratio(~iou_result(i, :));
    h = plot(keep_ratio_larger, (length(models)-i)*ones(size(keep_ratio_larger)), 'o', 'markersize', 14);
    set(h, 'markerfacecolor', get(h, 'color'))
    plot(keep_ratio_else, (length(models)-i)*ones(size(keep_ratio_else)), 'x', 'markersize', 14)
end
h = zeros(2, 1);
h(1) = plot(NaN, NaN, 'kx', 'markersize', 14, 'markerfacecolor', 'k');
h(2) = plot(NaN, NaN, 'ko', 'markersize', 14, 'markerfacecolor', 'k');
legend(h, 'Worse than full point clouds', 'Better than full point clouds', 'orientation', 'horizontal')
yticks(0:1:5)
ylim([-0.6, 6.3])
yticklabels({'Model 6', 'Model 5', 'Model 4', 'Model 3', 'Model 2', 'Model 1'})
xlabel('Point Cloud Size (%)')
box on
set(gca, 'ygrid', 'on', 'fontsize', 22)
hold off