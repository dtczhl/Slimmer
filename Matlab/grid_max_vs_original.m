% iou. Original vs Grid max

clear, clc

main_dir = '/home/dtc/MyGit/dtc-scannet-sparseconvnet/Result/pmserver/';
model_dir = {
    'scannet_m8_rep1_residualFalse-000000470';
    'scannet_m8_rep2_residualFalse-000000560';
    'scannet_m8_rep2_residualTrue-000000530';
    'scannet_m16_rep1_residualFalse-000000530';
    'scannet_m16_rep2_residualFalse-000000570';
    'scannet_m16_rep2_residualTrue-000000650'
};

result = zeros(length(model_dir), 3);

for i = 1:length(model_dir)
    f_random = strcat(main_dir, model_dir{i}, '/Backup/random_result_main.csv');
    data = csvread(f_random, 1, 0);
    original_iou = data(end, 3);
    f_grid = strcat(main_dir, model_dir{i}, '/Backup/grid_result_main.csv');
    data = csvread(f_grid, 1, 0);
    keep_ratio_grid = data(:, 1);
    [grid_max, grid_index] = max(data(:, 3));
    result(i, :) = [original_iou, keep_ratio_grid(grid_index), grid_max];
end

disp(result)