% speed first

clear, clc

iou_target = 70;


models = {
    'scannet_m8_rep1_residualFalse-000000470';
    'scannet_m8_rep2_residualFalse-000000560';
    'scannet_m8_rep2_residualTrue-000000530';
    'scannet_m16_rep1_residualFalse-000000530';
    'scannet_m16_rep2_residualFalse-000000570';
    'scannet_m16_rep2_residualTrue-000000650'
};

data_simplify = {
    'random';
    'grid';
    'hierarchy'
};


main_dir = '/home/dtc/MyGit/dtc-scannet-sparseconvnet/Result/';

iou_device = 'pmserver';
time_device = 'alienware';

result = [];

f_random_time = strcat(main_dir, 'ProcessingTime/alienware/Backup/Random.time.txt');
f_grid_time = strcat(main_dir, 'ProcessingTime/alienware/Backup/Grid.time.txt');
f_hierarchy_time = strcat(main_dir, 'ProcessingTime/alienware/Backup/Hierarchy.time.txt');
random_time = dlmread(f_random_time);
random_time(:, 2) = random_time(:, 2) / 1e3; % ms to s
grid_time = dlmread(f_grid_time);
grid_time(:, 2) = grid_time(:, 2) / 1e3;
hierarchy_time = dlmread(f_hierarchy_time);
hierarchy_time(:, 2) = hierarchy_time(:, 2) / 1e3;

for i_model = 1:length(models)
    for i_simplify = 1:length(data_simplify)
        f_data = strcat(main_dir, iou_device, '/', models{i_model}, '/Backup/', data_simplify{i_simplify}, '_result_main.csv');
        data = csvread(f_data, 1, 0);
        iou = data(:, 3);
        f_data = strcat(main_dir, time_device, '/', models{i_model}, '/Backup/', data_simplify{i_simplify}, '_result_memory.csv');
        data = csvread(f_data, 1, 0);
        keep_ratio = data(:, 1);
        time = data(:, 2);
    
        if length(iou) ~= length(time)
            error('length does not match')
        end
        for j = 1:length(iou)
            if iou(j) >= iou_target  
                ret_arr = [models(i_model), data_simplify(i_simplify), keep_ratio(j), time(j)];
                % add processing time
                if i_simplify == 1 
                    % random
                    k = find(random_time(:, 1) == keep_ratio(j), 1);
                    ret_arr(4) = num2cell(ret_arr{4} + random_time(k, 2));
                elseif i_simplify == 2
                    % grid
                    k = find(grid_time(:, 1) == keep_ratio(j), 1);
                    ret_arr(4) = num2cell(ret_arr{4} + grid_time(k, 2));
                elseif i_simplify == 3
                    % hierarchy
                    k = find(hierarchy_time(:, 1) == keep_ratio(j), 1);
                    ret_arr(4) = num2cell(ret_arr{4} + hierarchy_time(k, 2));
                else 
                    error ('i_simplify')
                end
                result = [result; ret_arr];
            end
        end
    end
end

time_arr = cell2mat(result(:, 4));
[min_v, min_k] = min(time_arr);
result = result(min_k, :);
fprintf('\t%s\n\t%s - %d - %.2f\n', result{:})