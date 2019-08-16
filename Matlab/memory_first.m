% memory first

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


for i_model = 1:length(models)
    for i_simplify = 1:length(data_simplify)
        f_data = strcat(main_dir, iou_device, '/', models{i_model}, '/Backup/', data_simplify{i_simplify}, '_result_main.csv');
        data = csvread(f_data, 1, 0);
        iou = data(:, 3);
        f_data = strcat(main_dir, time_device, '/', models{i_model}, '/Backup/', data_simplify{i_simplify}, '_result_memory.csv');
        data = csvread(f_data, 1, 0);
        keep_ratio = data(:, 1);
        memory = data(:, 4);
        fitobject = fit(keep_ratio, memory, 'poly2');
        memory = feval(fitobject, keep_ratio);
    
        if length(iou) ~= length(memory)
            error('length does not match')
        end
        for j = 1:length(iou)
            if iou(j) >= iou_target  
                ret_arr = [models(i_model), data_simplify(i_simplify), keep_ratio(j), memory(j)];
                result = [result; ret_arr];
            end
        end
    end
end

memory_arr = cell2mat(result(:, 4));
[min_v, min_k] = min(memory_arr);
result = result(min_k, :);
fprintf('\t%s\n\t%s - %d - %.2f\n', result{:})