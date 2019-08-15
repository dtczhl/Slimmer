% random processing time
clear, clc

main_dir = '../Result/ProcessingTime';
device = 'alienware';
data_type = 'Hierarchy';

raw_dir = strcat(main_dir, '/', device, '/', data_type, '/');

filePattern = fullfile(raw_dir, 'time.txt.*');
files = dir(filePattern);

% 1-row: keep ratio
% 2-row: time in ms
result = zeros(2, length(files));

for i = 1:length(files)
    filename = files(i).name;
    file_path = strcat(raw_dir, filename);
    data = dlmread(file_path)/1e3;  
    temp_keep_ratio = str2double(erase(filename, 'time.txt.'));
    result(1, i) = temp_keep_ratio;
    result(2, i) = mean(data);
end

% ascending
[~, idx] = sort(result(1, :));
result = result(:, idx);

save_file = strcat(main_dir, '/', device, '/', strcat(data_type, '.time.txt'));
fid = fopen(save_file, 'w');
fprintf(fid, '%d %f\n', result);
fclose(fid);

disp(strcat('Saved file to:', save_file))