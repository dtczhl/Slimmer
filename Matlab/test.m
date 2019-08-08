clear, clc


data = dlmread('/home/dtc/MyGit/dtc-sparseconvnet/log/labels.txt');

n_correct = sum(data(:, 1) == data(:, 2));

n_tot = size(data, 1);

disp(n_correct/n_tot)

