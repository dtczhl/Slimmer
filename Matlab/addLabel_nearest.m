% add missing labels with nearest point
clear,clc

pth_dir_original = '/home/dtc/Backup/Data/ScanNet/Ply';


scene_id = 'scene0011_01_vh_clean_2.ply';

pth_file_original = fullfile(pth_dir_original, scene_id);

pc_original = pcread(pth_file_original)

figure(2)
pcshow(pc_original)