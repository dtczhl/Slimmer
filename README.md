Running on Ubuntu 18.04.

## Setup

1.  Install Facebook SparseConvNet <https://github.com/facebookresearch/SparseConvNet>

    ```shell
    conda create -n sparseconvnet python=3.6
    conda install pytorch=1.1 torchvision cudatoolkit=10.0 -c pytorch
    conda install google-sparsehash -c bioconda
    conda install -c anaconda pillow

    cd SparseConvNet
    bash develop.sh
    ```

2.  Install extra packages

    ```shell
    pip install plyfile pandas
    conda install scipy psutil
    pip install pptk
    ```
    Fix `pptk` compatibility issue with Ubuntu 18.04
    ```shell
    cd $CONDA_PREFIX/lib/python3.x/site-packages/pptk/libs
    mv libz.so.1 libz.so.1.old
    sudo ln -s /lib/x86_64-linux-gnu/libz.so.1
    ```

3.  Add this git to `PYTHONPATH`

    1.  in conda environment

        ```shell
        cd $CONDA_PREFIX
        mkdir -p etc/conda/activate.d
        mkdir -p etc/conda/deactivate.d
        ```

    2.  add `PYTHONPATH` to this conda environment

        ```shell
        vi etc/conda/activate.d/env_vars.sh
        ```
        with content of
        ```shell
        #!/bin/sh
        export PYTHONPATH=path_to_this_git
        ```

4.  download ScanNet data. `DataProcessing/download_data.py`

    ```python
    scannet_dir = path to scannet directory
    ```

5.  copy data to `train` and `val` folder. `DataProcessing/split_data.py`

    ```python
    scannet_dir = path to scannet directory
    git_dir = path to this git
    ```

6.  convert `.ply` to `.pth`. `prepare_data.py`

7.  copy `.pth` to ScanNet. `DataProcessing/copy_val_pth.py`

    ```python
    git_dir =
    scannet_dir =
    ```
    Files are saved to `{scannet_dir}/Pth/Original`

## Training

1.  configure model structure. `unet.py`

    ```python
    m =
    residual_blocks =
    block_reps =
    ```

2.  change batch size if needed. `data.py`

    ```python
    batch_size =
    ```

3.  start training

    ```shell
    python unet.py
    ```

    Trained models are saved to `log` folder. Filename structure: `scannet_m{}_rep{}_residual{}-{epoch}.pth`

## Data Simplification

Data simplification is implemented in C++ under `Cpp/sample_data`. Require CGAL 4.11.

```shell
  sudo apt-get install libcgal-dev
```

1.  random simplification. `Sampling/random_crop_data.py`

    ```python
    scannet_dir =
    device = a_tag_for_your_device
    ```
    Simplified data are saved to `{scannet_dir}/Pth/Random/`. Folder structure: `{ratio of point cloud size}`  
    Data processing time are saved to `Result/{device}/Random/`. Filename structure: `time.txt.{ratio of point cloud size}`

2.  grid simplificaiton. `Sampling/grid_crop_data.py`

    ```python
    scannet_dir =
    device =
    ```
    Simplified data are saved to `{scannet_dir}/Pth/Grid/`. Folder structure: `{ratio of point cloud size}`  
    Data processing time are saved to `Result/{device}/Grid/`. Filename structure: `time.txt.{ratio of point cloud size}`

3.  hierarchy simplification. `Sampling/hierarchy_crop_data.py`

    ```python
    scannet_dir =
    device =
    ```
    Simplified data are saved to `{scannet_dir}/Pth/Hierarchy/`. Folder structure: `{ratio of point cloud size}`  
    Data processing time are saved to `Result/{device}/Hierarchy/`. Filename structure: `time.txt.{ratio of point cloud size}`


## Validation

1.  For IOU only. `Validation/main_valid.py`

    ```python
    scannet_dir =
    device =
    model_name = trained_model_in_../Model/
    data_type = [Random|Grid|Hierarchy]
    ```
    Data are saved to `Result/{device}/{model_name}/{data_type}/result_main.csv`. File structure
    ```plain
    ratio of point cloud, average number of points per point cloud, IOU, running time per point cloud, flop per point cloud, memory (do not use this value)
    ```

2.  Everything except IOU. Running on CPU. `Validation/memory_valid.py`

    ```python
    scannet_dir =
    device =
    model_name =
    data_type =
    specify_id =
    is_save_ply_label = save predication for each point
    ```
    Data are saved to `Result/{device}/{model_name}/{data_type}/result_memory.csv`. File structure
    ```plain
    ratio of point cloud, running time per point cloud, flop per point cloud, memory per point cloud
    ```
    If `is_save_ply_label = True`, then predications are saved to `{scannet_dir}/PlyLabel/{data_type}/{ratio of point cloud size}`

## Recover to Full-Size

1.  convert original pth to Ply. `DataProcessing/pth_to_ply.py`

    ```python
    scannet_dir =
    ```
    Data saved to `{scannet_dir}/Ply`

2.  save predication labels for simplified point cloud. `Validation/memory_valid.py` with `is_save_ply_label = True`.

    ```python
    scannet_dir =
    device =
    model_name =
    data_type =
    specify_id =
    is_save_ply_label =
    use_cuda =
    ```
    Data saved to `{scannet_dir/PlyLabel}`

3.  add missing label from nearest labels. `AddLabel/add_label_nearest.py`

    ```python
    data_type =
    specify_id =
    k_KNN = label from nearest k_KNN labels
    ```
    Data saved to `{scannet_dir}/AddMissingLabel/{data_type}/{k_KNN}/{keep_ratio}`. File structure `.txt`
    ```plain
    x (float), y (float), z (float), r (int), g (int), b (int), orig_label (int), pred_label (int)
    ```

4.  calculate IOU for adding labels. `AddLabel/iou_after_label.py`

    ```python
    scannet_dir =
    data_type =
    specify_id =
    k_KNN =

    device =
    model_name =
    ```
    Data saved to `Result/{device}/{model_name}/{data_type}/iou_knn_{k_KNN}.csv`. File structure
    ```plain
    ratio of point cloud, k_KNN, IOU(%)
    ```

## Predictor for Simplification

1.  copy pth of training data to ply. `DataProcessing/pth_to_ply_for_train.py`. Data saved to `{scannet_dir}/Train_ply`

2.  generate labels for training dataset. `AddLabel/recover_full.py`. Data saved to `{scannet_dir}/Train_ply_label`. Each point cloud is sparsified and results saved to csv

3.  calculate iou of each point cloud. `AddLabel/iou_after_label_each.py`. Data saved to `Result`.

### KNN Compare

1.  `recover_full.py`
2.  dtc-c-sparse/Matlab/k_nearest_neighbor.m

## Auxiliary

1.  validate one point cloud. `Validation/valid_one_point_cloud.py`

    ```python
    scannet_dir =
    model_name =
    data_type =
    keep_ratio =
    pth_filename =
    ```
    Data saved to `tmp/{pth_filename}.{data_type}.{keep_ratio}`

2.  show predication result for a point cloud. `Validation/valid_view_point_cloud.py`

    ```python
    pth_file = path_to_pth_file
    show_gt = [True|False] # show groundtruth?
    ```

3.  scannet ratio of empty cells. `PointFeature/scannet_empty_cell.py`

    Data are saved to `Result/EmptyCell/scannet_empty_cell.csv`

4.  KITTI ratio of empty cells. `PointFeature/kitti_empty_cell.csv`

    Data are saved to `Result/EmptyCell/kitti_empty_cell.csv`

5.  For C++ implementation. `DataProcessing/pth_to_bin.py`: convert pth data to binary.


## Trained Models

Pre-trained models are under `Model/`. Naming: `scannet_m{}_rep{}_residual{}_000000{epoch}.pth`.

Index | m | rep | residual | epoch | #parameters | batch size | FLOPs | Memory (GB) | Time (s) | IOU (%)
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
1 | 8 | 1 | False | 470 | 673836 | 8 | 2.59 x 10^{9} | 0.59 | 0.63 | 63.29
2 | 8 | 2 | False  | 560 | 1073788 | 8 | 4.15 x 10^{9} | 0.74 | 0.78 | 65.15
3 | 8 | 1 | True | 630 | 1085436 | 8 | - | - | - | 67.54
4 | 8 | 2 | True | 530 | 1885340 | 8 | 7.44 x 10^{9} | 1.06 | 1.13 | 68.52
5 | 16 | 1 | False | 530 | 2689860 | 8 | 1.03 x 10^{10} | 0.84 | 0.87 | 67.11
6 | 16 | 2 | False | 570 | 4288100 | 8 | 1.66 x 10^{10} | 1.12| 1.16 | 68.79
7 | 16 | 1 | True | 500 | 4334692 | 8 | - | - | - | 68.72
8 | 16 | 2 | True | 650 | 7531172 | 8 | 2.97 x 10^{10} | 1.85 | 1.81 | 69.79

## Point Cloud of Max point

scene0231_01_vh_clean_2.pth

438565

## Folder Structure
*   `AddLabel`. Adding labels for simplified point clouds
*   `Cpp`. C++ program for data simplification, label adding, etc
*   `DataProcessing`. Data pre-processing.
*   `log`. Checkpoints during training
*   `Matlab`.
*   `Model`. Trained models
*   `Image`. Experiment images
*   `PointFeature`. Exploring point cloud features
*   `Result`. Saved results
*   `Sampling`. Data simplification
*   `tmp`. As workspace
*   `train`. Training data. (ignore it)
*   `val`. Validation data. (ignore it)
*   `Validation`. Results processing
