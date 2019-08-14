
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
  pip install plyfile
  conda install scipy psutil
  ```

3.  Add this git to `PYTHONPATH`

  1.  in conda environment
    ```shell
    cd $CONDA_PREFIX
    mkdir etc/conda/activate.d
    mkdir etc/conda/deactivate.d
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
  Files are saved to `scannet_dir/Pth/Original`

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

1.  random simplification. `Sampling/random_crop_data.py`
  ```python
  scannet_dir =
  device = a_tag_for_your_device
  ```
  Simplified data are saved to `scannet_dir/Pth/Random/`. Folder structure: `{ratio of point cloud size}`  
  Data processing time are saved to `../Result/{device}/Random/`. Filename structure: `time.txt.{ratio of point cloud size}`

2.  grid simplificaiton. `Sampling/grid_crop_data.py`
  ```python
  scannet_dir =
  device =
  ```
  Simplified data are saved to `scannet_dir/Pth/Grid/`. Folder structure: `{ratio of point cloud size}`  
  Data processing time are saved to `../Result/{device}/Grid/`. Filename structure: `time.txt.{ratio of point cloud size}`

3.  hierarchy simplification. `Sampling/hierarchy_crop_data.py`
  ```python
  scannet_dir =
  device =
  ```
  Simplified data are saved to `scannet_dir/Pth/Hierarchy/`. Folder structure: `{ratio of point cloud size}`  
  Data processing time are saved to `../Result/{device}/Hierarchy/`. Filename structure: `time.txt.{ratio of point cloud size}`


## Validation

1.  All performance metrics except memory. `Validation/main_valid.py`
  ```python
  scannet_dir =
  device =
  model_name = trained_model_in_../Model/
  data_type = [Random|Grid|Hierarchy]
  ```
  Data are saved to `../Result/{device}/{model_name}/{data_type}/result_main.csv`. File structure
  ```plain
  ratio of point cloud, average number of points per point cloud, IOU, running time per point cloud, flop per point cloud, memory (do not use this value)
  ```

2.  Memory only. Running on CPU. `Validation/valid_memory.py`
  ```python
  scannet_dir =
  device =
  model_name =
  data_type =
  ```
  Data are saved to `../Result/{device}/{model_name}/{data_type}/result_memory.csv`. File structure
  ```plain
  ratio of point cloud, running time per point cloud, flop per point cloud, memory per point cloud
  ```

## Auxiliary

1.  validate one point cloud. 'Validation/valid_one_point_cloud.py'
  ```python
  scannet_dir =
  model_name =
  data_type =
  keep_ratio =
  pth_filename =
  ```
  Data saved to `../tmp/{pth_filename}.{data_type}.{keep_ratio}`

2.  show predication result for a point cloud. `Validation/valid_view_point_cloud.py`
  ```python
  pth_file = path_to_pth_file
  show_gt = [True|False] # show groundtruth?
  ```

## Folder Structure
*   `Cpp`. C++ program for data simplification
*   `DataProcessing`. Data pre-processing.
*   `Image`.
*   `log`. Checkpoints during training
*   `Matlab`.
*   `Model`. Trained models
*   `PointFeature`. Exploring point cloud features
*   `Result`. Saved results
*   `Sampling`. Data simplification
*   `tmp`. As workspace
*   `train`. Training data. (ignore it)
*   `val`. Validation data. (ignore it)
*   `Validation`. Results processing
