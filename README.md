
## Setup

Install Facebook SparseConvNet <https://github.com/facebookresearch/SparseConvNet>

```shell
  conda create -n sparseconvnet python=3.6
  conda install pytorch=1.1 torchvision cudatoolkit=10.0 -c pytorch
  conda install google-sparsehash -c bioconda
  conda install -c anaconda pillow

  cd SparseConvNet
  bash develop.sh
```

Configure this git

```shell
  pip install plyfile
  conda install scipy
```

1.  download ScanNet data. `DataProcessing/download_data.py`
```plain
  scannet_dir = path to scannet directory
```

2.  copy data to `train` and `val` folder. `DataProcessing/split_data.py`
```plain
  scannet_dir = path to scannet directory
  git_dir = path to this git
```

3.  `.ply` to `.pth`. `prepare_data.py`

4.  copy `.pth` to ScanNet. `DataProcessing/copy_val_pth.py`
```plain
  git_dir =
  scannet_dir =
```

## Training

Relevant files

1.  `unet.py`
```python
  m =
  residual_blocks =
  block_reps =
```

2.  `data.py`
```python
  batch_size =
```

## Data Simplification

1.  random simplification. `Sampling/random_crop_data.py`
```python
  scannet_dir =
  processing_time_only = [True|False]
  keep_ratio_arr = range(...)
```

2.  grid simplificaiton. `Sampling/grid_crop_data.py`
```python
  scannet_dir =
  processing_time_only = [True|False]
  cell_size_arr = np.linspace(0.01, 0.1, 100)
```

3.  hierarchy simplification. `Sampling/hierarchy_crop_data.py`
```python
  scannet_dir =
  processing_time_only = [True|False]
  cluster_size_arr = range(2, 30, 1)
  var_max = 0.33
```

## Validation

1.  `Validation/main_valid.py`
```
  scannet_dir =
  model_name =
  data_type =
  save_pixel_result =
  specify_id =
  use_cuda =

  m =
  residual_blocks =
  block_reps =
```
