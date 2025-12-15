# AQTP

## :sunny: Structure of AQTP
<img src="figs/1.png">



## Install the environment

```
conda create -n aqatrack python=3.8
conda activate aqatrack
bash install.sh
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training
Download pre-trained [HiViT-Base weights](https://drive.google.com/file/d/1VZQz4buhlepZ5akTcEvrA3a_nxsQZ8eQ/view?usp=share_link) and put it under `$PROJECT_ROOT$/pretrained_models` (see [HiViT](https://github.com/zhangxiaosong18/hivit) for more details).

```
bash train.sh
```


## Test
```
python test_epoch.py
```

## Evaluation 
```
python tracking/analysis_results.py
```



## Acknowledgments
* Thanks for the [EVPTrack](https://github.com/GXNU-ZhongLab/EVPTrack) [PyTracking](https://github.com/visionml/pytracking) and [AQATrack](https://github.com/GXNU-ZhongLab/AQATrack) library, which helps us to quickly implement our ideas.


## Citation


```

```
