## Introduction

This repository is for our ECCV 2022 paper: **Generalizable Medical Image Segmentation via Random Amplitude Mixup and Domain Specific Image Restoration**.

![](./pictures/architecture.pdf)

## Data Preparation

### Download Fundus Dataset
[Fundus](https://github.com/emma-sjwang/Dofe)


## Training and Testing
The training and testing process can all be done on one Nvidia RTX 2080Ti GPU with 11 GB memory.
### Train on Fundus Dataset (Target Domain 0)
```
python -W ignore train.py --data_root ../dataset --dataset fundus --domain_idxs 1,2,3 --test_domain_idx 0 --ram --rec --is_out_domain --consistency --consistency_type kd --save_path ../outdir/fundus/target0 --gpu 0
```

### Test on Fundus Dataset (Target Domain 0)
```
python -W ignore test_fundus_slice.py --model_file ../outdir/fundus/target0/final_model.pth --dataset fundus --data_dir ../dataset --datasetTest 0 --test_prediction_save_path ../results/fundus/target0 --save_result --gpu 0
```

