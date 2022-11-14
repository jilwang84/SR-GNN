__SR-GNN Model Implementation__

**Usage**

Data preprocessing:

`cd .\datasets\`

`python .\preprocess.py --dataset $DATASET_NAME$ --path .\$DATASET_NAME$\$RAW_DATA_FILE_NAME$ --partial_inforce True --item_threshold 5 --item_renumber True --split 1/8`

- `partial_inforce` is required for all the dataset
- `train_fraction` is needed for some of the dataset
- `item_renumber` is required for clef and rsc15, recommend for 30music
- `split` is recommended to replace `train_fraction` to aoivd too large and unknown test set. Recommend for xing and tmall. It composed of 2 parts: `a/b`, where `b` is the split number, and `a` is which slice to be chosen

Some configuration recommendations: (batch_size means exact 1 valid per epoch)
- 30music [threshold 2] (Need renumber): split: 1/8->(around 20000) batch_size: 256 __Not good accuracy__(<20%)
- aotm [threshold 2] (Better renumber): split: 1/4->(around 20000) batch_size: 256 __Still low accuracy__(<1%)
- clef (Need renumber): split: 1/128 or 1/256->(around 30000) batch_size: 64
- diginetica: Original settings. batch_size: 512
- nowplaying: split 1/4->(around 30000) batch_size: 256 __Not good accuracy__(<10%)
- retailrocket: split: 1/4->(around 20000) batch_size: 256
- rsc15 (Need renumber): split: 1/128 or 1/256->(around 20000) batch_size: 128
- tmall (Better renumber): split: 1/128->(around 30000) batch_size: 512 __Not good accuracy__(<10%)
- xing (Better renumber): split: 1/32->(around 15000) batch_size: 256 __Not good accuracy__(<10%)

It seems that 5 to 10 epoches would achieve good results

Training:

`python .\main.py --dataset $DATASET_NAME$ --batch_size 512 --epoch 5`

Required packages:
```
torch
pyg
pandas
matplotlib
tqdm
```

**Citation**

```
@inproceedings{Wu:2019ke,
title = {{Session-based Recommendation with Graph Neural Networks}},
author = {Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
year = 2019,
booktitle = {Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence},
location = {Honolulu, HI, USA},
month = jul,
volume = 33,
number = 1,
series = {AAAI '19},
pages = {346--353},
url = {https://aaai.org/ojs/index.php/AAAI/article/view/3804},
doi = {10.1609/aaai.v33i01.3301346},
editor = {Pascal Van Hentenryck and Zhi-Hua Zhou},
}
```

