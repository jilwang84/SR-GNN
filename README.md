__SR-GNN Model Implementation__

**Usage**

Data preprocessing:

`cd .\datasets\`

`python .\preprocess.py --dataset $DATASET_NAME$ --path .\$DATASET_NAME$\$RAW_DATA_FILE_NAME$ --partial_inforce True`

- `partial_inforce` is required for all the dataset
- `train_fraction` is needed for some of the dataset
- `item_renumber` is required for clef and rsc15, recommend for 30music
- `split` is recommended to replace `train_fraction` to aoivd too large and unknown test set. Recommend for xing and tmall.

Some configuration recommendations:
- 30music [threshold 2] (Need renumber): split: 1/8
- clef (Need renumber): split: 1/16
- rsc15 (Need renumber): split: 1/32
- tmall: split: 1/128
- xing : split: 1/64

Training:

`python .\main.py --dataset $DATASET_NAME$ --batch_size 512 --epoch 20 --train_fraction 4`

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

