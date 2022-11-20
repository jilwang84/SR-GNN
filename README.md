### SR-GNN & TA-GNN Model Implementation

#### How to Compile, Execute and Run the Training

1. First a python environment is needed. It is recommend to use `conda` to create virtual environments.

  Required packages:
  
  ```
  python~=3.10.6
  torch~=1.12.1
  pyg
  pandas
  matplotlib
  tqdm
  ```

2. Extract the raw data to target folder, amd do data preprocessing:

  `cd .\datasets\`

  `python .\preprocess.py --dataset $DATASET_NAME$ --path .\$DATASET_NAME$\$RAW_DATA_FILE_NAME$ --partial_inforce True --item_threshold 5 --item_renumber True --split 1/8`

  - `partial_inforce` is required for all the dataset
  - `train_fraction` is needed for some of the dataset
  - `item_renumber` is required for clef and rsc15, recommend for 30music
  - `split` is recommended to replace `train_fraction` to aoivd too large and unknown test set. Recommend for xing and tmall. It composed of 2 parts: `a/b`, where `b` is the split number, and `a` is which slice to be chosen


3. To start the training, back to project folder and do:

  `python .\main.py --model SR-GNN --dataset $DATASET_NAME$ --batch_size 512 --epoch 10`

  Some configuration recommendations: 
  - 30music [threshold 2] (Need renumber): split: 1/8->(around 20000) batch_size: 256 
  - aotm [threshold 2] (Better renumber): split: 1/4->(around 20000) batch_size: 256 
  - clef (Need renumber): split: 1/128 or 1/256->(around 30000) batch_size: 64
  - diginetica: Original settings. batch_size: 512
  - nowplaying: split 1/4->(around 30000) batch_size: 256 
  - retailrocket: split: 1/4->(around 20000) batch_size: 256
  - rsc15 (Need renumber): split: 1/128 or 1/256->(around 20000) batch_size: 128
  - tmall (Better renumber): split: 1/128->(around 30000) batch_size: 512 
  - xing (Better renumber): split: 1/32->(around 15000) batch_size: 256 

  It seems that 5 to 10 epoches would achieve good results.

  The results would be saved to `log` and `result` folder.

#### Description of Each Source File
```
- datasets
  - preprocess.py: Do the data preprocess
- src
  - base
    - result.py: An abstract class for results
  - dataset.py: An data processing file that convert data to suitable format for training and testing
  - result_saver.py: The model and loss saving class
  - SR_GNN.py: Implementation of SR-GNN
  - TA_GNN.py: Implementation of TA-GNN
- main.py: the main entrance for the whole running
```

#### Running Hardware and Software Information

**Hardware**: RTX3080 Laptop and RTX3060 Laptop

**Software**: Windows 10/11, with `cuda` available

#### Citation

```
@inproceedings{wu2019session,
  title={Session-based recommendation with graph neural networks},
  author={Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={33},
  number={01},
  pages={346--353},
  year={2019}
}
@inproceedings{yu2020tagnn,
  title={TAGNN: target attentive graph neural networks for session-based recommendation},
  author={Yu, Feng and Zhu, Yanqiao and Liu, Qiang and Wu, Shu and Wang, Liang and Tan, Tieniu},
  booktitle={Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval},
  pages={1921--1924},
  year={2020}
}
```

