# dp_multiparty_data_release
Official code of [Differentially Private Multi-Party Data Release for Linear Regression](https://openreview.net/forum?id=SAlemvIoql9) in UAI 2023 by Ruihan Wu, Xin Yang, Yuanshun Yao, Jiankai Sun, Tianyi Liu, Kilian Q Weinberger, Chong Wang

## 0. Data Preparation
Download the `data` folder from [here](https://drive.google.com/drive/folders/1JVXvoyCKd7RP1kJNFQ211OvhPhW5TILj?usp=drive_link) and the structure is
```
./data/
    UCI/
        bike_sharing/processed_data.ptl
        gpu/processed_data.ptl
        insurance/processed_data.ptl
        superconduct/processed_data.ptl
        year_prediction/processed_data.ptl
```

## 1. Run the Experiment with Synthetic Data
Run the following script for `seed=0, ..., 999`
```
python private_linear_regression_synthetic_multiparty.py --seed $seed --seed_num 1
```
To get the Figure 2, check the Jupyter notebook `Results-Synthetic.ipynb`.

## 2. Run the Experiment with Real-World Data
Run the following script for `dataset` in {`bike_sharing`, `gpu`, `insurance`, `superconduct`, `year_prediction`} and `eps` in {`0.1`, `0.3`, `1.0`}:
```
python private_linear_regression_realworld_multiparty.py --dataset $dataset --seed 0 --eps $eps
```
To Get the Table 1, check the Jupyter notebook `Results-RealWorld.ipynb`.
