# TimeEmb

此工作副本新增了 TimeEmb_FRCA，使用说明见 [README_FRCA.md](README_FRCA.md)。下方保留原 TimeEmb 说明。

## Getting Started

### 1、Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

```
conda create -n TimeEmb python=3.8
conda activate TimeEmb
pip install -r requirements.txt
```
### 2、Download Data
All the datasets needed for TimeEmb can be obtained from the [[Google Drive]](https://drive.google.com/drive/folders/1dfnzGafiaxo6BUsCMZbmlE0N6G5_yFqK?usp=sharing). 
Create a separate folder named ```./dataset``` and place all the CSV files in this directory. 
**Note**: Place the CSV files directly into this directory, such as "./dataset/ETTh1.csv"

### 3、Training Example
You can easily reproduce the results from the paper by running the provided script command. For instance, to reproduce the main results, execute the following command:

```
sh run_main.sh
```
