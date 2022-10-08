# Informer-AFSD
This project includes informer,MLP,RNN backbones with an AFSD detection part

## Step1
Extract this project in a folder

## Step2
Download the Wi-Fi dataset and the pretrained model [\[百度云\]](https://pan.baidu.com/s/146T_QCo1HGUL895mCFt8HQ?pwd=ftla) password: ftla

extract them in the root directory

## Step3
Creat a conda environment
```shell script
conda create -n Informer-AFSD python=3.7
```

## Step4
Download all the packages in requirements.txt and build AFSD module
```shell script
pip3 install -r requirements.txt
python3 setup.py develop
```
