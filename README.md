# Informer-AFSD
Including informer,MLP,RNN backbones with an AFSD detection part

## Step1
Extract this project in a folder

## Step2
Download the Wi-Fi dataset [\[百度云\]](https://pan.baidu.com/s/10uSTEQQmwOBhA2HXa-Jcuw?pwd=jdsp) password: jdsp

pre-trained model [\[微云\]](https://share.weiyun.com/bP62lmHj) password: w88y

extract them in the root directory

## Step3
Creat a conda environment
```shell script
use: conda create -n Informer-AFSD python=3.7
```

## Step4
Download all packages in requirements.txt, build AFSD module
```shell script
use: pip3 install -r requirements.txt
python3 setup.py develop
```
