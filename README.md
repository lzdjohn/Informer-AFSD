# Informer-AFSD
This project includes informer, I3D, MLP, RNN backbones with an AFSD detection part

## Step1
Extract this project into a folder

Download the Wi-Fi dataset and the pretrained model [\[百度云\]](https://pan.baidu.com/s/146T_QCo1HGUL895mCFt8HQ?pwd=ftla) password: ftla

extract them in the root directory

## Step2
Creat a conda environment
```shell script
conda create -n Informer-AFSD python=3.7
```
Install all the packages in requirements.txt and build AFSD module
```shell script
pip3 install -r requirements.txt
python3 setup.py develop
```
##Step3
修改/config/thumos14.yaml中5-7、12-15行

video_info_path: /home/lzdjohn/AFSD/AFSD30/thumos_annotations/val_video_info.csv

video_anno_path: /home/lzdjohn/AFSD/AFSD30/thumos_annotations/val_Annotation_ours.csv

video_data_path: /home/lzdjohn/AFSD/AFSD30/datasets/thumos14/validation_npy/

24行backbone_model: /home/lzdjohn/AFSD/AFSD30/models/thumos14/checkpoint-15.ckpt
    
到自己主机的路径上

修改/AFSD/common/config.py第八行

default='/home/lzdjohn/AFSD/AFSD30/configs/thumos14.yaml', nargs='?')

到自己主机的路径上
