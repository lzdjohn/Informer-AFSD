# Informer-AFSD
This project includes informer, I3D, MLP, RNN backbones with an AFSD detection part

# 安装说明
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
## Step3
修改/config/thumos14.yaml中5-7、12-15行、第24行

video_info_path: /home/lzdjohn/AFSD/AFSD30/thumos_annotations/val_video_info.csv

video_anno_path: /home/lzdjohn/AFSD/AFSD30/thumos_annotations/val_Annotation_ours.csv

video_data_path: /home/lzdjohn/AFSD/AFSD30/datasets/thumos14/validation_npy/

backbone_model: /home/lzdjohn/AFSD/AFSD30/models/thumos14/checkpoint-15.ckpt
    
变更为自己主机的路径上

修改/AFSD/common/config.py第八行

default='/home/lzdjohn/AFSD/AFSD30/configs/thumos14.yaml', nargs='?')

变更为自己主机的路径上

# 使用说明
运行文件在/AFSD/thumos14文件夹中。请运行train.py文件开始训练，每次训练的模型均会存储；训练完毕后用测试集对每轮模型进行测试生成json文件，json文件里标定动作起止时间及概率；最后train文件运行结束。

若要测试，请运行eval.py文件，该文件显示每轮模型得到的json文件与真实json文件数据的tiou精度。
