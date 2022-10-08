import argparse
import sys
sys.path.append('.') 
from AFSD.evaluation.eval_detection import ANETdetection
import os
import matplotlib.pyplot as plt
from AFSD.common.config import config
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/eval")
parser = argparse.ArgumentParser()
args = parser.parse_args()
max_epoch = config['training']['max_epoch']

max_maps = 0
max_average_map = 0
max_ap = 0
x = []
y = []

for i in range(1, max_epoch + 1):
# for i in range(1, 2):
    gt_json = '/home/lzdjohn/AFSD/AFSD30/thumos_annotations/thumos_gt.json'
    output_json = '/home/lzdjohn/AFSD/AFSD30/AFSD/thumos14/output/'+str(i)+".json"
    tious = [0.3, 0.4, 0.5, 0.6, 0.7]
    anet_detection = ANETdetection(
        ground_truth_filename=gt_json,
        prediction_filename=output_json,
        subset='test', tiou_thresholds=tious)
    mAPs, average_mAP, ap = anet_detection.evaluate()
    print("epoch", i)
    for (tiou, mAP) in zip(tious, mAPs):
        print("mAP at tIoU {} is {}".format(tiou, mAP))
    print(average_mAP, "\n")

    if average_mAP > max_average_map:
        max_epoch = i
        max_maps = mAPs
        max_average_map = average_mAP
        max_ap = ap

    writer.add_scalar('average_mAP', round(average_mAP, 4), i)
    # x.append(i)
    # y.append(round(average_mAP, 4))



# plt.plot(x, y)
# plt.xlabel("epoch")
# plt.ylabel("average mAP")
# plt.show()
# print("The best epoch is", max_epoch)
# print(max_maps)
# print(round(max_average_map, 4))
# print(max_ap[2])
