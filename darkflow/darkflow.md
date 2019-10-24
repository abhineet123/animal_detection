# 5k

## training

CUDA_VISIBLE_DEVICES=0 python3 flow --model cfg/yolo_acamp5k.cfg --labels ../labelling_tool/data/predefined_classes_orig.txt --load ../yolo/pretrained/darknet19_448.conv.23 --train --dataset /data/acamp/acamp5k/train/images --batch 16 --gpu 0.9 --gpuName /gpu:0

CUDA_VISIBLE_DEVICES=0 python3 flow --model cfg/yolo_acamp5k.cfg --labels ../labelling_tool/data/predefined_classes_orig.txt --load ../yolo/pretrained/yolo.weights --train --dataset /data/acamp/acamp5k/train/images --batch 16 --gpu 0.9 --gpuName /gpu:0

## testing

CUDA_VISIBLE_DEVICES=1 python3 flow --model cfg/yolo_acamp5k.cfg --labels ../labelling_tool/data/predefined_classes_orig.txt --load 104125 --dataset /data/acamp/acamp5k/test/images --out_dir results/yolo_acamp5k --batch 16 --gpu 0.9 --gpuName /gpu:1
