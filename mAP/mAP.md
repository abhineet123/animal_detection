python3 main.py --det_path=../results/yolo/results/yolov2/marcin/bear_1_1_mAP --img_path=/home/abhineet/N/Datasets/Acamp/marcin/bear_1_1 --gt_path=/home/abhineet/N/Datasets/Acamp/marcin/bear_1_1/mAP

# ========================================================= #
# yolov2
# ========================================================= #

python3 main.py --det_path_list_file=../results/yolo/results/yolov2/marcin/yolov2_list.txt --pkl_files_path=pkl_files/yolov2_marcin --img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt --gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/yolov2_marcin_all.mkv --show_animation=1

## 100_on_5k

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\yolo\yolov2_100_on_5k --pkl_files_path=pkl_files/yolov2_100_on_5k --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/yolov2_100_on_5k.mkv --save_animation=0 --show_animation=0

## 5k

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\yolo\results\acamp5k_yolov2\list.txt" --pkl_files_path=pkl_files/acamp5k_yolov2_40K --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images\acamp5k_test_seq_list.txt" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels\acamp5k_test_seq_list.txt" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/acamp5k_yolov2_40K.mkv --show_animation=1

# ========================================================= #
# yolov1 - darkflow
# ========================================================= #

## 5k

python main.py --det_path_list_file=../darkflow/results/yolo_acamp5k --pkl_files_path=pkl_files/yolo_darkflow_acamp5k --img_path_list_file=../tf_api/acamp5k_test.txt --img_root_dir=/data/acamp/acamp5k/test/images --labels_path=../labelling_tool/data/predefined_classes_orig.txt --save_file_name=results/videos/yolo_darkflow_acamp5k.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0


# ========================================================= #
# faster_rcnn_inception
# ========================================================= #

python3 main.py --det_path_list_file=../results/faster_rcnn/first_training_f-rcnn_inception_v2/faster_rcnn_list.txt --pkl_files_path=pkl_files/first_training_f-rcnn_inception_v2_marcin --img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt --gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/frcnn_marcin_all.mkv --show_animation=1

## 100_on_5k

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn\first_training_f-rcnn_inception_v2_100_on_5k\list.txt" --pkl_files_path=pkl_files/first_training_f-rcnn_inception_v2_100_on_5k --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images\acamp5k_test_seq_list.txt" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels\acamp5k_test_seq_list.txt" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/first_training_f-rcnn_inception_v2_100_on_5k.mkv --save_animation=0 --show_animation=0

## 5k

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn\faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k\inference_200K --pkl_files_path=pkl_files/acamp5k_faster_rcnn_inception_200K --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/acamp5k_faster_rcnn_inception_200K.mkv --save_animation=0 --show_animation=0

# ========================================================= #
# faster_rcnn_resnet101
# ========================================================= #

python3 main.py --det_path_list_file=../results/faster_rcnn_resnet101/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_list.txt --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_marcin --img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt --gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_marcin_all.mkv --show_animation=1

## pretrained on 10kh

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_pretrained --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_pretrained_on_10kh --img_path_list_file=acamp10kh_test_images.txt --gt_path_list_file=acamp10kh_test_labels.txt --labels_path=../labelling_tool/data//predefined_classes_person.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_pretrained_on_10kh.mkv --show_animation=1


## 100_on_5k

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_100_on_5k\list.txt" --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_100_on_5k --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images\acamp5k_test_seq_list.txt" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels\acamp5k_test_seq_list.txt" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_100_on_5k.mkv --save_animation=0 --show_animation=0

## 100_on_20k4

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\100\faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4" --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4 --img_path_list_file=../tf_api/acamp20k4_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4.mkv --save_animation=0 --show_animation=0

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\100\faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4vid" --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4vid --img_path_list_file=../tf_api/acamp20k4vid_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4vid.mkv --save_animation=0 --show_animation=0

## 5k

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_5k\inference_536475" --pkl_files_path=pkl_files/acamp5k_faster_rcnn_resnet101_536475 --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/acamp5k_faster_rcnn_resnet101_536475.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1

### fiasco

python3 main.py --det_path_list_file=../results/faster_rcnn_resnet101/faster_rcnn_resnet101_coco_2018_01_28_5k/faster_rcnn_resnet101_5k_list.txt --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_5k --img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt --gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_5k.mkv --show_animation=1

## 5k on 10ka

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_5k_on_10ka\inference_536475 --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_5k_on_10ka --img_path_list_file=acamp10ka_test_images.txt --gt_path_list_file=acamp10ka_test_labels.txt --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_5k_on_10ka.mkv --save_animation=0 --show_animation=0

### 5k on 20k4

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\5k\faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4 --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4 --img_path_list_file=../tf_api/acamp20k4_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4.mkv --save_animation=0 --show_animation=0

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\5k\faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4vid --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4vid --img_path_list_file=../tf_api/acamp20k4vid_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4vid.mkv --save_animation=0 --show_animation=0

### 5k on train

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_coco_2018_01_28_5k_on_train_inference_536475 --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_5k_on_train_inference_536475 --img_path_list_file=../tf_api/acamp5k_train.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_5k_on_train_inference_536475.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

## 10k

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_10k\inference_746846" --pkl_files_path=pkl_files/acamp10k_faster_rcnn_resnet101_746846 --img_path_list_file="N:\Datasets\Acamp\acamp10k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp10k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_10k.txt --save_file_name=results/videos/acamp10k_faster_rcnn_resnet101_746846.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1


## 10k_ar

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\batch_6\faster_rcnn_resnet101_coco_2018_01_28_10k_ar --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10k_ar_batch_6 --img_path_list_file="N:\Datasets\Acamp\acamp10k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp10k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_10k.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10k_ar_batch_6.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1

## 10ka_on_5k

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_10ka_on_5k --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_5k --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images\acamp5k_test_seq_list.txt" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels\acamp5k_test_seq_list.txt" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_5k.mkv --save_animation=0 --show_animation=0

### 10ka on train

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\10k\faster_rcnn_resnet101_coco_2018_01_28_10ka_on_train --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_train --img_path_list_file=../tf_api/acamp10ka_train.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_train.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

## 10ka

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_10ka --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10ka --img_path_list_file=acamp10ka_test_images.txt --gt_path_list_file=acamp10ka_test_labels.txt --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10ka.mkv --save_animation=0 --show_animation=0

### only TP

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_5k\inference_536475" --pkl_files_path=pkl_files/acamp5k_faster_rcnn_resnet101_536475 --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/acamp5k_faster_rcnn_resnet101_536475_no_gt_stats_only_tp.mkv --show_animation=1 --show_stats=0 --show_gt=0

### 10ka on 20k4

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\10k\faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4 --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4 --img_path_list_file=../tf_api/acamp20k4_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4.mkv --save_animation=0 --show_animation=0

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\10k\faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4vid --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4vid --img_path_list_file=../tf_api/acamp20k4vid_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4vid.mkv --save_animation=0 --show_animation=0


## 20k3

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851" --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851 --img_path_list_file=../tf_api/acamp20k3_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1


## 20k3_hs

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k3_hs_inference_474099" --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs_inference_474099 --img_path_list_file=../tf_api/acamp20k3_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs_inference_474099.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1

### 20k3_hs on 20k3_coco

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_20k3_coco_inference_769115" --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco_inference_769115 --img_path_list_file=../tf_api/acamp20k3_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco_inference_769115.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1

### 20k3_coco

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851_20k3_coco" --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851_20k3_coco --img_path_list_file=../tf_api/acamp20k3_coco_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851_20k3_coco.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_20k3_coco_inference_801515" --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_801515_20k3_coco --img_path_list_file=../tf_api/acamp20k3_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_801515_20k3_coco.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1

#### new

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_10k_on_20k3_coco" --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10k_on_20k3_coco --img_path_list_file=../tf_api/acamp20k3_coco_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10k_on_20k3_coco.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1

### 20k3 on 20k3_no_human

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human --pkl_files_path=pkl_files/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human --img_path_list_file=../tf_api/acamp20k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 20k3 on 20k3_no_human_no_static

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human_no_static --pkl_files_path=pkl_files/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human_no_static --img_path_list_file=../tf_api/acamp20k3_test_no_human_no_static.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human_no_static.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 20k3_coco on 40k3

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp40k3_test_no_human --pkl_files_path=pkl_files/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp40k3_test_no_human --img_path_list_file=../tf_api/acamp40k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp40k3_test_no_human.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 20k4

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k4_inference_1050850 --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k4_inference_1050850 --img_path_list_file=../tf_api/acamp20k4_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k4_inference_1050850.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

#### vid

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k4vid_inference_1050850 --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k4vid_inference_1050850 --img_path_list_file=../tf_api/acamp20k4vid_test.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k4vid_inference_1050850.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 20k4 on 5k

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850 --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850 --img_path_list_file=../tf_api/acamp5k_test.txt --img_root_dir=N:\Datasets\Acamp\acamp5k\test\images --gt_root_dir="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

#### vis

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850 --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850 --img_path_list_file=../tf_api/acamp5k_test.txt --img_root_dir=N:\Datasets\Acamp\acamp5k\test\images --gt_root_dir="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850.mkv --show_animation=1 --show_stats=0 --show_gt=0 --show_only_tp=1 --show_text=0

### 20k4 on train

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k4_on_train_inference_1050850 --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_train_inference_1050850 --img_path_list_file=../tf_api/acamp20k4_train.txt --img_root_dir=N:\Datasets\Acamp\acamp20k --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_train_inference_1050850.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 20k7 on 20k4

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_coco_2018_01_28_20k7_on_20k4 --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k7_on_20k4 --img_path_list_file=../tf_api/acamp20k4_test.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k7_on_20k4.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 20k5 on 20k4

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_coco_2018_01_28_20k5_on_20k4 --pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k5_on_20k4 --img_path_list_file=../tf_api/acamp20k4_test.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k5_on_20k4.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 40k3

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_40K_3_class_1182K_steps --pkl_files_path=pkl_files/faster_rcnn_resnet101_40K_3_class_1182K_steps --img_path_list_file=../tf_api/acamp40k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_40K_3_class_1182K_steps.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 40k3 on 20k3

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3 --pkl_files_path=pkl_files/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3 --img_path_list_file=../tf_api/acamp20k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0


### 40k3 on 20k3_no_human_no_static

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3_no_human_no_static --pkl_files_path=pkl_files/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3_no_human_no_static --img_path_list_file=../tf_api/acamp20k3_test_no_human_no_static.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3_no_human_no_static.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 40k3 on train

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_acamp40k3_coco_train --pkl_files_path=pkl_files/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_acamp40k3_coco_train --img_path_list_file=../tf_api/acamp40k3_coco_train.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_acamp40k3_coco_train.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 1600_static3 on 20k3_no_human

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human --pkl_files_path=pkl_files/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human --img_path_list_file=../tf_api/acamp20k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 1600_static3 on 20k3_no_human_no_static

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human_no_static --pkl_files_path=pkl_files/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human_no_static --img_path_list_file=../tf_api/acamp20k3_test_no_human_no_static.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human_no_static.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

### 1600_static3 on train

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp1600_static3_train --pkl_files_path=pkl_files/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp1600_static3_train --img_path_list_file=../tf_api/acamp1600_static3_train.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp1600_static3_train.mkv --save_animation=0 --show_animation=0 --show_stats=0 --show_gt=0 --show_only_tp=0 --show_text=0

# ========================================================= #
# faster_rcnn_nas
# ========================================================= #

## 100_on_5k

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_nas\faster_rcnn_nas_coco_2018_01_28_100_on_5k --pkl_files_path=pkl_files/faster_rcnn_nas_coco_2018_01_28_100_on_5k --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_nas_coco_2018_01_28_100_on_5k.mkv --save_animation=0 --show_animation=0

python3 main.py --det_path_list_file=../results/faster_rcnn_nas/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_list.txt --pkl_files_path=pkl_files/faster_rcnn_nas_coco_2018_01_28_marcin --img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt --gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_nas_coco_2018_01_28_marcin_all.mkv --show_animation=1

## 5k

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_nas\faster_rcnn_nas_coco_2018_01_28_5k\inference_200K" --pkl_files_path=pkl_files/faster_rcnn_nas_coco_2018_01_28_5k_200K --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/faster_rcnn_nas_coco_2018_01_28_5k_200K.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1

# ========================================================= #
# ssd
# ========================================================= #

python3 main.py --det_path_list_file=../results/ssd/ssd_inception_v2_coco_2017_11_17/ssd_list.txt --pkl_files_path=pkl_files/ssd_inception_v2_coco_2017_11_17 --img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt --gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/ssd_inception_v2_coco_2017_11_17_marcin_all.mkv --show_animation=1

## 100_on_5k

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\ssd\ssd_inception_v2_coco_2017_11_17_100_on_5k --pkl_files_path=pkl_files/ssd_inception_v2_coco_2017_11_17_100_on_5k --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/ssd_inception_v2_coco_2017_11_17_100_on_5k.mkv --save_animation=0 --show_animation=0

## 5k

## 200K

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\ssd\ssd_inception_v2_coco_2017_11_17_5k --pkl_files_path=pkl_files/ssd_inception_v2_coco_2017_11_17_5k --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/ssd_inception_v2_coco_2017_11_17_5k.mkv --save_animation=0 --show_animation=0

## 25k3

### 1000

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_25K_3_class_1000K_steps_on_acamp40k3_test_no_human --pkl_files_path=pkl_files/faster_rcnn_resnet101_25K_3_class_1000K_steps_on_acamp40k3_test_no_human --img_path_list_file=../tf_api/acamp40k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_25K_3_class_1000K_steps_on_acamp40k3_test_no_human.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1 --show_only_tp=0 --show_text=0

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_25K_3_class_800K_steps_on_acamp40k3_test_no_human --pkl_files_path=pkl_files/faster_rcnn_resnet101_25K_3_class_800K_steps_on_acamp40k3_test_no_human --img_path_list_file=../tf_api/acamp40k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_25K_3_class_800K_steps_on_acamp40k3_test_no_human.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1 --show_only_tp=0 --show_text=0

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_25K_3_class_600K_steps_on_acamp40k3_test_no_human --pkl_files_path=pkl_files/faster_rcnn_resnet101_25K_3_class_600K_steps_on_acamp40k3_test_no_human --img_path_list_file=../tf_api/acamp40k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_25K_3_class_600K_steps_on_acamp40k3_test_no_human.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1 --show_only_tp=0 --show_text=0

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_25K_3_class_400K_steps_on_acamp40k3_test_no_human --pkl_files_path=pkl_files/faster_rcnn_resnet101_25K_3_class_400K_steps_on_acamp40k3_test_no_human --img_path_list_file=../tf_api/acamp40k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_25K_3_class_400K_steps_on_acamp40k3_test_no_human.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1 --show_only_tp=0 --show_text=0

python main.py --det_path_list_file=../tf_api/results/faster_rcnn_resnet101_25K_3_class_200K_steps_on_acamp40k3_test_no_human --pkl_files_path=pkl_files/faster_rcnn_resnet101_25K_3_class_200K_steps_on_acamp40k3_test_no_human --img_path_list_file=../tf_api/acamp40k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/faster_rcnn_resnet101_25K_3_class_200K_steps_on_acamp40k3_test_no_human.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1 --show_only_tp=0 --show_text=0

## 40k3_coco

python main.py --det_path_list_file=../tf_api/results/ssd_40K_3_class_215K_steps_on_acamp40k3_test_no_human --pkl_files_path=pkl_files/ssd_40K_3_class_215K_steps_on_acamp40k3_test_no_human --img_path_list_file=../tf_api/acamp40k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/ssd_40K_3_class_215K_steps_on_acamp40k3_test_no_human.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1 --show_only_tp=0 --show_text=0

# ========================================================= #
# rfcn
# ========================================================= #

python3 main.py --det_path_list_file=../results/rfcn/rfcn_resnet101_coco_2018_01_28/rfcn_list.txt --pkl_files_path=pkl_files/rfcn_resnet101_coco_2018_01_28 --img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt --gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/rfcn_resnet101_coco_2018_01_28_marcin_all.mkv --show_animation=1

## 100_on_5k

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\rfcn\rfcn_resnet101_coco_2018_01_28_100_on_5k" --pkl_files_path=pkl_files/rfcn_resnet101_coco_2018_01_28_100_on_5k --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/rfcn_resnet101_coco_2018_01_28_100_on_5k.mkv --save_animation=0 --show_animation=0

### 40k3

python main.py --det_path_list_file=../tf_api/results/rfcn_40K_3_class_1243K_steps --pkl_files_path=pkl_files/rfcn_40K_3_class_1243K_steps --img_path_list_file=../tf_api/acamp40k3_test_no_human.txt --img_root_dir=/data/acamp/acamp20k --labels_path=../labelling_tool/data/predefined_classes_20k3.txt --save_file_name=results/videos/rfcn_40K_3_class_1243K_steps.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1 --show_only_tp=0 --show_text=0

## 5k

## 300K

python main.py --det_path_list_file="H:\UofA\Acamp\code\results\rfcn\rfcn_resnet101_coco_2018_01_28_5k\rfcn_list.txt" --pkl_files_path=pkl_files/acamp5k_rfcn_300K --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images\acamp5k_test_seq_list.txt" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels\acamp5k_test_seq_list.txt" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/rfcn_resnet101_coco_2018_01_28_5k_300K.mkv --save_animation=0 --show_animation=0

## 500K

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\rfcn\rfcn_resnet101_coco_2018_01_28_5k_500K --pkl_files_path=pkl_files/rfcn_resnet101_coco_2018_01_28_5k_500K --img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" --gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" --labels_path=../labelling_tool/data//predefined_classes_orig.txt --save_file_name=results/videos/rfcn_resnet101_coco_2018_01_28_5k_500K.mkv --save_animation=0 --show_animation=0

## 10k_ar

python main.py --det_path_list_file=H:\UofA\Acamp\code\results\rfcn\batch_6\rfcn_resnet101_coco_2018_01_28_10k_ar_500K --pkl_files_path=pkl_files/rfcn_resnet101_coco_2018_01_28_10k_ar_500K_batch_6 --img_path_list_file=../labelling_tool/acamp10k_test.txt --gt_path_list_file="N:\Datasets\Acamp\acamp10k\test\labels" --labels_path=../labelling_tool/data/predefined_classes_10k.txt --save_file_name=results/videos/rfcn_resnet101_coco_2018_01_28_10k_ar_500K_batch_6.mkv --save_animation=0 --show_animation=0 --show_stats=1 --show_gt=1




