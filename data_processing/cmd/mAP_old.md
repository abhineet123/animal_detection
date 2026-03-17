<!-- MarkdownTOC -->

- [yolov2](#yolov_2_)
    - [100_on_5k       @ yolov2](#100_on_5k___yolov2_)
    - [5k       @ yolov2](#5k___yolov2_)
- [yolov1 - darkflow](#yolov1_darkflow_)
    - [5k       @ yolov1_-_darkflow](#5k___yolov1_darkflo_w_)
- [faster_rcnn_inception](#faster_rcnn_inception_)
    - [100_on_5k       @ faster_rcnn_inception](#100_on_5k___faster_rcnn_inceptio_n_)
    - [5k       @ faster_rcnn_inception](#5k___faster_rcnn_inceptio_n_)
- [faster_rcnn_resnet101](#faster_rcnn_resnet101_)
    - [pretrained_on_10kh       @ faster_rcnn_resnet101](#pretrained_on_10kh___faster_rcnn_resnet10_1_)
    - [100_on_5k       @ faster_rcnn_resnet101](#100_on_5k___faster_rcnn_resnet10_1_)
    - [100_on_20k4       @ faster_rcnn_resnet101](#100_on_20k4___faster_rcnn_resnet10_1_)
    - [5k       @ faster_rcnn_resnet101](#5k___faster_rcnn_resnet10_1_)
        - [fiasco       @ 5k/faster_rcnn_resnet101](#fiasco___5k_faster_rcnn_resnet101_)
    - [5k_on_10ka       @ faster_rcnn_resnet101](#5k_on_10ka___faster_rcnn_resnet10_1_)
        - [5k_on_20k4       @ 5k_on_10ka/faster_rcnn_resnet101](#5k_on_20k4___5k_on_10ka_faster_rcnn_resnet101_)
        - [5k_on_train       @ 5k_on_10ka/faster_rcnn_resnet101](#5k_on_train___5k_on_10ka_faster_rcnn_resnet101_)
    - [10k       @ faster_rcnn_resnet101](#10k___faster_rcnn_resnet10_1_)
    - [10k_ar       @ faster_rcnn_resnet101](#10k_ar___faster_rcnn_resnet10_1_)
    - [10ka_on_5k       @ faster_rcnn_resnet101](#10ka_on_5k___faster_rcnn_resnet10_1_)
        - [10ka_on_train       @ 10ka_on_5k/faster_rcnn_resnet101](#10ka_on_train___10ka_on_5k_faster_rcnn_resnet101_)
    - [10ka       @ faster_rcnn_resnet101](#10ka___faster_rcnn_resnet10_1_)
        - [only_TP       @ 10ka/faster_rcnn_resnet101](#only_tp___10ka_faster_rcnn_resnet101_)
        - [10ka_on_20k4       @ 10ka/faster_rcnn_resnet101](#10ka_on_20k4___10ka_faster_rcnn_resnet101_)
    - [20k3       @ faster_rcnn_resnet101](#20k3___faster_rcnn_resnet10_1_)
    - [20k3_hs       @ faster_rcnn_resnet101](#20k3_hs___faster_rcnn_resnet10_1_)
        - [20k3_hs_on_20k3_coco       @ 20k3_hs/faster_rcnn_resnet101](#20k3_hs_on_20k3_coco___20k3_hs_faster_rcnn_resnet10_1_)
        - [20k3_coco       @ 20k3_hs/faster_rcnn_resnet101](#20k3_coco___20k3_hs_faster_rcnn_resnet10_1_)
            - [new       @ 20k3_coco/20k3_hs/faster_rcnn_resnet101](#new___20k3_coco_20k3_hs_faster_rcnn_resnet10_1_)
        - [20k3_on_20k3_no_human       @ 20k3_hs/faster_rcnn_resnet101](#20k3_on_20k3_no_human___20k3_hs_faster_rcnn_resnet10_1_)
        - [20k3_on_20k3_no_human_no_static       @ 20k3_hs/faster_rcnn_resnet101](#20k3_on_20k3_no_human_no_static___20k3_hs_faster_rcnn_resnet10_1_)
        - [20k3_coco_on_40k3       @ 20k3_hs/faster_rcnn_resnet101](#20k3_coco_on_40k3___20k3_hs_faster_rcnn_resnet10_1_)
        - [20k4       @ 20k3_hs/faster_rcnn_resnet101](#20k4___20k3_hs_faster_rcnn_resnet10_1_)
            - [vid       @ 20k4/20k3_hs/faster_rcnn_resnet101](#vid___20k4_20k3_hs_faster_rcnn_resnet101_)
        - [20k4_on_5k       @ 20k3_hs/faster_rcnn_resnet101](#20k4_on_5k___20k3_hs_faster_rcnn_resnet10_1_)
            - [vis       @ 20k4_on_5k/20k3_hs/faster_rcnn_resnet101](#vis___20k4_on_5k_20k3_hs_faster_rcnn_resnet101_)
        - [20k4_on_train       @ 20k3_hs/faster_rcnn_resnet101](#20k4_on_train___20k3_hs_faster_rcnn_resnet10_1_)
        - [20k7_on_20k4       @ 20k3_hs/faster_rcnn_resnet101](#20k7_on_20k4___20k3_hs_faster_rcnn_resnet10_1_)
        - [20k5_on_20k4       @ 20k3_hs/faster_rcnn_resnet101](#20k5_on_20k4___20k3_hs_faster_rcnn_resnet10_1_)
        - [40k3       @ 20k3_hs/faster_rcnn_resnet101](#40k3___20k3_hs_faster_rcnn_resnet10_1_)
        - [40k3_on_20k3       @ 20k3_hs/faster_rcnn_resnet101](#40k3_on_20k3___20k3_hs_faster_rcnn_resnet10_1_)
        - [40k3_on_20k3_no_human_no_static       @ 20k3_hs/faster_rcnn_resnet101](#40k3_on_20k3_no_human_no_static___20k3_hs_faster_rcnn_resnet10_1_)
        - [40k3_on_train       @ 20k3_hs/faster_rcnn_resnet101](#40k3_on_train___20k3_hs_faster_rcnn_resnet10_1_)
        - [1600_static3_on_20k3_no_human       @ 20k3_hs/faster_rcnn_resnet101](#1600_static3_on_20k3_no_human___20k3_hs_faster_rcnn_resnet10_1_)
        - [1600_static3_on_20k3_no_human_no_static       @ 20k3_hs/faster_rcnn_resnet101](#1600_static3_on_20k3_no_human_no_static___20k3_hs_faster_rcnn_resnet10_1_)
        - [1600_static3_on_train       @ 20k3_hs/faster_rcnn_resnet101](#1600_static3_on_train___20k3_hs_faster_rcnn_resnet10_1_)
- [faster_rcnn_nas](#faster_rcnn_nas_)
    - [100_on_5k       @ faster_rcnn_nas](#100_on_5k___faster_rcnn_na_s_)
    - [5k       @ faster_rcnn_nas](#5k___faster_rcnn_na_s_)
- [ssd](#ssd_)
    - [100_on_5k       @ ssd](#100_on_5k___ss_d_)
    - [5k       @ ssd](#5k___ss_d_)
    - [200K       @ ssd](#200k___ss_d_)
    - [25k3       @ ssd](#25k3___ss_d_)
        - [1000       @ 25k3/ssd](#1000___25k3_ssd_)
    - [40k3_coco       @ ssd](#40k3_coco___ss_d_)
- [rfcn](#rfc_n_)
    - [100_on_5k       @ rfcn](#100_on_5k___rfcn_)
        - [40k3       @ 100_on_5k/rfcn](#40k3___100_on_5k_rfcn_)
    - [5k       @ rfcn](#5k___rfcn_)
    - [300K       @ rfcn](#300k___rfcn_)
    - [500K       @ rfcn](#500k___rfcn_)
    - [10k_ar       @ rfcn](#10k_ar___rfcn_)

<!-- /MarkdownTOC -->


<a id="yolov_2_"></a>
# yolov2
python33 mAP.py det_path_list_file=../results/yolo/results/yolov2/marcin/yolov2_list.txt pkl_files_path=pkl_files/yolov2_marcin img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/yolov2_marcin_all.mkv show_anim=1

<a id="100_on_5k___yolov2_"></a>
## 100_on_5k       @ yolov2-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\yolo\yolov2_100_on_5k pkl_files_path=pkl_files/yolov2_100_on_5k img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/yolov2_100_on_5k.mkv save_anim=0 show_anim=0

<a id="5k___yolov2_"></a>
## 5k       @ yolov2-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\yolo\results\acamp5k_yolov2\list.txt" pkl_files_path=pkl_files/acamp5k_yolov2_40K img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images\acamp5k_test_seq_list.txt" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels\acamp5k_test_seq_list.txt" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/acamp5k_yolov2_40K.mkv show_anim=1

<a id="yolov1_darkflow_"></a>
# yolov1 - darkflow

<a id="5k___yolov1_darkflo_w_"></a>
## 5k       @ yolov1_-_darkflow-->mAP_old
python3 eval_det.py det_path_list_file=../darkflow/results/yolo_acamp5k pkl_files_path=pkl_files/yolo_darkflow_acamp5k img_path_list_file=../tf_api/acamp5k_test.txt img_root_dir=/data/acamp/acamp5k/test/images labels_path=../labelling_tool/data/predefined_classes_orig.txt save_file_name=results/videos/yolo_darkflow_acamp5k.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0


<a id="faster_rcnn_inception_"></a>
# faster_rcnn_inception
python33 mAP.py det_path_list_file=../results/faster_rcnn/first_training_f-rcnn_inception_v2/faster_rcnn_list.txt pkl_files_path=pkl_files/first_training_f-rcnn_inception_v2_marcin img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/frcnn_marcin_all.mkv show_anim=1

<a id="100_on_5k___faster_rcnn_inceptio_n_"></a>
## 100_on_5k       @ faster_rcnn_inception-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn\first_training_f-rcnn_inception_v2_100_on_5k\list.txt" pkl_files_path=pkl_files/first_training_f-rcnn_inception_v2_100_on_5k img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images\acamp5k_test_seq_list.txt" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels\acamp5k_test_seq_list.txt" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/first_training_f-rcnn_inception_v2_100_on_5k.mkv save_anim=0 show_anim=0

<a id="5k___faster_rcnn_inceptio_n_"></a>
## 5k       @ faster_rcnn_inception-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn\faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k\inference_200K pkl_files_path=pkl_files/acamp5k_faster_rcnn_inception_200K img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/acamp5k_faster_rcnn_inception_200K.mkv save_anim=0 show_anim=0

<a id="faster_rcnn_resnet101_"></a>
# faster_rcnn_resnet101
python33 mAP.py det_path_list_file=../results/faster_rcnn_resnet101/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_list.txt pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_marcin img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_marcin_all.mkv show_anim=1

<a id="pretrained_on_10kh___faster_rcnn_resnet10_1_"></a>
## pretrained_on_10kh       @ faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_pretrained pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_pretrained_on_10kh img_path_list_file=acamp10kh_test_images.txt gt_path_list_file=acamp10kh_test_labels.txt labels_path=../labelling_tool/data//predefined_classes_person.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_pretrained_on_10kh.mkv show_anim=1


<a id="100_on_5k___faster_rcnn_resnet10_1_"></a>
## 100_on_5k       @ faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_100_on_5k\list.txt" pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_100_on_5k img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images\acamp5k_test_seq_list.txt" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels\acamp5k_test_seq_list.txt" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_100_on_5k.mkv save_anim=0 show_anim=0

<a id="100_on_20k4___faster_rcnn_resnet10_1_"></a>
## 100_on_20k4       @ faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\100\faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4" pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4 img_path_list_file=../tf_api/acamp20k4_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4.mkv save_anim=0 show_anim=0

python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\100\faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4vid" pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4vid img_path_list_file=../tf_api/acamp20k4vid_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4vid.mkv save_anim=0 show_anim=0

<a id="5k___faster_rcnn_resnet10_1_"></a>
## 5k       @ faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_5k\inference_536475" pkl_files_path=pkl_files/acamp5k_faster_rcnn_resnet101_536475 img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/acamp5k_faster_rcnn_resnet101_536475.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1

<a id="fiasco___5k_faster_rcnn_resnet101_"></a>
### fiasco       @ 5k/faster_rcnn_resnet101-->mAP_old
python33 mAP.py det_path_list_file=../results/faster_rcnn_resnet101/faster_rcnn_resnet101_coco_2018_01_28_5k/faster_rcnn_resnet101_5k_list.txt pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_5k img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_5k.mkv show_anim=1

<a id="5k_on_10ka___faster_rcnn_resnet10_1_"></a>
## 5k_on_10ka       @ faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_5k_on_10ka\inference_536475 pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_5k_on_10ka img_path_list_file=acamp10ka_test_images.txt gt_path_list_file=acamp10ka_test_labels.txt labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_5k_on_10ka.mkv save_anim=0 show_anim=0

<a id="5k_on_20k4___5k_on_10ka_faster_rcnn_resnet101_"></a>
### 5k_on_20k4       @ 5k_on_10ka/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\5k\faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4 pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4 img_path_list_file=../tf_api/acamp20k4_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4.mkv save_anim=0 show_anim=0

python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\5k\faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4vid pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4vid img_path_list_file=../tf_api/acamp20k4vid_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4vid.mkv save_anim=0 show_anim=0

<a id="5k_on_train___5k_on_10ka_faster_rcnn_resnet101_"></a>
### 5k_on_train       @ 5k_on_10ka/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_coco_2018_01_28_5k_on_train_inference_536475 pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_5k_on_train_inference_536475 img_path_list_file=../tf_api/acamp5k_train.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_5k_on_train_inference_536475.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="10k___faster_rcnn_resnet10_1_"></a>
## 10k       @ faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_10k\inference_746846" pkl_files_path=pkl_files/acamp10k_faster_rcnn_resnet101_746846 img_path_list_file="N:\Datasets\Acamp\acamp10k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp10k\test\labels" labels_path=../labelling_tool/data//predefined_classes_10k.txt save_file_name=results/videos/acamp10k_faster_rcnn_resnet101_746846.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1


<a id="10k_ar___faster_rcnn_resnet10_1_"></a>
## 10k_ar       @ faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\batch_6\faster_rcnn_resnet101_coco_2018_01_28_10k_ar pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10k_ar_batch_6 img_path_list_file="N:\Datasets\Acamp\acamp10k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp10k\test\labels" labels_path=../labelling_tool/data//predefined_classes_10k.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10k_ar_batch_6.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1

<a id="10ka_on_5k___faster_rcnn_resnet10_1_"></a>
## 10ka_on_5k       @ faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_10ka_on_5k pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_5k img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images\acamp5k_test_seq_list.txt" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels\acamp5k_test_seq_list.txt" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_5k.mkv save_anim=0 show_anim=0

<a id="10ka_on_train___10ka_on_5k_faster_rcnn_resnet101_"></a>
### 10ka_on_train       @ 10ka_on_5k/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\10k\faster_rcnn_resnet101_coco_2018_01_28_10ka_on_train pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_train img_path_list_file=../tf_api/acamp10ka_train.txt img_root_dir=N:\Datasets\Acamp\acamp20k labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_train.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="10ka___faster_rcnn_resnet10_1_"></a>
## 10ka       @ faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_10ka pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10ka img_path_list_file=acamp10ka_test_images.txt gt_path_list_file=acamp10ka_test_labels.txt labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10ka.mkv save_anim=0 show_anim=0

<a id="only_tp___10ka_faster_rcnn_resnet101_"></a>
### only_TP       @ 10ka/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_5k\inference_536475" pkl_files_path=pkl_files/acamp5k_faster_rcnn_resnet101_536475 img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/acamp5k_faster_rcnn_resnet101_536475_no_gt_stats_only_tp.mkv show_anim=1 show_stats=0 show_gt=0

<a id="10ka_on_20k4___10ka_faster_rcnn_resnet101_"></a>
### 10ka_on_20k4       @ 10ka/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\10k\faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4 pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4 img_path_list_file=../tf_api/acamp20k4_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4.mkv save_anim=0 show_anim=0

python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\10k\faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4vid pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4vid img_path_list_file=../tf_api/acamp20k4vid_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4vid.mkv save_anim=0 show_anim=0


<a id="20k3___faster_rcnn_resnet10_1_"></a>
## 20k3       @ faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851" pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851 img_path_list_file=../tf_api/acamp20k3_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" labels_path=../labelling_tool/data//predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1


<a id="20k3_hs___faster_rcnn_resnet10_1_"></a>
## 20k3_hs       @ faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k3_hs_inference_474099" pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs_inference_474099 img_path_list_file=../tf_api/acamp20k3_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" labels_path=../labelling_tool/data//predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs_inference_474099.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1

<a id="20k3_hs_on_20k3_coco___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 20k3_hs_on_20k3_coco       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_20k3_coco_inference_769115" pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco_inference_769115 img_path_list_file=../tf_api/acamp20k3_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" labels_path=../labelling_tool/data//predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco_inference_769115.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1

<a id="20k3_coco___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 20k3_coco       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851_20k3_coco" pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851_20k3_coco img_path_list_file=../tf_api/acamp20k3_coco_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" labels_path=../labelling_tool/data//predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851_20k3_coco.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1

python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_20k3_coco_inference_801515" pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_801515_20k3_coco img_path_list_file=../tf_api/acamp20k3_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" labels_path=../labelling_tool/data//predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_801515_20k3_coco.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1

<a id="new___20k3_coco_20k3_hs_faster_rcnn_resnet10_1_"></a>
#### new       @ 20k3_coco/20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_10k_on_20k3_coco" pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_10k_on_20k3_coco img_path_list_file=../tf_api/acamp20k3_coco_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k gt_root_dir="N:\Datasets\Acamp\acamp20k\test\labels" labels_path=../labelling_tool/data//predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_10k_on_20k3_coco.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1

<a id="20k3_on_20k3_no_human___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 20k3_on_20k3_no_human       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human pkl_files_path=pkl_files/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human img_path_list_file=../tf_api/acamp20k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="20k3_on_20k3_no_human_no_static___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 20k3_on_20k3_no_human_no_static       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human_no_static pkl_files_path=pkl_files/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human_no_static img_path_list_file=../tf_api/acamp20k3_test_no_human_no_static.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp20k3_test_no_human_no_static.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="20k3_coco_on_40k3___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 20k3_coco_on_40k3       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp40k3_test_no_human pkl_files_path=pkl_files/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp40k3_test_no_human img_path_list_file=../tf_api/acamp40k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_20K_3_class_COCO_801K_steps_on_acamp40k3_test_no_human.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="20k4___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 20k4       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k4_inference_1050850 pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k4_inference_1050850 img_path_list_file=../tf_api/acamp20k4_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k4_inference_1050850.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="vid___20k4_20k3_hs_faster_rcnn_resnet101_"></a>
#### vid       @ 20k4/20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k4vid_inference_1050850 pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k4vid_inference_1050850 img_path_list_file=../tf_api/acamp20k4vid_test.txt img_root_dir=N:\Datasets\Acamp\acamp20k labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k4vid_inference_1050850.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="20k4_on_5k___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 20k4_on_5k       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850 pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850 img_path_list_file=../tf_api/acamp5k_test.txt img_root_dir=N:\Datasets\Acamp\acamp5k\test\images gt_root_dir="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="vis___20k4_on_5k_20k3_hs_faster_rcnn_resnet101_"></a>
#### vis       @ 20k4_on_5k/20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850 pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850 img_path_list_file=../tf_api/acamp5k_test.txt img_root_dir=N:\Datasets\Acamp\acamp5k\test\images gt_root_dir="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_5k_inference_1050850.mkv show_anim=1 show_stats=0 show_gt=0 show_tp=2 show_text=0

<a id="20k4_on_train___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 20k4_on_train       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\20k\faster_rcnn_resnet101_coco_2018_01_28_20k4_on_train_inference_1050850 pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_train_inference_1050850 img_path_list_file=../tf_api/acamp20k4_train.txt img_root_dir=N:\Datasets\Acamp\acamp20k labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_train_inference_1050850.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="20k7_on_20k4___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 20k7_on_20k4       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_coco_2018_01_28_20k7_on_20k4 pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k7_on_20k4 img_path_list_file=../tf_api/acamp20k4_test.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k7_on_20k4.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="20k5_on_20k4___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 20k5_on_20k4       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_coco_2018_01_28_20k5_on_20k4 pkl_files_path=pkl_files/faster_rcnn_resnet101_coco_2018_01_28_20k5_on_20k4 img_path_list_file=../tf_api/acamp20k4_test.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_resnet101_coco_2018_01_28_20k5_on_20k4.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="40k3___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 40k3       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_40K_3_class_1182K_steps pkl_files_path=pkl_files/faster_rcnn_resnet101_40K_3_class_1182K_steps img_path_list_file=../tf_api/acamp40k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_40K_3_class_1182K_steps.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="40k3_on_20k3___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 40k3_on_20k3       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3 pkl_files_path=pkl_files/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3 img_path_list_file=../tf_api/acamp20k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0


<a id="40k3_on_20k3_no_human_no_static___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 40k3_on_20k3_no_human_no_static       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3_no_human_no_static pkl_files_path=pkl_files/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3_no_human_no_static img_path_list_file=../tf_api/acamp20k3_test_no_human_no_static.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_20k3_no_human_no_static.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="40k3_on_train___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 40k3_on_train       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_acamp40k3_coco_train pkl_files_path=pkl_files/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_acamp40k3_coco_train img_path_list_file=../tf_api/acamp40k3_coco_train.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_40K_3_class_1182K_steps_on_acamp40k3_coco_train.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="1600_static3_on_20k3_no_human___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 1600_static3_on_20k3_no_human       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human pkl_files_path=pkl_files/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human img_path_list_file=../tf_api/acamp20k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="1600_static3_on_20k3_no_human_no_static___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 1600_static3_on_20k3_no_human_no_static       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human_no_static pkl_files_path=pkl_files/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human_no_static img_path_list_file=../tf_api/acamp20k3_test_no_human_no_static.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp20k3_test_no_human_no_static.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0

<a id="1600_static3_on_train___20k3_hs_faster_rcnn_resnet10_1_"></a>
### 1600_static3_on_train       @ 20k3_hs/faster_rcnn_resnet101-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp1600_static3_train pkl_files_path=pkl_files/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp1600_static3_train img_path_list_file=../tf_api/acamp1600_static3_train.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_1600_static_3_class_1000K_steps_on_acamp1600_static3_train.mkv save_anim=0 show_anim=0 show_stats=0 show_gt=0 show_tp=1 show_text=0


<a id="faster_rcnn_nas_"></a>
# faster_rcnn_nas

<a id="100_on_5k___faster_rcnn_na_s_"></a>
## 100_on_5k       @ faster_rcnn_nas-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\faster_rcnn_nas\faster_rcnn_nas_coco_2018_01_28_100_on_5k pkl_files_path=pkl_files/faster_rcnn_nas_coco_2018_01_28_100_on_5k img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_nas_coco_2018_01_28_100_on_5k.mkv save_anim=0 show_anim=0

python33 mAP.py det_path_list_file=../results/faster_rcnn_nas/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_list.txt pkl_files_path=pkl_files/faster_rcnn_nas_coco_2018_01_28_marcin img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_nas_coco_2018_01_28_marcin_all.mkv show_anim=1

<a id="5k___faster_rcnn_na_s_"></a>
## 5k       @ faster_rcnn_nas-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\faster_rcnn_nas\faster_rcnn_nas_coco_2018_01_28_5k\inference_200K" pkl_files_path=pkl_files/faster_rcnn_nas_coco_2018_01_28_5k_200K img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/faster_rcnn_nas_coco_2018_01_28_5k_200K.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1

<a id="ssd_"></a>
# ssd
python33 mAP.py det_path_list_file=../results/ssd/ssd_inception_v2_coco_2017_11_17/ssd_list.txt pkl_files_path=pkl_files/ssd_inception_v2_coco_2017_11_17 img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/ssd_inception_v2_coco_2017_11_17_marcin_all.mkv show_anim=1

<a id="100_on_5k___ss_d_"></a>
## 100_on_5k       @ ssd-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\ssd\ssd_inception_v2_coco_2017_11_17_100_on_5k pkl_files_path=pkl_files/ssd_inception_v2_coco_2017_11_17_100_on_5k img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/ssd_inception_v2_coco_2017_11_17_100_on_5k.mkv save_anim=0 show_anim=0

<a id="5k___ss_d_"></a>
## 5k       @ ssd-->mAP_old

<a id="200k___ss_d_"></a>
## 200K       @ ssd-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\ssd\ssd_inception_v2_coco_2017_11_17_5k pkl_files_path=pkl_files/ssd_inception_v2_coco_2017_11_17_5k img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/ssd_inception_v2_coco_2017_11_17_5k.mkv save_anim=0 show_anim=0

<a id="25k3___ss_d_"></a>
## 25k3       @ ssd-->mAP_old

<a id="1000___25k3_ssd_"></a>
### 1000       @ 25k3/ssd-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_25K_3_class_1000K_steps_on_acamp40k3_test_no_human pkl_files_path=pkl_files/faster_rcnn_resnet101_25K_3_class_1000K_steps_on_acamp40k3_test_no_human img_path_list_file=../tf_api/acamp40k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_25K_3_class_1000K_steps_on_acamp40k3_test_no_human.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1 show_tp=1 show_text=0

python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_25K_3_class_800K_steps_on_acamp40k3_test_no_human pkl_files_path=pkl_files/faster_rcnn_resnet101_25K_3_class_800K_steps_on_acamp40k3_test_no_human img_path_list_file=../tf_api/acamp40k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_25K_3_class_800K_steps_on_acamp40k3_test_no_human.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1 show_tp=1 show_text=0

python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_25K_3_class_600K_steps_on_acamp40k3_test_no_human pkl_files_path=pkl_files/faster_rcnn_resnet101_25K_3_class_600K_steps_on_acamp40k3_test_no_human img_path_list_file=../tf_api/acamp40k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_25K_3_class_600K_steps_on_acamp40k3_test_no_human.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1 show_tp=1 show_text=0

python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_25K_3_class_400K_steps_on_acamp40k3_test_no_human pkl_files_path=pkl_files/faster_rcnn_resnet101_25K_3_class_400K_steps_on_acamp40k3_test_no_human img_path_list_file=../tf_api/acamp40k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_25K_3_class_400K_steps_on_acamp40k3_test_no_human.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1 show_tp=1 show_text=0

python3 eval_det.py det_path_list_file=../tf_api/results/faster_rcnn_resnet101_25K_3_class_200K_steps_on_acamp40k3_test_no_human pkl_files_path=pkl_files/faster_rcnn_resnet101_25K_3_class_200K_steps_on_acamp40k3_test_no_human img_path_list_file=../tf_api/acamp40k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/faster_rcnn_resnet101_25K_3_class_200K_steps_on_acamp40k3_test_no_human.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1 show_tp=1 show_text=0

<a id="40k3_coco___ss_d_"></a>
## 40k3_coco       @ ssd-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/ssd_40K_3_class_215K_steps_on_acamp40k3_test_no_human pkl_files_path=pkl_files/ssd_40K_3_class_215K_steps_on_acamp40k3_test_no_human img_path_list_file=../tf_api/acamp40k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/ssd_40K_3_class_215K_steps_on_acamp40k3_test_no_human.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1 show_tp=1 show_text=0

<a id="rfc_n_"></a>
# rfcn
python33 mAP.py det_path_list_file=../results/rfcn/rfcn_resnet101_coco_2018_01_28/rfcn_list.txt pkl_files_path=pkl_files/rfcn_resnet101_coco_2018_01_28 img_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/img_list.txt gt_path_list_file=/home/abhineet/N/Datasets/Acamp/marcin_180608/map_list.txt labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/rfcn_resnet101_coco_2018_01_28_marcin_all.mkv show_anim=1

<a id="100_on_5k___rfcn_"></a>
## 100_on_5k       @ rfcn-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\rfcn\rfcn_resnet101_coco_2018_01_28_100_on_5k" pkl_files_path=pkl_files/rfcn_resnet101_coco_2018_01_28_100_on_5k img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/rfcn_resnet101_coco_2018_01_28_100_on_5k.mkv save_anim=0 show_anim=0

<a id="40k3___100_on_5k_rfcn_"></a>
### 40k3       @ 100_on_5k/rfcn-->mAP_old
python3 eval_det.py det_path_list_file=../tf_api/results/rfcn_40K_3_class_1243K_steps pkl_files_path=pkl_files/rfcn_40K_3_class_1243K_steps img_path_list_file=../tf_api/acamp40k3_test_no_human.txt img_root_dir=/data/acamp/acamp20k labels_path=../labelling_tool/data/predefined_classes_20k3.txt save_file_name=results/videos/rfcn_40K_3_class_1243K_steps.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1 show_tp=1 show_text=0

<a id="5k___rfcn_"></a>
## 5k       @ rfcn-->mAP_old

<a id="300k___rfcn_"></a>
## 300K       @ rfcn-->mAP_old
python3 eval_det.py det_path_list_file="H:\UofA\Acamp\code\results\rfcn\rfcn_resnet101_coco_2018_01_28_5k\rfcn_list.txt" pkl_files_path=pkl_files/acamp5k_rfcn_300K img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images\acamp5k_test_seq_list.txt" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels\acamp5k_test_seq_list.txt" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/rfcn_resnet101_coco_2018_01_28_5k_300K.mkv save_anim=0 show_anim=0

<a id="500k___rfcn_"></a>
## 500K       @ rfcn-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\rfcn\rfcn_resnet101_coco_2018_01_28_5k_500K pkl_files_path=pkl_files/rfcn_resnet101_coco_2018_01_28_5k_500K img_path_list_file="N:\Datasets\Acamp\acamp5k\test\images" gt_path_list_file="N:\Datasets\Acamp\acamp5k\test\labels" labels_path=../labelling_tool/data//predefined_classes_orig.txt save_file_name=results/videos/rfcn_resnet101_coco_2018_01_28_5k_500K.mkv save_anim=0 show_anim=0

<a id="10k_ar___rfcn_"></a>
## 10k_ar       @ rfcn-->mAP_old
python3 eval_det.py det_path_list_file=H:\UofA\Acamp\code\results\rfcn\batch_6\rfcn_resnet101_coco_2018_01_28_10k_ar_500K pkl_files_path=pkl_files/rfcn_resnet101_coco_2018_01_28_10k_ar_500K_batch_6 img_path_list_file=../labelling_tool/acamp10k_test.txt gt_path_list_file="N:\Datasets\Acamp\acamp10k\test\labels" labels_path=../labelling_tool/data/predefined_classes_10k.txt save_file_name=results/videos/rfcn_resnet101_coco_2018_01_28_10k_ar_500K_batch_6.mkv save_anim=0 show_anim=0 show_stats=1 show_gt=1
