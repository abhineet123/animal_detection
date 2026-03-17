<!-- MarkdownTOC -->

- [train](#train_)
    - [5k       @ train](#5k___trai_n_)
    - [10k       @ train](#10k___trai_n_)
    - [10ka       @ train](#10ka___trai_n_)
        - [failed       @ 10ka/train](#failed___10ka_train_)
    - [40K_3       @ train](#40k_3___trai_n_)
    - [40K_3a       @ train](#40k_3a___trai_n_)
        - [rt       @ 40K_3a/train](#rt___40k_3a_train_)
    - [1K_3a       @ train](#1k_3a___trai_n_)
    - [1k3a_sampled       @ train](#1k3a_sampled___trai_n_)
    - [500_static3a       @ train](#500_static3a___trai_n_)
    - [1600_static3       @ train](#1600_static3___trai_n_)
    - [10k6_vid_entire_seq       @ train](#10k6_vid_entire_seq___trai_n_)
    - [20K6_5       @ train](#20k6_5___trai_n_)
        - [rt_100_per_img       @ 20K6_5/train](#rt_100_per_img___20k6_5_train_)
    - [20K6_60       @ train](#20k6_60___trai_n_)
    - [1k8_vid_entire_seq       @ train](#1k8_vid_entire_seq___trai_n_)
    - [1k8_vid_even_min_1       @ train](#1k8_vid_even_min_1___trai_n_)
- [tf_api_eval](#tf_api_eval_)
    - [500_static3a       @ tf_api_eval](#500_static3a___tf_api_eva_l_)
        - [class_agnostic       @ 500_static3a/tf_api_eval](#class_agnostic___500_static3a_tf_api_eval_)
        - [nms_thresh_0       @ 500_static3a/tf_api_eval](#nms_thresh_0___500_static3a_tf_api_eval_)
    - [1k3a_sampled       @ tf_api_eval](#1k3a_sampled___tf_api_eva_l_)
    - [20k6_5       @ tf_api_eval](#20k6_5___tf_api_eva_l_)
            - [inverted       @ 20k6_5/tf_api_eval](#inverted___20k6_5_tf_api_eval_)
                - [combine_sequences_416x416       @ inverted/20k6_5/tf_api_eval](#combine_sequences_416x416___inverted_20k6_5_tf_api_eva_l_)
            - [inverted_only_video       @ 20k6_5/tf_api_eval](#inverted_only_video___20k6_5_tf_api_eval_)
        - [acamp_1_per_class_6_class       @ 20k6_5/tf_api_eval](#acamp_1_per_class_6_class___20k6_5_tf_api_eval_)
    - [20k6_5_rt_100_per_img       @ tf_api_eval](#20k6_5_rt_100_per_img___tf_api_eva_l_)
    - [20k6_60       @ tf_api_eval](#20k6_60___tf_api_eva_l_)
    - [10k6_vid_entire_seq       @ tf_api_eval](#10k6_vid_entire_seq___tf_api_eva_l_)
        - [nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval](#nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_)
    - [1600_static_3       @ tf_api_eval](#1600_static_3___tf_api_eva_l_)
        - [on_train       @ 1600_static_3/tf_api_eval](#on_train___1600_static_3_tf_api_eva_l_)
            - [person_coco17_1600       @ on_train/1600_static_3/tf_api_eval](#person_coco17_1600___on_train_1600_static_3_tf_api_eval_)
        - [acamp_no_static_2_class       @ 1600_static_3/tf_api_eval](#acamp_no_static_2_class___1600_static_3_tf_api_eva_l_)
    - [40k3       @ tf_api_eval](#40k3___tf_api_eva_l_)
        - [on_train       @ 40k3/tf_api_eval](#on_train___40k3_tf_api_eval_)
            - [no_human       @ on_train/40k3/tf_api_eval](#no_human___on_train_40k3_tf_api_eva_l_)
        - [acamp_no_static_2_class       @ 40k3/tf_api_eval](#acamp_no_static_2_class___40k3_tf_api_eval_)
    - [40k3a       @ tf_api_eval](#40k3a___tf_api_eva_l_)
        - [on_train       @ 40k3a/tf_api_eval](#on_train___40k3a_tf_api_eva_l_)
    - [1k8_vid_entire_seq       @ tf_api_eval](#1k8_vid_entire_seq___tf_api_eva_l_)
    - [1k8_vid_even_min_1       @ tf_api_eval](#1k8_vid_even_min_1___tf_api_eva_l_)

<!-- /MarkdownTOC -->

protoc object_detection/protos/*.proto --python_out=.
cp -r models/research/slim ./

<a id="train_"></a>
# train

CUDA_VISIBLE_DEVICES=1 python3 train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/pipeline.config --train_dir=trained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28

<a id="5k___trai_n_"></a>
## 5k       @ train

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_5k.config --train_dir=trained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k

<a id="10k___trai_n_"></a>
## 10k       @ train

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_10k.config --train_dir=trained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_10k

<a id="10ka___trai_n_"></a>
## 10ka       @ train

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_10ka.config --train_dir=trained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_10ka

<a id="failed___10ka_train_"></a>
### failed       @ 10ka/train

CUDA_VISIBLE_DEVICES=1 python3 train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_5k.config --train_dir=trained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k


<a id="40k_3___trai_n_"></a>
## 40K_3       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_40k3.config --train_dir=trained/faster_rcnn_inception_resnet_v2_40k3 --n_steps=500000 --save_interval_secs=600

<a id="40k_3a___trai_n_"></a>
## 40K_3a       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_40k3a.config --train_dir=trained/faster_rcnn_inception_resnet_v2_40k3a --n_steps=500000 --save_interval_secs=600

<a id="rt___40k_3a_train_"></a>
### rt       @ 40K_3a/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_40k3a.config --train_dir=trained/faster_rcnn_inception_resnet_v2_40k3a_rt --n_steps=500000 --save_interval_secs=600

<a id="1k_3a___trai_n_"></a>
## 1K_3a       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1k3a.config --train_dir=trained/faster_rcnn_inception_resnet_v2_1k3a --n_steps=500000 --save_interval_secs=600

<a id="1k3a_sampled___trai_n_"></a>
## 1k3a_sampled       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1k3a_sampled.config --train_dir=trained/faster_rcnn_inception_resnet_v2_1k3a_sampled --n_steps=500000 --save_interval_secs=600

<a id="500_static3a___trai_n_"></a>
## 500_static3a       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_500_static3a.config --train_dir=trained/faster_rcnn_inception_resnet_v2_500_static3a --n_steps=500000 --save_interval_secs=600

<a id="1600_static3___trai_n_"></a>
## 1600_static3       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1600_static_3.config --train_dir=trained/faster_rcnn_inception_resnet_v2_1600_static_3 --n_steps=500000 --save_interval_secs=600


<a id="10k6_vid_entire_seq___trai_n_"></a>
## 10k6_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_10k6_vid_entire_seq.config --train_dir=trained/faster_rcnn_inception_resnet_v2_10k6_vid_entire_seq --n_steps=500000 --save_interval_secs=600

<a id="20k6_5___trai_n_"></a>
## 20K6_5       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_20k6_5.config --train_dir=trained/faster_rcnn_inception_resnet_v2_20k6_5 --n_steps=500000 --save_interval_secs=600

<a id="rt_100_per_img___20k6_5_train_"></a>
### rt_100_per_img       @ 20K6_5/train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_20k6_5_rt_100_per_img.config --train_dir=trained/faster_rcnn_inception_resnet_v2_20k6_5_rt_100_per_img --n_steps=500000 --save_interval_secs=600

<a id="20k6_60___trai_n_"></a>
## 20K6_60       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_20k6_60.config --train_dir=trained/faster_rcnn_inception_resnet_v2_20k6_60 --n_steps=500000 --save_interval_secs=600


<a id="1k8_vid_entire_seq___trai_n_"></a>
## 1k8_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1k8_vid_entire_seq.config --train_dir=trained/faster_rcnn_inception_resnet_v2_1k8_vid_entire_seq --n_steps=500000 --save_interval_secs=600

<a id="1k8_vid_even_min_1___trai_n_"></a>
## 1k8_vid_even_min_1       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1k8_vid_even_min_1.config --train_dir=trained/faster_rcnn_inception_resnet_v2_1k8_vid_even_min_1 --n_steps=500000 --save_interval_secs=600

<a id="tf_api_eval_"></a>
# tf_api_eval

<a id="500_static3a___tf_api_eva_l_"></a>
## 500_static3a       @ tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="class_agnostic___500_static3a_tf_api_eval_"></a>
### class_agnostic       @ 500_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1

<a id="nms_thresh_0___500_static3a_tf_api_eval_"></a>
### nms_thresh_0       @ 500_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0

<a id="1k3a_sampled___tf_api_eva_l_"></a>
## 1k3a_sampled       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_1k3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1k3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="20k6_5___tf_api_eva_l_"></a>
## 20k6_5       @ tf_api_eval

<a id="inverted___20k6_5_tf_api_eval_"></a>
#### inverted       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=2 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="combine_sequences_416x416___inverted_20k6_5_tf_api_eva_l_"></a>
##### combine_sequences_416x416       @ inverted/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=8 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 combine_sequences=1 input_size=416x416

<a id="inverted_only_video___20k6_5_tf_api_eval_"></a>
#### inverted_only_video       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/faster_rcnn_inception_resnet_v2_20k6_5/79958_on_inverted out_postfix=inv_only_vid load_det=1 score_thresholds=0:1:0.001

<a id="acamp_1_per_class_6_class___20k6_5_tf_api_eval_"></a>
### acamp_1_per_class_6_class       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=8 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_1_per_class_6_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_20k6_5.config sleep_time=10 write_summary=1 save_det=1  load_det=0 score_thresholds=0:1:0.01


<a id="20k6_5_rt_100_per_img___tf_api_eva_l_"></a>
## 20k6_5_rt_100_per_img       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_20k6_5_rt_100_per_img labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=2 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_20k6_5_rt_100_per_img.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="20k6_60___tf_api_eva_l_"></a>
## 20k6_60       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_20k6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_20k6_60.config --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp20k6_60_train inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0

<a id="10k6_vid_entire_seq___tf_api_eva_l_"></a>
## 10k6_vid_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_"></a>
### nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1 inference_dir=inference_nms_thresh_0 n_threads=8

<a id="1600_static_3___tf_api_eva_l_"></a>
## 1600_static_3       @ tf_api_eval

<a id="on_train___1600_static_3_tf_api_eva_l_"></a>
### on_train       @ 1600_static_3/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_1600_static_3 labels_path=data/wildlife_label_map_20k3.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp1600_static3_train.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1600_static_3.config sampling_ratio=1.0 random_sampling=0 start_seq_id=0 write_summary=1 save_det=1 load_det=0 

<a id="person_coco17_1600___on_train_1600_static_3_tf_api_eval_"></a>
#### person_coco17_1600       @ on_train/1600_static_3/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_1600_static_3 labels_path=data/wildlife_label_map_20k3.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=person_coco17_1600 pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1600_static_3.config sampling_ratio=1.0 random_sampling=0 start_seq_id=0 write_summary=1 save_det=1 load_det=1

<a id="acamp_no_static_2_class___1600_static_3_tf_api_eva_l_"></a>
### acamp_no_static_2_class       @ 1600_static_3/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_1600_static_3 labels_path=data/wildlife_label_map_20k2.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_no_static_2_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1600_static_3.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1 save_det=1

<a id="40k3___tf_api_eva_l_"></a>
## 40k3       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_40k3 labels_path=data/wildlife_label_map_20k3.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=10 root_dir=/data/acamp/acamp20k seq_paths=acamp40k3_test.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_40k3.config sampling_ratio=0.1 random_sampling=1 sleep_time=10 write_summary=1 save_det=1

<a id="on_train___40k3_tf_api_eval_"></a>
### on_train       @ 40k3/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_40k3 labels_path=data/wildlife_label_map_20k3.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp40k3_coco_train.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_40k3.config sampling_ratio=1.0 random_sampling=0 start_seq_id=0 write_summary=1 save_det=1 load_det=0

<a id="no_human___on_train_40k3_tf_api_eva_l_"></a>
#### no_human       @ on_train/40k3/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_40k3 labels_path=data/wildlife_label_map_20k3.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp40k3_coco_train_no_human.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_40k3.config sampling_ratio=1.0 random_sampling=0 start_seq_id=0 write_summary=1 save_det=1 load_det=0

<a id="acamp_no_static_2_class___40k3_tf_api_eval_"></a>
### acamp_no_static_2_class       @ 40k3/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_40k3 labels_path=data/wildlife_label_map_20k2.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_no_static_2_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_40k3.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=0 write_summary=1 save_det=1


<a id="40k3a___tf_api_eva_l_"></a>
## 40k3a       @ tf_api_eval

<a id="on_train___40k3a_tf_api_eva_l_"></a>
### on_train       @ 40k3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_40k3a_rt labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp40k3a_train.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_40k3a.config write_summary=1 save_det=1 out_postfix=train

<a id="1k8_vid_entire_seq___tf_api_eva_l_"></a>
## 1k8_vid_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_1k8_vid_entire_seq labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=6 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1k8_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1 start_seq_id=87

<a id="1k8_vid_even_min_1___tf_api_eva_l_"></a>
## 1k8_vid_even_min_1       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=6 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1 
