<!-- MarkdownTOC -->

- [train](#train_)
    - [v1       @ train](#v1___trai_n_)
        - [500_static3a       @ v1/train](#500_static3a___v1_train_)
        - [1K_3a_sampled       @ v1/train](#1k_3a_sampled___v1_train_)
        - [10k6_vid_entire_seq       @ v1/train](#10k6_vid_entire_seq___v1_train_)
        - [20K6_5       @ v1/train](#20k6_5___v1_train_)
    - [v2       @ train](#v2___trai_n_)
        - [500_static3a       @ v2/train](#500_static3a___v2_train_)
        - [1K_3a_sampled       @ v2/train](#1k_3a_sampled___v2_train_)
        - [20K_vid3a       @ v2/train](#20k_vid3a___v2_train_)
        - [10k6_vid_entire_seq       @ v2/train](#10k6_vid_entire_seq___v2_train_)
        - [20K6_5       @ v2/train](#20k6_5___v2_train_)
        - [1k8_vid_entire_seq       @ v2/train](#1k8_vid_entire_seq___v2_train_)
        - [1k8_vid_even_min_1       @ v2/train](#1k8_vid_even_min_1___v2_train_)
- [tf_api_eval](#tf_api_eval_)
    - [v1       @ tf_api_eval](#v1___tf_api_eva_l_)
        - [10k6_vid_entire_seq       @ v1/tf_api_eval](#10k6_vid_entire_seq___v1_tf_api_eval_)
        - [20k6_5       @ v1/tf_api_eval](#20k6_5___v1_tf_api_eval_)
    - [v2       @ tf_api_eval](#v2___tf_api_eva_l_)
        - [500_static3a       @ v2/tf_api_eval](#500_static3a___v2_tf_api_eval_)
            - [class_agnostic       @ 500_static3a/v2/tf_api_eval](#class_agnostic___500_static3a_v2_tf_api_eva_l_)
            - [nms_thresh_0       @ 500_static3a/v2/tf_api_eval](#nms_thresh_0___500_static3a_v2_tf_api_eva_l_)
        - [20K_vid3a       @ v2/tf_api_eval](#20k_vid3a___v2_tf_api_eval_)
        - [acamp200_static3a_inverted       @ v2/tf_api_eval](#acamp200_static3a_inverted___v2_tf_api_eval_)
        - [all_static       @ v2/tf_api_eval](#all_static___v2_tf_api_eval_)
        - [1K_3a_sampled       @ v2/tf_api_eval](#1k_3a_sampled___v2_tf_api_eval_)
            - [inverted       @ 1K_3a_sampled/v2/tf_api_eval](#inverted___1k_3a_sampled_v2_tf_api_eval_)
                - [map_plotting       @ inverted/1K_3a_sampled/v2/tf_api_eval](#map_plotting___inverted_1k_3a_sampled_v2_tf_api_eva_l_)
                - [win       @ inverted/1K_3a_sampled/v2/tf_api_eval](#win___inverted_1k_3a_sampled_v2_tf_api_eva_l_)
            - [all_vid       @ 1K_3a_sampled/v2/tf_api_eval](#all_vid___1k_3a_sampled_v2_tf_api_eval_)
        - [10k6_vid_entire_seq       @ v2/tf_api_eval](#10k6_vid_entire_seq___v2_tf_api_eval_)
            - [class_agnostic       @ 10k6_vid_entire_seq/v2/tf_api_eval](#class_agnostic___10k6_vid_entire_seq_v2_tf_api_eval_)
            - [nms_thresh_0       @ 10k6_vid_entire_seq/v2/tf_api_eval](#nms_thresh_0___10k6_vid_entire_seq_v2_tf_api_eval_)
        - [20k6_5       @ v2/tf_api_eval](#20k6_5___v2_tf_api_eval_)
            - [inverted       @ 20k6_5/v2/tf_api_eval](#inverted___20k6_5_v2_tf_api_eva_l_)
            - [combine_sequences_640x360       @ 20k6_5/v2/tf_api_eval](#combine_sequences_640x360___20k6_5_v2_tf_api_eva_l_)
            - [combine_sequences_416x416       @ 20k6_5/v2/tf_api_eval](#combine_sequences_416x416___20k6_5_v2_tf_api_eva_l_)
            - [inverted_only_video       @ 20k6_5/v2/tf_api_eval](#inverted_only_video___20k6_5_v2_tf_api_eva_l_)
                - [nms_thresh_0       @ inverted_only_video/20k6_5/v2/tf_api_eval](#nms_thresh_0___inverted_only_video_20k6_5_v2_tf_api_eva_l_)
        - [1k8_vid_entire_seq       @ v2/tf_api_eval](#1k8_vid_entire_seq___v2_tf_api_eval_)
        - [1k8_vid_even_min_1       @ v2/tf_api_eval](#1k8_vid_even_min_1___v2_tf_api_eval_)

<!-- /MarkdownTOC -->

<a id="train_"></a>
# train

<a id="v1___trai_n_"></a>
## v1       @ train

<a id="500_static3a___v1_train_"></a>
### 500_static3a       @ v1/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco_2018_01_28_500_static3a.config --train_dir=trained/ssd_mobilenet_v1_coco_2018_01_28_500_static3a --n_steps=600000 --save_interval_secs=600

<a id="1k_3a_sampled___v1_train_"></a>
### 1K_3a_sampled       @ v1/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco_2018_01_28_1K_3a_sampled.config --train_dir=trained/ssd_mobilenet_v1_coco_2018_01_28_1K_3a_sampled --n_steps=600000 --save_interval_secs=600

<a id="10k6_vid_entire_seq___v1_train_"></a>
### 10k6_vid_entire_seq       @ v1/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco_2018_01_28_10k6_vid_entire_seq.config --train_dir=trained/ssd_mobilenet_v1_coco_2018_01_28_10k6_vid_entire_seq --n_steps=600000 --save_interval_secs=600


<a id="20k6_5___v1_train_"></a>
### 20K6_5       @ v1/train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco_2018_01_28_20k6_5.config --train_dir=trained/ssd_mobilenet_v1_coco_2018_01_28_20k6_5 --n_steps=600000 --save_interval_secs=600

<a id="v2___trai_n_"></a>
## v2       @ train

<a id="500_static3a___v2_train_"></a>
### 500_static3a       @ v2/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_500_static3a.config --train_dir=trained/ssd_mobilenet_v2_500_static3a --n_steps=600000 --save_interval_secs=600

<a id="1k_3a_sampled___v2_train_"></a>
### 1K_3a_sampled       @ v2/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_1K_3a_sampled.config --train_dir=trained/ssd_mobilenet_v2_1K_3a_sampled --n_steps=600000 --save_interval_secs=600

<a id="20k_vid3a___v2_train_"></a>
### 20K_vid3a       @ v2/train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_20K_vid3a.config --train_dir=trained/ssd_mobilenet_v2_20K_vid3a --n_steps=200000 --save_interval_secs=600

<a id="10k6_vid_entire_seq___v2_train_"></a>
### 10k6_vid_entire_seq       @ v2/train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_10k6_vid_entire_seq.config --train_dir=trained/ssd_mobilenet_v2_10k6_vid_entire_seq --n_steps=600000 --save_interval_secs=600


<a id="20k6_5___v2_train_"></a>
### 20K6_5       @ v2/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_20k6_5.config --train_dir=trained/ssd_mobilenet_v2_20k6_5 --n_steps=600000 --save_interval_secs=600

<a id="1k8_vid_entire_seq___v2_train_"></a>
### 1k8_vid_entire_seq       @ v2/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_1k8_vid_entire_seq.config --train_dir=trained/ssd_mobilenet_v2_1k8_vid_entire_seq --n_steps=600000 --save_interval_secs=600

<a id="1k8_vid_even_min_1___v2_train_"></a>
### 1k8_vid_even_min_1       @ v2/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_1k8_vid_even_min_1.config --train_dir=trained/ssd_mobilenet_v2_1k8_vid_even_min_1 --n_steps=600000 --save_interval_secs=600

<a id="tf_api_eval_"></a>
# tf_api_eval

<a id="v1___tf_api_eva_l_"></a>
## v1       @ tf_api_eval

<a id="10k6_vid_entire_seq___v1_tf_api_eval_"></a>
### 10k6_vid_entire_seq       @ v1/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v1_coco_2018_01_28_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco_2018_01_28_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="20k6_5___v1_tf_api_eval_"></a>
### 20k6_5       @ v1/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v1_coco_2018_01_28_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco_2018_01_28_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="v2___tf_api_eva_l_"></a>
## v2       @ tf_api_eval

<a id="500_static3a___v2_tf_api_eval_"></a>
### 500_static3a       @ v2/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="class_agnostic___500_static3a_v2_tf_api_eva_l_"></a>
#### class_agnostic       @ 500_static3a/v2/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1

<a id="nms_thresh_0___500_static3a_v2_tf_api_eva_l_"></a>
#### nms_thresh_0       @ 500_static3a/v2/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0

<a id="20k_vid3a___v2_tf_api_eval_"></a>
### 20K_vid3a       @ v2/tf_api_eval

<a id="acamp200_static3a_inverted___v2_tf_api_eval_"></a>
### acamp200_static3a_inverted       @ v2/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_20K_vid3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_20K_vid3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp200_static3a_inverted load_det=1 load_dir=trained/ssd_mobilenet_v2_20K_vid3a/38575_on_acamp200_static3a_inverted score_thresholds=0:1:0.001

<a id="all_static___v2_tf_api_eval_"></a>
### all_static       @ v2/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_20K_vid3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_20K_vid3a.config sleep_time=10 write_summary=1 save_det=1 out_postfix=all_static score_thresholds=0:1:0.001


<a id="1k_3a_sampled___v2_tf_api_eval_"></a>
### 1K_3a_sampled       @ v2/tf_api_eval

<a id="inverted___1k_3a_sampled_v2_tf_api_eval_"></a>
#### inverted       @ 1K_3a_sampled/v2/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_1K_3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=32 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_1K_3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="map_plotting___inverted_1k_3a_sampled_v2_tf_api_eva_l_"></a>
##### map_plotting       @ inverted/1K_3a_sampled/v2/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_1K_3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_1K_3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=1 eval_every=0 load_dir=trained/ssd_mobilenet_v2_1K_3a_sampled/19507_on_inverted show_animation=0 draw_plot=0 results_path=results/trained_ssd_mobilenet_v2_1K_3a_sampled_19507_on_inverted score_thresholds=0:1:0.001


<a id="win___inverted_1k_3a_sampled_v2_tf_api_eva_l_"></a>
##### win       @ inverted/1K_3a_sampled/v2/tf_api_eval

python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_1K_3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_1K_3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=1 eval_every=0 load_dir=H:\UofA\PhD\Reports\190721_wacv_animal_detection_paper\results\retinanet\trained_ssd_mobilenet_v2_1K_3a_sampled_19507_on_inverted_grs_190615_101104 show_animation=0  draw_plot=1 results_path=results/trained_ssd_mobilenet_v2_1K_3a_sampled_19507_on_inverted_grs_190615_101104 score_thresholds=0.19

<a id="all_vid___1k_3a_sampled_v2_tf_api_eval_"></a>
#### all_vid       @ 1K_3a_sampled/v2/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_1K_3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=32 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_video_3_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_1K_3a_sampled.config sleep_time=10 write_summary=1 save_det=1 out_postfix=all_vid load_det=0 score_thresholds=0:1:0.001

<a id="10k6_vid_entire_seq___v2_tf_api_eval_"></a>
### 10k6_vid_entire_seq       @ v2/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="class_agnostic___10k6_vid_entire_seq_v2_tf_api_eval_"></a>
#### class_agnostic       @ 10k6_vid_entire_seq/v2/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1 allow_seq_skipping=1

<a id="nms_thresh_0___10k6_vid_entire_seq_v2_tf_api_eval_"></a>
#### nms_thresh_0       @ 10k6_vid_entire_seq/v2/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1 inference_dir=inference_nms_thresh_0

<a id="20k6_5___v2_tf_api_eval_"></a>
### 20k6_5       @ v2/tf_api_eval

<a id="inverted___20k6_5_v2_tf_api_eva_l_"></a>
#### inverted       @ 20k6_5/v2/tf_api_eval


CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_coco_2018_03_29_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ .config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="combine_sequences_640x360___20k6_5_v2_tf_api_eva_l_"></a>
#### combine_sequences_640x360       @ 20k6_5/v2/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_coco_2018_03_29_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=480 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001  combine_sequences=1 input_size=640x360

<a id="combine_sequences_416x416___20k6_5_v2_tf_api_eva_l_"></a>
#### combine_sequences_416x416       @ 20k6_5/v2/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_coco_2018_03_29_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=480 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001  combine_sequences=1 input_size=416x416

<a id="inverted_only_video___20k6_5_v2_tf_api_eva_l_"></a>
#### inverted_only_video       @ 20k6_5/v2/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_coco_2018_03_29_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/ssd_mobilenet_v2_coco_2018_03_29_20k6_5/213946_on_inverted out_postfix=inv_only_vid load_det=1 score_thresholds=0:1:0.001

<a id="nms_thresh_0___inverted_only_video_20k6_5_v2_tf_api_eva_l_"></a>
##### nms_thresh_0       @ inverted_only_video/20k6_5/v2/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_coco_2018_03_29_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/ssd_mobilenet_v2_coco_2018_03_29_20k6_5/213946_on_nms_thresh_0_inv_only_vid out_postfix=nms_thresh_0_inv_only_vid load_det=1 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0


<a id="1k8_vid_entire_seq___v2_tf_api_eval_"></a>
### 1k8_vid_entire_seq       @ v2/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_1k8_vid_entire_seq labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_1k8_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 allow_seq_skipping=1 n_threads=4

<a id="1k8_vid_even_min_1___v2_tf_api_eval_"></a>
### 1k8_vid_even_min_1       @ v2/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=128 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 allow_seq_skipping=1 n_threads=4
