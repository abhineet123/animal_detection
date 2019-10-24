<!-- MarkdownTOC -->

- [train](#train_)
    - [5k       @ train](#5k___trai_n_)
        - [eval       @ 5k/train](#eval___5k_train_)
    - [10k       @ train](#10k___trai_n_)
        - [eval       @ 10k/train](#eval___10k_trai_n_)
    - [10k_ar       @ train](#10k_ar___trai_n_)
    - [10ka       @ train](#10ka___trai_n_)
        - [eval       @ 10ka/train](#eval___10ka_train_)
    - [20k       @ train](#20k___trai_n_)
    - [20k_hs       @ train](#20k_hs___trai_n_)
    - [20k_coco       @ train](#20k_coco___trai_n_)
    - [20k3       @ train](#20k3___trai_n_)
    - [20k3_hs       @ train](#20k3_hs___trai_n_)
    - [20k3_hs       @ train](#20k3_hs___trai_n__1)
    - [20k4       @ train](#20k4___trai_n_)
    - [20K6_5       @ train](#20k6_5___trai_n_)
    - [20K6_60       @ train](#20k6_60___trai_n_)
    - [10k6_vid_entire_seq       @ train](#10k6_vid_entire_seq___trai_n_)
    - [1_per_seq_6_class_vid_67       @ train](#1_per_seq_6_class_vid_67___trai_n_)
    - [2_per_seq_6_class_vid_67       @ train](#2_per_seq_6_class_vid_67___trai_n_)
    - [5_per_seq_6_class_vid_67       @ train](#5_per_seq_6_class_vid_67___trai_n_)
    - [10_per_seq_6_class_vid_67       @ train](#10_per_seq_6_class_vid_67___trai_n_)
    - [20k7       @ train](#20k7___trai_n_)
    - [20k5       @ train](#20k5___trai_n_)
    - [20K6_5       @ train](#20k6_5___trai_n__1)
    - [25k3       @ train](#25k3___trai_n_)
    - [25k7       @ train](#25k7___trai_n_)
    - [25k5       @ train](#25k5___trai_n_)
    - [25k3       @ train](#25k3___trai_n__1)
    - [40k3_coco       @ train](#40k3_coco___trai_n_)
    - [1600_static3       @ train](#1600_static3___trai_n_)
    - [40k3a       @ train](#40k3a___trai_n_)
        - [rt       @ 40k3a/train](#rt___40k3a_trai_n_)
    - [1K_3a       @ train](#1k_3a___trai_n_)
    - [1K_3a_sampled       @ train](#1k_3a_sampled___trai_n_)
        - [mp       @ 1K_3a_sampled/train](#mp___1k_3a_sampled_trai_n_)
    - [500_static3a       @ train](#500_static3a___trai_n_)
    - [200_static3a       @ train](#200_static3a___trai_n_)
    - [20K_vid3a       @ train](#20k_vid3a___trai_n_)
    - [1k8_vid_entire_seq       @ train](#1k8_vid_entire_seq___trai_n_)
    - [1k8_vid_even_min_1       @ train](#1k8_vid_even_min_1___trai_n_)
    - [1k8_vid_entire_seq_entire_seq       @ train](#1k8_vid_entire_seq_entire_seq___trai_n_)
- [tf_api_eval](#tf_api_eval_)
    - [500_static3a       @ tf_api_eval](#500_static3a___tf_api_eva_l_)
        - [class_agnostic       @ 500_static3a/tf_api_eval](#class_agnostic___500_static3a_tf_api_eval_)
        - [nms_thresh_0       @ 500_static3a/tf_api_eval](#nms_thresh_0___500_static3a_tf_api_eval_)
    - [acamp1K_static3a_sampled       @ tf_api_eval](#acamp1k_static3a_sampled___tf_api_eva_l_)
        - [inverted       @ acamp1K_static3a_sampled/tf_api_eval](#inverted___acamp1k_static3a_sampled_tf_api_eval_)
            - [mp       @ inverted/acamp1K_static3a_sampled/tf_api_eval](#mp___inverted_acamp1k_static3a_sampled_tf_api_eva_l_)
        - [all_vid       @ acamp1K_static3a_sampled/tf_api_eval](#all_vid___acamp1k_static3a_sampled_tf_api_eval_)
    - [acamp200K_static3a       @ tf_api_eval](#acamp200k_static3a___tf_api_eva_l_)
    - [20K_vid3a       @ tf_api_eval](#20k_vid3a___tf_api_eva_l_)
        - [acamp200_static3a_inverted       @ 20K_vid3a/tf_api_eval](#acamp200_static3a_inverted___20k_vid3a_tf_api_eva_l_)
        - [all_static       @ 20K_vid3a/tf_api_eval](#all_static___20k_vid3a_tf_api_eva_l_)
    - [10k6_vid_entire_seq       @ tf_api_eval](#10k6_vid_entire_seq___tf_api_eva_l_)
        - [nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval](#nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_)
    - [20k6_5       @ tf_api_eval](#20k6_5___tf_api_eva_l_)
        - [inverted       @ 20k6_5/tf_api_eval](#inverted___20k6_5_tf_api_eval_)
            - [nms_thresh_0       @ inverted/20k6_5/tf_api_eval](#nms_thresh_0___inverted_20k6_5_tf_api_eva_l_)
        - [inverted_only_video       @ 20k6_5/tf_api_eval](#inverted_only_video___20k6_5_tf_api_eval_)
            - [nms_thresh_0       @ inverted_only_video/20k6_5/tf_api_eval](#nms_thresh_0___inverted_only_video_20k6_5_tf_api_eval_)
        - [combine_sequences_416x416       @ 20k6_5/tf_api_eval](#combine_sequences_416x416___20k6_5_tf_api_eval_)
        - [frozen_graph_path       @ 20k6_5/tf_api_eval](#frozen_graph_path___20k6_5_tf_api_eval_)
    - [20k6_60       @ tf_api_eval](#20k6_60___tf_api_eva_l_)
        - [acamp_1_per_class_6_class       @ 20k6_60/tf_api_eval](#acamp_1_per_class_6_class___20k6_60_tf_api_eva_l_)
    - [1_per_seq_6_class_vid_67       @ tf_api_eval](#1_per_seq_6_class_vid_67___tf_api_eva_l_)
        - [inverted       @ 1_per_seq_6_class_vid_67/tf_api_eval](#inverted___1_per_seq_6_class_vid_67_tf_api_eval_)
        - [acamp_all_6_class_video_67       @ 1_per_seq_6_class_vid_67/tf_api_eval](#acamp_all_6_class_video_67___1_per_seq_6_class_vid_67_tf_api_eval_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 1_per_seq_6_class_vid_67/tf_api_eval](#1_2_5_10_per_seq_6_class_vid_67_inverted___1_per_seq_6_class_vid_67_tf_api_eval_)
    - [2_per_seq_6_class_vid_67       @ tf_api_eval](#2_per_seq_6_class_vid_67___tf_api_eva_l_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 2_per_seq_6_class_vid_67/tf_api_eval](#1_2_5_10_per_seq_6_class_vid_67_inverted___2_per_seq_6_class_vid_67_tf_api_eval_)
    - [5_per_seq_6_class_vid_67       @ tf_api_eval](#5_per_seq_6_class_vid_67___tf_api_eva_l_)
        - [inverted       @ 5_per_seq_6_class_vid_67/tf_api_eval](#inverted___5_per_seq_6_class_vid_67_tf_api_eval_)
        - [acamp_all_6_class_video_67       @ 5_per_seq_6_class_vid_67/tf_api_eval](#acamp_all_6_class_video_67___5_per_seq_6_class_vid_67_tf_api_eval_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 5_per_seq_6_class_vid_67/tf_api_eval](#1_2_5_10_per_seq_6_class_vid_67_inverted___5_per_seq_6_class_vid_67_tf_api_eval_)
    - [10_per_seq_6_class_vid_67       @ tf_api_eval](#10_per_seq_6_class_vid_67___tf_api_eva_l_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 10_per_seq_6_class_vid_67/tf_api_eval](#1_2_5_10_per_seq_6_class_vid_67_inverted___10_per_seq_6_class_vid_67_tf_api_eva_l_)
    - [40k3a_rt       @ tf_api_eval](#40k3a_rt___tf_api_eva_l_)
        - [on_train       @ 40k3a_rt/tf_api_eval](#on_train___40k3a_rt_tf_api_eval_)
    - [1k8_vid_entire_seq       @ tf_api_eval](#1k8_vid_entire_seq___tf_api_eva_l_)
        - [bison       @ 1k8_vid_entire_seq/tf_api_eval](#bison___1k8_vid_entire_seq_tf_api_eval_)
    - [1k8_vid_even_min_1       @ tf_api_eval](#1k8_vid_even_min_1___tf_api_eva_l_)

<!-- /MarkdownTOC -->

<a id="train_"></a>
# train

CUDA_VISIBLE_DEVICES=1 python3 train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/pipeline.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/pipeline.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28

<a id="5k___trai_n_"></a>
## 5k       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_5k.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_5k

<a id="eval___5k_train_"></a>
### eval       @ 5k/train

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/eval.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_5k.config--checkpoint_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_5k --eval_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_5k/eval --num_examples=0


<a id="10k___trai_n_"></a>
## 10k       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_10k

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/batch_2/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k.config --train_dir=trained/batch_2/faster_rcnn_resnet101_coco_2018_01_28_10k

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/batch_6/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k.config --train_dir=trained/batch_6/faster_rcnn_resnet101_coco_2018_01_28_10k

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_10k

<a id="eval___10k_trai_n_"></a>
### eval       @ 10k/train

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/eval.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k.config --checkpoint_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_10k --eval_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_10k/eval --num_examples=0

<a id="10k_ar___trai_n_"></a>
## 10k_ar       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/batch_6/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k_ar.config --train_dir=trained/batch_6/faster_rcnn_resnet101_coco_2018_01_28_10k_ar

<a id="10ka___trai_n_"></a>
## 10ka       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10ka.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_10ka

<a id="eval___10ka_train_"></a>
### eval       @ 10ka/train

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/eval.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k.config --checkpoint_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_10k --eval_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_10k/eval --num_examples=0

<a id="20k___trai_n_"></a>
## 20k       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_20k

<a id="20k_hs___trai_n_"></a>
## 20k_hs       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k_hs.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_20k_hs

<a id="20k_coco___trai_n_"></a>
## 20k_coco       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k_coco.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_20k_coco

<a id="20k3___trai_n_"></a>
## 20k3       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k3.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3

faster_rcnn_resnet101_coco_2018_01_28_20k3_grs_201807111627.zip

<a id="20k3_hs___trai_n_"></a>
## 20k3_hs       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k3_hs.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs

faster_rcnn_resnet101_coco_2018_01_28_20k3_hs_grs_201807111629.zip


<a id="20k3_hs___trai_n__1"></a>
## 20k3_hs       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k3_hs.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs

<a id="20k4___trai_n_"></a>
## 20k4       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k4.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_20k4

<a id="20k6_5___trai_n_"></a>
## 20K6_5       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k6_5.config --train_dir=trained/faster_rcnn_resnet101_20k6_5 --n_steps=600000 --save_interval_secs=600 --reset_train=0

<a id="20k6_60___trai_n_"></a>
## 20K6_60       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k6_60.config --train_dir=trained/faster_rcnn_resnet101_20k6_60 --n_steps=600000 --save_interval_secs=600 --reset_train=0

<a id="10k6_vid_entire_seq___trai_n_"></a>
## 10k6_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k6_vid_entire_seq.config --train_dir=trained/faster_rcnn_resnet101_10k6_vid_entire_seq --n_steps=600000 --save_interval_secs=600 --reset_train=0


<a id="1_per_seq_6_class_vid_67___trai_n_"></a>
## 1_per_seq_6_class_vid_67       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1_per_seq_6_class_vid_67.config --train_dir=trained/faster_rcnn_resnet101_1_per_seq_6_class_vid_67 --n_steps=100000 --save_interval_secs=600

<a id="2_per_seq_6_class_vid_67___trai_n_"></a>
## 2_per_seq_6_class_vid_67       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_2_per_seq_6_class_vid_67.config --train_dir=trained/faster_rcnn_resnet101_2_per_seq_6_class_vid_67 --n_steps=100000 --save_interval_secs=600

<a id="5_per_seq_6_class_vid_67___trai_n_"></a>
## 5_per_seq_6_class_vid_67       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_5_per_seq_6_class_vid_67.config --train_dir=trained/faster_rcnn_resnet101_5_per_seq_6_class_vid_67 --n_steps=100000 --save_interval_secs=600

<a id="10_per_seq_6_class_vid_67___trai_n_"></a>
## 10_per_seq_6_class_vid_67       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10_per_seq_6_class_vid_67.config --train_dir=trained/faster_rcnn_resnet101_10_per_seq_6_class_vid_67 --n_steps=100000 --save_interval_secs=600


<a id="20k7___trai_n_"></a>
## 20k7       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k7.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_20k7

<a id="20k5___trai_n_"></a>
## 20k5       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k5.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_20k5

<a id="20k6_5___trai_n__1"></a>
## 20K6_5       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k6_5.config --train_dir=trained/faster_rcnn_resnet101_20k6_5 --n_steps=1000000 --save_interval_secs=600

<a id="25k3___trai_n_"></a>
## 25k3       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_25k3.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_25k3

<a id="25k7___trai_n_"></a>
## 25k7       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_25k7.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_25k7

<a id="25k5___trai_n_"></a>
## 25k5       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_25k5.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_25k5

<a id="25k3___trai_n__1"></a>
## 25k3       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_25k3.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_25k3

<a id="40k3_coco___trai_n_"></a>
## 40k3_coco       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_40k3_coco.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_40k3_coco

<a id="1600_static3___trai_n_"></a>
## 1600_static3       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1600_static3_train.config --train_dir=trained/faster_rcnn_resnet101_coco_2018_01_28_1600_static3_train

<a id="40k3a___trai_n_"></a>
## 40k3a       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_40k3a.config --train_dir=trained/faster_rcnn_resnet101_40k3a

<a id="rt___40k3a_trai_n_"></a>
### rt       @ 40k3a/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_40k3a.config --train_dir=trained/faster_rcnn_resnet101_40k3a_rt

<a id="1k_3a___trai_n_"></a>
## 1K_3a       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k3a.config --train_dir=trained/faster_rcnn_resnet101_1k3a --n_steps=500000 --save_interval_secs=600

<a id="1k_3a_sampled___trai_n_"></a>
## 1K_3a_sampled       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k3a_sampled.config --train_dir=trained/faster_rcnn_resnet101_1k3a_sampled --n_steps=500000 --save_interval_secs=600

<a id="mp___1k_3a_sampled_trai_n_"></a>
### mp       @ 1K_3a_sampled/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k3a_sampled.config --train_dir=trained/faster_rcnn_resnet101_1k3a_sampled_mp --n_steps=500000 --save_interval_secs=600 --enable_mixed_precision=1

<a id="500_static3a___trai_n_"></a>
## 500_static3a       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_500_static3a.config --train_dir=trained/faster_rcnn_resnet101_500_static3a --n_steps=500000 --save_interval_secs=600

<a id="200_static3a___trai_n_"></a>
## 200_static3a       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_200_static3a.config --train_dir=trained/faster_rcnn_resnet101_200_static3a --n_steps=500000 --save_interval_secs=600

<a id="20k_vid3a___trai_n_"></a>
## 20K_vid3a       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20K_vid3a.config --train_dir=trained/faster_rcnn_resnet101_20K_vid3a --n_steps=200000 --save_interval_secs=600

<a id="1k8_vid_entire_seq___trai_n_"></a>
## 1k8_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k8_vid_entire_seq.config --train_dir=trained/faster_rcnn_resnet101_1k8_vid_entire_seq --n_steps=500000 --save_interval_secs=600

<a id="1k8_vid_even_min_1___trai_n_"></a>
## 1k8_vid_even_min_1       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k8_vid_even_min_1.config --train_dir=trained/faster_rcnn_resnet101_1k8_vid_even_min_1 --n_steps=500000 --save_interval_secs=600

<a id="1k8_vid_entire_seq_entire_seq___trai_n_"></a>
## 1k8_vid_entire_seq_entire_seq       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k8_vid_entire_seq_entire_seq.config --train_dir=trained/faster_rcnn_resnet101_1k8_vid_entire_seq_entire_seq --n_steps=500000 --save_interval_secs=600

tensorboard --logdir=trained
<a id="http12700116006"></a>
http://127.0.0.1:16006

<a id="tf_api_eval_"></a>
# tf_api_eval

<a id="500_static3a___tf_api_eva_l_"></a>
## 500_static3a       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="class_agnostic___500_static3a_tf_api_eval_"></a>
### class_agnostic       @ 500_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1

<a id="nms_thresh_0___500_static3a_tf_api_eval_"></a>
### nms_thresh_0       @ 500_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0

<a id="acamp1k_static3a_sampled___tf_api_eva_l_"></a>
## acamp1K_static3a_sampled       @ tf_api_eval

<a id="inverted___acamp1k_static3a_sampled_tf_api_eval_"></a>
### inverted       @ acamp1K_static3a_sampled/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_1k3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="mp___inverted_acamp1k_static3a_sampled_tf_api_eva_l_"></a>
#### mp       @ inverted/acamp1K_static3a_sampled/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_1k3a_sampled_mp labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="all_vid___acamp1k_static3a_sampled_tf_api_eval_"></a>
### all_vid       @ acamp1K_static3a_sampled/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_1k3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=32 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_video_3_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k3a_sampled.config sleep_time=10 write_summary=1 save_det=1 out_postfix=all_vid load_det=0 score_thresholds=0:1:0.001


<a id="acamp200k_static3a___tf_api_eva_l_"></a>
## acamp200K_static3a       @ tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_200_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_200_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="20k_vid3a___tf_api_eva_l_"></a>
## 20K_vid3a       @ tf_api_eval

<a id="acamp200_static3a_inverted___20k_vid3a_tf_api_eva_l_"></a>
### acamp200_static3a_inverted       @ 20K_vid3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_20K_vid3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20K_vid3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp200_static3a_inverted load_det=0 score_thresholds=0:1:0.001

<a id="all_static___20k_vid3a_tf_api_eva_l_"></a>
### all_static       @ 20K_vid3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_20K_vid3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20K_vid3a.config sleep_time=10 write_summary=1 save_det=1 out_postfix=all_static load_det=0 score_thresholds=0:1:0.001

<a id="10k6_vid_entire_seq___tf_api_eva_l_"></a>
## 10k6_vid_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=2 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_"></a>
### nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=2 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1 inference_dir=inference_nms_thresh_0 n_threads=8

<a id="20k6_5___tf_api_eva_l_"></a>
## 20k6_5       @ tf_api_eval

<a id="inverted___20k6_5_tf_api_eval_"></a>
### inverted       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001

<a id="nms_thresh_0___inverted_20k6_5_tf_api_eva_l_"></a>
#### nms_thresh_0       @ inverted/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted_nms_thresh_0 load_det=0 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0

<a id="inverted_only_video___20k6_5_tf_api_eval_"></a>
### inverted_only_video       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/faster_rcnn_resnet101_20k6_5/94945_on_inverted out_postfix=inv_only_vid load_det=1 score_thresholds=0:1:0.001

<a id="nms_thresh_0___inverted_only_video_20k6_5_tf_api_eval_"></a>
#### nms_thresh_0       @ inverted_only_video/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/faster_rcnn_resnet101_20k6_5/94945_on_inverted_nms_thresh_0 out_postfix=nms_thresh_0_inv_only_vid load_det=1 score_thresholds=0:1:0.001

<a id="combine_sequences_416x416___20k6_5_tf_api_eval_"></a>
### combine_sequences_416x416       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 combine_sequences=1 input_size=416x416

<a id="frozen_graph_path___20k6_5_tf_api_eval_"></a>
### frozen_graph_path       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 frozen_graph_path=trained/faster_rcnn_resnet101_20k6_5/inference_94945/frozen_inference_graph.pb

<a id="20k6_60___tf_api_eva_l_"></a>
## 20k6_60       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_20k6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=36 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k6_60.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_train inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 frozen_graph_path=trained/faster_rcnn_resnet101_20k6_60/inference_600000/frozen_inference_graph.pb trained_checkpoint_prefix=600000 score_thresholds=0:1:0.1

<a id="acamp_1_per_class_6_class___20k6_60_tf_api_eva_l_"></a>
### acamp_1_per_class_6_class       @ 20k6_60/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_20k6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=36 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_1_per_class_6_class.txt pipeline_config_path=configs/faster_rcnn_resne_coco_2018_01_28/faster_rcnn_resnet101_20k6_60.config sleep_time=10 write_summary=1 save_det=1 frozen_graph_path=trained/faster_rcnn_resnet101_20k6_60/inference_600000/frozen_inference_graph.pb trained_checkpoint_prefix=600000


<a id="1_per_seq_6_class_vid_67___tf_api_eva_l_"></a>
## 1_per_seq_6_class_vid_67       @ tf_api_eval

<a id="inverted___1_per_seq_6_class_vid_67_tf_api_eval_"></a>
### inverted       @ 1_per_seq_6_class_vid_67/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_1_per_seq_6_class_vid_67 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=8 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1_per_seq_6_class_vid_67.config  load_samples=1 load_samples_root=/data/acamp/acamp20k/1_per_seq_6_class_vid_67 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 

<a id="acamp_all_6_class_video_67___1_per_seq_6_class_vid_67_tf_api_eval_"></a>
### acamp_all_6_class_video_67       @ 1_per_seq_6_class_vid_67/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_1_per_seq_6_class_vid_67 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1_per_seq_6_class_vid_67.config  sleep_time=10 write_summary=1 save_det=1 load_det=0 

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___1_per_seq_6_class_vid_67_tf_api_eval_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 1_per_seq_6_class_vid_67/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_1_per_seq_6_class_vid_67 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1_per_seq_6_class_vid_67.config sleep_time=10 write_summary=1 save_det=1 load_det=0 load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted out_postfix=1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="2_per_seq_6_class_vid_67___tf_api_eva_l_"></a>
## 2_per_seq_6_class_vid_67       @ tf_api_eval

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___2_per_seq_6_class_vid_67_tf_api_eval_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 2_per_seq_6_class_vid_67/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_2_per_seq_6_class_vid_67 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_2_per_seq_6_class_vid_67.config sleep_time=10 write_summary=1 save_det=1 load_det=0 load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted out_postfix=1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="5_per_seq_6_class_vid_67___tf_api_eva_l_"></a>
## 5_per_seq_6_class_vid_67       @ tf_api_eval

<a id="inverted___5_per_seq_6_class_vid_67_tf_api_eval_"></a>
### inverted       @ 5_per_seq_6_class_vid_67/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_5_per_seq_6_class_vid_67 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_5_per_seq_6_class_vid_67.config  load_samples=1 load_samples_root=/data/acamp/acamp20k/5_per_seq_6_class_vid_67 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 

<a id="acamp_all_6_class_video_67___5_per_seq_6_class_vid_67_tf_api_eval_"></a>
### acamp_all_6_class_video_67       @ 5_per_seq_6_class_vid_67/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_5_per_seq_6_class_vid_67 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_5_per_seq_6_class_vid_67.config  sleep_time=10 write_summary=1 save_det=1 load_det=0 

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___5_per_seq_6_class_vid_67_tf_api_eval_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 5_per_seq_6_class_vid_67/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_5_per_seq_6_class_vid_67 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_5_per_seq_6_class_vid_67.config sleep_time=10 write_summary=1 save_det=1 load_det=0 load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted out_postfix=1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="10_per_seq_6_class_vid_67___tf_api_eva_l_"></a>
## 10_per_seq_6_class_vid_67       @ tf_api_eval

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___10_per_seq_6_class_vid_67_tf_api_eva_l_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 10_per_seq_6_class_vid_67/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_10_per_seq_6_class_vid_67 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10_per_seq_6_class_vid_67.config sleep_time=10 write_summary=1 save_det=1 load_det=0 load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted out_postfix=1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001


<a id="40k3a_rt___tf_api_eva_l_"></a>
## 40k3a_rt       @ tf_api_eval

<a id="on_train___40k3a_rt_tf_api_eval_"></a>
### on_train       @ 40k3a_rt/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_40k3a_rt labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp40k3a_train.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_40k3a.config write_summary=1 save_det=1 out_postfix=train

<a id="1k8_vid_entire_seq___tf_api_eva_l_"></a>
## 1k8_vid_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_1k8_vid_entire_seq labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=36 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k8_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 draw_plot=0 allow_seq_skipping=1 n_threads=0

<a id="bison___1k8_vid_entire_seq_tf_api_eval_"></a>
### bison       @ 1k8_vid_entire_seq/tf_api_eval

python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_1k8_vid_entire_seq labels_path=data/wildlife_label_map_bison.pbtxt n_frames=0 batch_size=36 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_bison.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k8_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 draw_plot=1

<a id="1k8_vid_even_min_1___tf_api_eva_l_"></a>
## 1k8_vid_even_min_1       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 allow_seq_skipping=1







