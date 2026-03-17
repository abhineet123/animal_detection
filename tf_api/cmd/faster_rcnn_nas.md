<!-- MarkdownTOC -->

- [train](#train_)
    - [5k       @ train](#5k___trai_n_)
    - [1k3a_sampled       @ train](#1k3a_sampled___trai_n_)
    - [500_static3a       @ train](#500_static3a___trai_n_)
    - [10k6_vid_entire_seq       @ train](#10k6_vid_entire_seq___trai_n_)
    - [1k8_vid_entire_seq       @ train](#1k8_vid_entire_seq___trai_n_)
    - [1k8_vid_even_min_1       @ train](#1k8_vid_even_min_1___trai_n_)
    - [20K6_5       @ train](#20k6_5___trai_n_)
    - [20k6_60       @ train](#20k6_60___trai_n_)
- [tf_api_eval](#tf_api_eval_)
    - [500_static3a       @ tf_api_eval](#500_static3a___tf_api_eva_l_)
        - [class_agnostic       @ 500_static3a/tf_api_eval](#class_agnostic___500_static3a_tf_api_eval_)
        - [nms_thresh_0       @ 500_static3a/tf_api_eval](#nms_thresh_0___500_static3a_tf_api_eval_)
    - [1k3a_sampled       @ tf_api_eval](#1k3a_sampled___tf_api_eva_l_)
        - [inverted       @ 1k3a_sampled/tf_api_eval](#inverted___1k3a_sampled_tf_api_eval_)
    - [10k6_vid_entire_seq       @ tf_api_eval](#10k6_vid_entire_seq___tf_api_eva_l_)
        - [nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval](#nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_)
    - [20k6_5       @ tf_api_eval](#20k6_5___tf_api_eva_l_)
            - [inverted       @ 20k6_5/tf_api_eval](#inverted___20k6_5_tf_api_eval_)
            - [inverted_only_video       @ 20k6_5/tf_api_eval](#inverted_only_video___20k6_5_tf_api_eval_)
                - [nms_thresh_0       @ inverted_only_video/20k6_5/tf_api_eval](#nms_thresh_0___inverted_only_video_20k6_5_tf_api_eval_)
        - [combine_sequences_416x416       @ 20k6_5/tf_api_eval](#combine_sequences_416x416___20k6_5_tf_api_eval_)
    - [20k6_60       @ tf_api_eval](#20k6_60___tf_api_eva_l_)
        - [acamp_1_per_class_6_class       @ 20k6_60/tf_api_eval](#acamp_1_per_class_6_class___20k6_60_tf_api_eva_l_)
    - [1k8_vid_entire_seq       @ tf_api_eval](#1k8_vid_entire_seq___tf_api_eva_l_)
    - [1k8_vid_even_min_1       @ tf_api_eval](#1k8_vid_even_min_1___tf_api_eva_l_)
- [eval](#eva_l_)
- [export_inference_graph](#export_inference_grap_h_)
    - [5K       @ export_inference_graph](#5k___export_inference_graph_)
- [test](#tes_t_)
- [100 on 5k](#100_on_5k_)
    - [5K       @ 100_on_5k](#5k___100_on_5_k_)
        - [grizzly_bear       @ 5K/100_on_5k](#grizzly_bear___5k_100_on_5k_)
            - [batch_tests       @ grizzly_bear/5K/100_on_5k](#batch_tests___grizzly_bear_5k_100_on_5_k_)
            - [more_bear_videos       @ grizzly_bear/5K/100_on_5k](#more_bear_videos___grizzly_bear_5k_100_on_5_k_)
                - [zip       @ more_bear_videos/grizzly_bear/5K/100_on_5k](#zip___more_bear_videos_grizzly_bear_5k_100_on_5k_)
        - [moose       @ 5K/100_on_5k](#moose___5k_100_on_5k_)
        - [deer       @ 5K/100_on_5k](#deer___5k_100_on_5k_)
        - [coyote       @ 5K/100_on_5k](#coyote___5k_100_on_5k_)
- [evaluation](#evaluatio_n_)
- [ots inference](#ots_inferenc_e_)
        - [marcin_180608       @ ots_inference/](#marcin_180608___ots_inference_)
            - [bear       @ marcin_180608/ots_inference/](#bear___marcin_180608_ots_inference_)
            - [moose       @ marcin_180608/ots_inference/](#moose___marcin_180608_ots_inference_)
            - [coyote       @ marcin_180608/ots_inference/](#coyote___marcin_180608_ots_inference_)
- [deer](#dee_r_)

<!-- /MarkdownTOC -->

<a id="train_"></a>
# train

CUDA_VISIBLE_DEVICES=1 python3 train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/pipeline.config --train_dir=trained/faster_rcnn_nas_coco_2018_01_28

`AttributeError: module 'nets.nasnet.nasnet' has no attribute '_large_imagenet_config'`

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/pipeline.config --train_dir=trained/faster_rcnn_nas_coco_2018_01_28

<a id="5k___trai_n_"></a>
## 5k       @ train

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_5k.config --train_dir=trained/faster_rcnn_nas_coco_2018_01_28_5k

<a id="1k3a_sampled___trai_n_"></a>
## 1k3a_sampled       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_1k3a_sampled.config --train_dir=trained/faster_rcnn_nas_1k3a_sampled --n_steps=100000 --save_interval_secs=600

<a id="500_static3a___trai_n_"></a>
## 500_static3a       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_500_static3a.config --train_dir=trained/faster_rcnn_nas_500_static3a --n_steps=100000 --save_interval_secs=600

<a id="10k6_vid_entire_seq___trai_n_"></a>
## 10k6_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_10k6_vid_entire_seq.config --train_dir=trained/faster_rcnn_nas_10k6_vid_entire_seq --n_steps=100000 --save_interval_secs=600

<a id="1k8_vid_entire_seq___trai_n_"></a>
## 1k8_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_1k8_vid_entire_seq.config --train_dir=trained/faster_rcnn_nas_1k8_vid_entire_seq --n_steps=100000 --save_interval_secs=600


<a id="1k8_vid_even_min_1___trai_n_"></a>
## 1k8_vid_even_min_1       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_1k8_vid_even_min_1.config --train_dir=trained/faster_rcnn_nas_1k8_vid_even_min_1 --n_steps=100000 --save_interval_secs=600


<a id="20k6_5___trai_n_"></a>
## 20K6_5       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_20k6_5.config --train_dir=trained/faster_rcnn_nas_20k6_5 --n_steps=100000 --save_interval_secs=600

<a id="20k6_60___trai_n_"></a>
## 20k6_60       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_20k6_60.config --train_dir=trained/faster_rcnn_nas_20k6_60 --n_steps=100000 --save_interval_secs=600

<a id="tf_api_eval_"></a>
# tf_api_eval

<a id="500_static3a___tf_api_eva_l_"></a>
## 500_static3a       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="class_agnostic___500_static3a_tf_api_eval_"></a>
### class_agnostic       @ 500_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1

<a id="nms_thresh_0___500_static3a_tf_api_eval_"></a>
### nms_thresh_0       @ 500_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0 n_threads=8

<a id="1k3a_sampled___tf_api_eva_l_"></a>
## 1k3a_sampled       @ tf_api_eval

<a id="inverted___1k3a_sampled_tf_api_eval_"></a>
### inverted       @ 1k3a_sampled/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_1k3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_1k3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="10k6_vid_entire_seq___tf_api_eva_l_"></a>
## 10k6_vid_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=2 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_"></a>
### nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=2 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1 inference_dir=inference_nms_thresh_0 n_threads=8

<a id="20k6_5___tf_api_eva_l_"></a>
## 20k6_5       @ tf_api_eval

<a id="inverted___20k6_5_tf_api_eval_"></a>
#### inverted       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 

<a id="inverted_only_video___20k6_5_tf_api_eval_"></a>
#### inverted_only_video       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/faster_rcnn_nas_20k6_5/100000_on_inverted out_postfix=inv_only_vid load_det=1 score_thresholds=0:1:0.001

<a id="nms_thresh_0___inverted_only_video_20k6_5_tf_api_eval_"></a>
##### nms_thresh_0       @ inverted_only_video/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/faster_rcnn_nas_20k6_5/100000_on_inverted out_postfix=nms_thresh_0_inv_only_vid load_det=0 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0

<a id="combine_sequences_416x416___20k6_5_tf_api_eval_"></a>
### combine_sequences_416x416       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 combine_sequences=1 input_size=416x416


<a id="20k6_60___tf_api_eva_l_"></a>
## 20k6_60       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_20k6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=2 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_20k6_60.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_train inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0

<a id="acamp_1_per_class_6_class___20k6_60_tf_api_eva_l_"></a>
### acamp_1_per_class_6_class       @ 20k6_60/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_20k6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=2 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_1_per_class_6_class.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_20k6_60.config  sleep_time=10 write_summary=1 save_det=1 load_det=0

<a id="1k8_vid_entire_seq___tf_api_eva_l_"></a>
## 1k8_vid_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_1k8_vid_entire_seq labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=3 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_1k8_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1 

<a id="1k8_vid_even_min_1___tf_api_eva_l_"></a>
## 1k8_vid_even_min_1       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=3 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1


<a id="eva_l_"></a>
# eval

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/eval.py --logtostderr --pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_5k.config --checkpoint_dir=trained/faster_rcnn_nas_coco_2018_01_28_5k --eval_dir=trained/faster_rcnn_nas_coco_2018_01_28_5k/eval --num_examples=0

<a id="export_inference_grap_h_"></a>
# export_inference_graph
    
CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_nas_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix trained/faster_rcnn_nas_coco_2018_01_28/model.ckpt-200000 --output_directory trained/faster_rcnn_nas_coco_2018_01_28/inference

<a id="5k___export_inference_graph_"></a>
## 5K       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_5k.config --trained_checkpoint_prefix trained/faster_rcnn_nas_coco_2018_01_28_5k/model.ckpt-200000 --output_directory trained/faster_rcnn_nas_coco_2018_01_28_5k/inference_200K


<a id="tes_t_"></a>
# test

<a id="100_on_5k_"></a>
# 100 on 5k

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt root_dir=/data/acamp/acamp5k/test/images save_dir=results/faster_rcnn_nas_coco_2018_01_28_100_on_5k/ n_frames=0 batch_size=3


CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt root_dir=/data/acamp/acamp5k/test/images list_file_name=acamp5k_test_19_25.txt save_dir=results/faster_rcnn_nas_coco_2018_01_28_100_on_5k/ n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt root_dir=/data/acamp/acamp5k/test/images list_file_name=acamp5k_test_25_31.txt save_dir=results/faster_rcnn_nas_coco_2018_01_28_100_on_5k/ n_frames=0 batch_size=3

<a id="5k___100_on_5_k_"></a>
## 5K       @ 100_on_5k

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28_5k/inference_200K/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt root_dir=/data/acamp/acamp5k/test/images save_dir=results/faster_rcnn_nas_coco_2018_01_28_5k/inference_200K n_frames=0 batch_size=2

zr faster_rcnn_nas_coco_2018_01_28_5k_inference_200K faster_rcnn_nas_coco_2018_01_28_5k/inference_200K

<a id="grizzly_bear___5k_100_on_5k_"></a>
### grizzly_bear       @ 5K/100_on_5k

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det.mkv n_frames=0

<a id="batch_tests___grizzly_bear_5k_100_on_5_k_"></a>
#### batch_tests       @ grizzly_bear/5K/100_on_5k

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_1.mkv n_frames=0 batch_size=1

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_2.mkv n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_3.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_5.mkv n_frames=0 batch_size=5

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_10.mkv n_frames=0 batch_size=10

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_25.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_30.mkv n_frames=0 batch_size=30

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_40.mkv n_frames=0 batch_size=40

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_50.mkv n_frames=0 batch_size=50

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_55.mkv n_frames=0 batch_size=55

<a id="more_bear_videos___grizzly_bear_5k_100_on_5_k_"></a>
#### more_bear_videos       @ grizzly_bear/5K/100_on_5k

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_3.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_2.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_2_det_batch_3.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_3.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_3_det_batch_3.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_4.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_4_det_batch_3.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_5.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_5_det_batch_3.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_6.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_6_det_batch_3.mkv n_frames=0 batch_size=3

<a id="zip___more_bear_videos_grizzly_bear_5k_100_on_5k_"></a>
##### zip       @ more_bear_videos/grizzly_bear/5K/100_on_5k

zr grizzly_bear_1_det_batch_3_faster_rcnn_nas_coco_2018_01_28 results/faster_rcnn_nas_coco_2018_01_28/grizzly_bear_1_det_batch_3.mkv

zr faster_rcnn_nas_coco_2018_01_28_multiple results/faster_rcnn_nas_coco_2018_01_28


<a id="moose___5k_100_on_5k_"></a>
### moose       @ 5K/100_on_5k

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/moose_1_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_2.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/moose_2_det.mkv n_frames=0 batch_size=3

<a id="deer___5k_100_on_5k_"></a>
### deer       @ 5K/100_on_5k

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/deer_1_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_2.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/deer_2_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_3.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/deer_3_det.mkv n_frames=0 batch_size=3

<a id="coyote___5k_100_on_5k_"></a>
### coyote       @ 5K/100_on_5k

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_1.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/coyote_1_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_2.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/coyote_2_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_3.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/coyote_3_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_4.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/coyote_4_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_5.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/coyote_5_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_6.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/coyote_6_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_7.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/coyote_7_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_8.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/coyote_8_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_9.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/coyote_9_det.mkv n_frames=0 batch_size=3

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_10.mp4 save_file_name=results/faster_rcnn_nas_coco_2018_01_28/coyote_10_det.mkv n_frames=0 batch_size=3


<a id="evaluatio_n_"></a>
# evaluation

CUDA_VISIBLE_DEVICES=1 python3 eval.py --logtostderr --pipeline_config_path=faster_rcnn_nas_coco_2018_01_28_pipeline.config --checkpoint_dir=results/faster_rcnn_nas_coco_2018_01_28 --eval_dir=results/faster_rcnn_nas_coco_2018_01_28/eval --run_once=1

CUDA_VISIBLE_DEVICES=1 python3 wildlife_detection_eval.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt test_path=images/test save_path=images/test_vis n_frames=0 batch_size=25

<a id="ots_inferenc_e_"></a>
# ots inference

CUDA_VISIBLE_DEVICES=1 python3 infer.py ckpt_path=pre_trained_models/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb  batch_size=1  labels_path=data/mscoco_label_map.pbtxt src_path=videos/human save_path=results/human save_path=results/faster_rcnn_nas_coco_2018_01_28_ots/human save_fmt=1

CUDA_VISIBLE_DEVICES=1 python3 infer.py ckpt_path=pre_trained_models/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb  batch_size=1 labels_path=data/mscoco_label_map.pbtxt src_path=videos/human2 save_path=results/faster_rcnn_nas_coco_2018_01_28_ots/human2 save_fmt=1

CUDA_VISIBLE_DEVICES=1 python3 infer.py ckpt_path=pre_trained_models/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb  batch_size=1 labels_path=data/mscoco_label_map.pbtxt src_path=videos/human3 save_path=results/faster_rcnn_nas_coco_2018_01_28_ots/human3.mkv  save_fmt=2 n_frames=0 codec=H264

CUDA_VISIBLE_DEVICES=1 python3 infer.py ckpt_path=pre_trained_models/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb  batch_size=1 labels_path=data/mscoco_label_map.pbtxt src_path=videos/human4 save_path=results/faster_rcnn_nas_coco_2018_01_28_ots/human4.mkv  save_fmt=2 n_frames=0 codec=H264

CUDA_VISIBLE_DEVICES=1 python3 infer.py ckpt_path=pre_trained_models/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb  batch_size=1 labels_path=data/mscoco_label_map.pbtxt src_path=videos/human5 save_path=results/faster_rcnn_nas_coco_2018_01_28_ots/human5.mkv  save_fmt=2 n_frames=0 codec=H264



<a id="marcin_180608___ots_inference_"></a>
### marcin_180608       @ ots_inference/

<a id="bear___marcin_180608_ots_inference_"></a>
#### bear       @ marcin_180608/ots_inference/

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_3 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_4 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_5 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_6 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_3 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_4 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_5 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/grizzly_bear_6_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

<a id="moose___marcin_180608_ots_inference_"></a>
#### moose       @ marcin_180608/ots_inference/

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_3 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_3 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_4 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_5 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_6 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_7 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_8 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_9 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_10 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_12 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_13 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_3 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_7_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_8_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_9_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_11_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_3 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_13_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

<a id="coyote___marcin_180608_ots_inference_"></a>
#### coyote       @ marcin_180608/ots_inference/

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_3 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2



<a id="dee_r_"></a>
# deer

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_3 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_4 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_2_2
/data/acamp/marcin_180613/deer_2_4 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_3 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_4_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_4_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_5_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_3 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_4 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_7_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_8_1 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_nas_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_8_2 save_dir=results/faster_rcnn_nas_coco_2018_01_28/ n_frames=0 batch_size=2

