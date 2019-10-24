<!-- MarkdownTOC -->

- [train](#train_)
    - [5k       @ train](#5k___trai_n_)
    - [10k       @ train](#10k___trai_n_)
    - [40k3_coco       @ train](#40k3_coco___trai_n_)
    - [1K_3a_sampled       @ train](#1k_3a_sampled___trai_n_)
    - [500_static3a       @ train](#500_static3a___trai_n_)
    - [200_static3a       @ train](#200_static3a___trai_n_)
    - [20K_vid3a       @ train](#20k_vid3a___trai_n_)
    - [10k6_vid_entire_seq       @ train](#10k6_vid_entire_seq___trai_n_)
    - [20K6_5       @ train](#20k6_5___trai_n_)
    - [20k6_60       @ train](#20k6_60___trai_n_)
    - [1k8_vid       @ train](#1k8_vid___trai_n_)
    - [1k8_vid_entire_seq       @ train](#1k8_vid_entire_seq___trai_n_)
    - [1k8_vid_even_min_1       @ train](#1k8_vid_even_min_1___trai_n_)
- [tf_api_eval](#tf_api_eval_)
    - [acamp1K_static3a_sampled       @ tf_api_eval](#acamp1k_static3a_sampled___tf_api_eva_l_)
    - [acamp200K_static3a       @ tf_api_eval](#acamp200k_static3a___tf_api_eva_l_)
    - [500_static3a       @ tf_api_eval](#500_static3a___tf_api_eva_l_)
    - [500_static3a       @ tf_api_eval](#500_static3a___tf_api_eva_l__1)
        - [nms_thresh_0       @ 500_static3a/tf_api_eval](#nms_thresh_0___500_static3a_tf_api_eval_)
    - [20K_vid3a       @ tf_api_eval](#20k_vid3a___tf_api_eva_l_)
        - [acamp200_static3a_inverted       @ 20K_vid3a/tf_api_eval](#acamp200_static3a_inverted___20k_vid3a_tf_api_eva_l_)
    - [10k6_vid_entire_seq       @ tf_api_eval](#10k6_vid_entire_seq___tf_api_eva_l_)
        - [nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval](#nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_)
    - [20k6_5       @ tf_api_eval](#20k6_5___tf_api_eva_l_)
            - [inverted       @ 20k6_5/tf_api_eval](#inverted___20k6_5_tf_api_eval_)
                - [combine_sequences_416x416       @ inverted/20k6_5/tf_api_eval](#combine_sequences_416x416___inverted_20k6_5_tf_api_eva_l_)
            - [inverted_only_video       @ 20k6_5/tf_api_eval](#inverted_only_video___20k6_5_tf_api_eval_)
                - [nms_thresh_0       @ inverted_only_video/20k6_5/tf_api_eval](#nms_thresh_0___inverted_only_video_20k6_5_tf_api_eval_)
    - [20k6_60       @ tf_api_eval](#20k6_60___tf_api_eva_l_)
        - [inverted       @ 20k6_60/tf_api_eval](#inverted___20k6_60_tf_api_eva_l_)
            - [frozen_inference_graph_308201       @ inverted/20k6_60/tf_api_eval](#frozen_inference_graph_308201___inverted_20k6_60_tf_api_eval_)
                - [combine_sequences_640x360       @ frozen_inference_graph_308201/inverted/20k6_60/tf_api_eval](#combine_sequences_640x360___frozen_inference_graph_308201_inverted_20k6_60_tf_api_eval_)
        - [acamp_1_per_class_6_class       @ 20k6_60/tf_api_eval](#acamp_1_per_class_6_class___20k6_60_tf_api_eva_l_)
    - [1k8_vid_entire_seq       @ tf_api_eval](#1k8_vid_entire_seq___tf_api_eva_l_)
    - [1k8_vid_even_min_1       @ tf_api_eval](#1k8_vid_even_min_1___tf_api_eva_l_)

<!-- /MarkdownTOC -->

<a id="train_"></a>
# train

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/pipeline.config --train_dir=trained/ssd_inception_v2_coco_2017_11_17

<a id="5k___trai_n_"></a>
## 5k       @ train

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_5k.config --train_dir=trained/ssd_inception_v2_coco_2017_11_17_5k

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=/home/abhineet/acamp/acamp_code/tf_api/configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_5k.config --train_dir=/home/abhineet/acamp/acamp_code/tf_api/trained/ssd_inception_v2_coco_2017_11_17_5k

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_5k.config --train_dir=trained/ssd_inception_v2_coco_2017_11_17_5k

<a id="10k___trai_n_"></a>
## 10k       @ train

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_10k.config --train_dir=trained/ssd_inception_v2_coco_2017_11_17_10k


CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/batch_8/ssd_inception_v2_coco_2017_11_17/ssd_inception_10k.config --train_dir=trained/batch_8/ssd_inception_v2_coco_2017_11_17_10k

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/batch_10/ssd_inception_v2_coco_2017_11_17/ssd_inception_10k.config --train_dir=trained/batch_10/ssd_inception_v2_coco_2017_11_17_10k

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/batch_12/ssd_inception_v2_coco_2017_11_17/ssd_inception_10k.config --train_dir=trained/batch_12/ssd_inception_v2_coco_2017_11_17_10k

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/batch_16/ssd_inception_v2_coco_2017_11_17/ssd_inception_10k.config --train_dir=trained/batch_16/ssd_inception_v2_coco_2017_11_17_10k

<a id="40k3_coco___trai_n_"></a>
## 40k3_coco       @ train

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_40k3.config --train_dir=trained/ssd_inception_v2_coco_2017_11_17_40k3_coco


<a id="1k_3a_sampled___trai_n_"></a>
## 1K_3a_sampled       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_1k3a_sampled.config --train_dir=trained/ssd_inception_1k3a_sampled --n_steps=500000 --save_interval_secs=600

<a id="500_static3a___trai_n_"></a>
## 500_static3a       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_500_static3a.config --train_dir=trained/ssd_inception_500_static3a --n_steps=500000 --save_interval_secs=600

<a id="200_static3a___trai_n_"></a>
## 200_static3a       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_200_static3a.config --train_dir=trained/ssd_inception_200_static3a --n_steps=500000 --save_interval_secs=600

<a id="20k_vid3a___trai_n_"></a>
## 20K_vid3a       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20K_vid3a.config --train_dir=trained/ssd_inception_20K_vid3a --n_steps=100000 --save_interval_secs=600

<a id="10k6_vid_entire_seq___trai_n_"></a>
## 10k6_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_10k6_vid_entire_seq.config --train_dir=trained/ssd_inception_10k6_vid_entire_seq --n_steps=600000 --save_interval_secs=600

<a id="20k6_5___trai_n_"></a>
## 20K6_5       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_5.config --train_dir=trained/ssd_inception_20k6_5 --n_steps=600000 --save_interval_secs=600

<a id="20k6_60___trai_n_"></a>
## 20k6_60       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_60.config --train_dir=trained/ssd_inception_20k6_60 --n_steps=600000 --save_interval_secs=600


<a id="1k8_vid___trai_n_"></a>
## 1k8_vid       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_1k8_vid.config --train_dir=trained/ssd_inception_1k8_vid --n_steps=500000 --save_interval_secs=600


<a id="1k8_vid_entire_seq___trai_n_"></a>
## 1k8_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_1k8_vid_entire_seq.config --train_dir=trained/ssd_inception_1k8_vid_entire_seq --n_steps=600000 --save_interval_secs=600

<a id="1k8_vid_even_min_1___trai_n_"></a>
## 1k8_vid_even_min_1       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_1k8_vid_even_min_1.config --train_dir=trained/ssd_inception_1k8_vid_even_min_1 --n_steps=600000 --save_interval_secs=600


<a id="tf_api_eval_"></a>
# tf_api_eval

<a id="acamp1k_static3a_sampled___tf_api_eva_l_"></a>
## acamp1K_static3a_sampled       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_1k3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_1k3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="acamp200k_static3a___tf_api_eva_l_"></a>
## acamp200K_static3a       @ tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_200_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_200_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="500_static3a___tf_api_eva_l_"></a>
## 500_static3a       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="500_static3a___tf_api_eva_l__1"></a>
## 500_static3a       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1

<a id="nms_thresh_0___500_static3a_tf_api_eval_"></a>
### nms_thresh_0       @ 500_static3a/tf_api_eval


CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0

<a id="20k_vid3a___tf_api_eva_l_"></a>
## 20K_vid3a       @ tf_api_eval

<a id="acamp200_static3a_inverted___20k_vid3a_tf_api_eva_l_"></a>
### acamp200_static3a_inverted       @ 20K_vid3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_20K_vid3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20K_vid3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp200_static3a_inverted load_det=0 score_thresholds=0:1:0.001

<a id="10k6_vid_entire_seq___tf_api_eva_l_"></a>
## 10k6_vid_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_"></a>
### nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1 inference_dir=inference_nms_thresh_0 n_threads=8


<a id="20k6_5___tf_api_eva_l_"></a>
## 20k6_5       @ tf_api_eval

<a id="inverted___20k6_5_tf_api_eval_"></a>
#### inverted       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 n_threads=8

<a id="combine_sequences_416x416___inverted_20k6_5_tf_api_eva_l_"></a>
##### combine_sequences_416x416       @ inverted/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=7 show_img=0 n_classes=6  eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 combine_sequences=1 input_size=416x416

<a id="inverted_only_video___20k6_5_tf_api_eval_"></a>
#### inverted_only_video       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/ssd_inception_20k6_5/51446_on_inverted out_postfix=inv_only_vid load_det=1 score_thresholds=0:1:0.001

<a id="nms_thresh_0___inverted_only_video_20k6_5_tf_api_eval_"></a>
##### nms_thresh_0       @ inverted_only_video/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/ssd_inception_20k6_5/51446_on_nms_thresh_0_inv_only_vid out_postfix=nms_thresh_0_inv_only_vid load_det=1 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0

<a id="20k6_60___tf_api_eva_l_"></a>
## 20k6_60       @ tf_api_eval

<a id="inverted___20k6_60_tf_api_eva_l_"></a>
### inverted       @ 20k6_60/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_20k6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_60.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_train inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0

<a id="frozen_inference_graph_308201___inverted_20k6_60_tf_api_eval_"></a>
#### frozen_inference_graph_308201       @ inverted/20k6_60/tf_api_eval


CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_20k6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=160 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_60.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_train inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 frozen_graph_path=trained/ssd_inception_20k6_60/inference_308201/frozen_inference_graph.pb trained_checkpoint_prefix=308201

<a id="combine_sequences_640x360___frozen_inference_graph_308201_inverted_20k6_60_tf_api_eval_"></a>
##### combine_sequences_640x360       @ frozen_inference_graph_308201/inverted/20k6_60/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_20k6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_60.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_train inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 frozen_graph_path=trained/ssd_inception_20k6_60/inference_308201/frozen_inference_graph.pb trained_checkpoint_prefix=308201 combine_sequences=1 input_size=640x360 show_animation=1 save_animation=1 show_img=1 eval_every=-1

<a id="acamp_1_per_class_6_class___20k6_60_tf_api_eva_l_"></a>
### acamp_1_per_class_6_class       @ 20k6_60/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_20k6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=160 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_1_per_class_6_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_60.config sleep_time=10 write_summary=1 save_det=1 frozen_graph_path=trained/ssd_inception_20k6_60/inference_308201/frozen_inference_graph.pb trained_checkpoint_prefix=308201

<a id="1k8_vid_entire_seq___tf_api_eva_l_"></a>
## 1k8_vid_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_1k8_vid_entire_seq labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_1k8_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 allow_seq_skipping=1 n_threads=0

<a id="1k8_vid_even_min_1___tf_api_eva_l_"></a>
## 1k8_vid_even_min_1       @ tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 allow_seq_skipping=1 n_threads=0


