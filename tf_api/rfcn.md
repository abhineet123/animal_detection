<!-- MarkdownTOC -->

- [train](#train_)
    - [5k       @ train](#5k___trai_n_)
    - [10k       @ train](#10k___trai_n_)
    - [10k_ar       @ train](#10k_ar___trai_n_)
    - [10ka       @ train](#10ka___trai_n_)
    - [1K_3a       @ train](#1k_3a___trai_n_)
        - [mp       @ 1K_3a/train](#mp___1k_3a_trai_n_)
    - [1K_3a_sampled       @ train](#1k_3a_sampled___trai_n_)
    - [500_static3a       @ train](#500_static3a___trai_n_)
    - [200_static3a       @ train](#200_static3a___trai_n_)
    - [20K_vid3a       @ train](#20k_vid3a___trai_n_)
    - [10k6_vid_entire_seq       @ train](#10k6_vid_entire_seq___trai_n_)
    - [20K6_5       @ train](#20k6_5___trai_n_)
    - [20K6_60       @ train](#20k6_60___trai_n_)
    - [1k8_vid       @ train](#1k8_vid___trai_n_)
    - [1k8_vid_entire_seq       @ train](#1k8_vid_entire_seq___trai_n_)
    - [1k8_vid_even_min_1       @ train](#1k8_vid_even_min_1___trai_n_)
- [tf_api_eval](#tf_api_eval_)
    - [500_static3a       @ tf_api_eval](#500_static3a___tf_api_eva_l_)
        - [class_agnostic       @ 500_static3a/tf_api_eval](#class_agnostic___500_static3a_tf_api_eval_)
        - [nms_thresh_0       @ 500_static3a/tf_api_eval](#nms_thresh_0___500_static3a_tf_api_eval_)
    - [200_static3a       @ tf_api_eval](#200_static3a___tf_api_eva_l_)
    - [20K_vid3a       @ tf_api_eval](#20k_vid3a___tf_api_eva_l_)
        - [acamp200_static3a_inverted       @ 20K_vid3a/tf_api_eval](#acamp200_static3a_inverted___20k_vid3a_tf_api_eva_l_)
    - [1k3a_sampled       @ tf_api_eval](#1k3a_sampled___tf_api_eva_l_)
    - [10k6_vid_entire_seq       @ tf_api_eval](#10k6_vid_entire_seq___tf_api_eva_l_)
        - [nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval](#nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_)
    - [20k6_5       @ tf_api_eval](#20k6_5___tf_api_eva_l_)
        - [inverted       @ 20k6_5/tf_api_eval](#inverted___20k6_5_tf_api_eval_)
        - [inverted_only_video       @ 20k6_5/tf_api_eval](#inverted_only_video___20k6_5_tf_api_eval_)
            - [nms_thresh_0       @ inverted_only_video/20k6_5/tf_api_eval](#nms_thresh_0___inverted_only_video_20k6_5_tf_api_eval_)
        - [combine_sequences_640x360       @ 20k6_5/tf_api_eval](#combine_sequences_640x360___20k6_5_tf_api_eval_)
        - [combine_sequences_416x416       @ 20k6_5/tf_api_eval](#combine_sequences_416x416___20k6_5_tf_api_eval_)
    - [20k6_60       @ tf_api_eval](#20k6_60___tf_api_eva_l_)
        - [acamp_1_per_class_6_class       @ 20k6_60/tf_api_eval](#acamp_1_per_class_6_class___20k6_60_tf_api_eva_l_)
    - [1k8_vid       @ tf_api_eval](#1k8_vid___tf_api_eva_l_)
    - [1k8_vid_entire_seq       @ tf_api_eval](#1k8_vid_entire_seq___tf_api_eva_l_)
    - [1k8_vid_even_min_1       @ tf_api_eval](#1k8_vid_even_min_1___tf_api_eva_l_)
- [eval](#eva_l_)
    - [20k7       @ eval](#20k7___eval_)
    - [20k5       @ eval](#20k5___eval_)
    - [40k3_coco       @ eval](#40k3_coco___eval_)
    - [# export_inference_graph](#__export_inference_graph_)
    - [pre_trained       @ export_inference_graph](#pre_trained___export_inference_graph_)
    - [5k       @ export_inference_graph](#5k___export_inference_graph_)
        - [300K       @ 5k/export_inference_graph](#300k___5k_export_inference_grap_h_)
        - [500K       @ 5k/export_inference_graph](#500k___5k_export_inference_grap_h_)
    - [10k       @ export_inference_graph](#10k___export_inference_graph_)
    - [10k_ar       @ export_inference_graph](#10k_ar___export_inference_graph_)
    - [10ka       @ export_inference_graph](#10ka___export_inference_graph_)
    - [40k3_coco       @ export_inference_graph](#40k3_coco___export_inference_graph_)
        - [fiasco       @ 40k3_coco/export_inference_graph](#fiasco___40k3_coco_export_inference_graph_)
    - [# test](#__test_)
    - [pretrained       @ test](#pretrained___test_)
        - [camera       @ pretrained/test](#camera___pretrained_tes_t_)
            - [only_person       @ camera/pretrained/test](#only_person___camera_pretrained_test_)
            - [Pt_Grey       @ camera/pretrained/test](#pt_grey___camera_pretrained_test_)
            - [mtf       @ camera/pretrained/test](#mtf___camera_pretrained_test_)
    - [100_on_5k       @ test](#100_on_5k___test_)
    - [5K       @ test](#5k___test_)
        - [300K       @ 5K/test](#300k___5k_tes_t_)
        - [500K       @ 5K/test](#500k___5k_tes_t_)
    - [10K       @ test](#10k___test_)
        - [camera       @ 10K/test](#camera___10k_test_)
    - [10K_ar       @ test](#10k_ar___test_)
    - [40k3_coco       @ test](#40k3_coco___test_)
    - [40k3_coco_on_20K4_static       @ test](#40k3_coco_on_20k4_static___test_)
- [](#_)
- [](#__1)
- [](#__2)

<!-- /MarkdownTOC -->

mv /usr/local/lib/python3.5/dist-packages/object_detection-0.1-py3.5.egg/object_detection_annoying/ /usr/local/lib/python3.5/dist-packages/object_detection-0.1-py3.5.egg/object_detection

mv /usr/local/lib/python3.5/dist-packages/object_detection-0.1-py3.5.egg/object_detection /usr/local/lib/python3.5/dist-packages/object_detection-0.1-py3.5.egg/object_detection_annoying

protoc protos/*.proto --python_out=.
protoc object_detection/protos/*.proto --python_out=.

<a id="train_"></a>
# train

CUDA_VISIBLE_DEVICES=1 python3 train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/pipeline.config --train_dir=trained/rfcn_resnet101_coco_2018_01_28

<a id="5k___trai_n_"></a>
## 5k       @ train

CUDA_VISIBLE_DEVICES=0 python3 train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_5k.config --train_dir=trained/rfcn_resnet101_coco_2018_01_28_5k

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=/home/abhineet/acamp/acamp_code/tf_api/configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_5k.config --train_dir=/home/abhineet/acamp/acamp_code/tf_api/trained/rfcn_resnet101_coco_2018_01_28_5k

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_5k.config --train_dir=trained/rfcn_resnet101_coco_2018_01_28_5k

<a id="10k___trai_n_"></a>
## 10k       @ train

CUDA_VISIBLE_DEVICES=0 python3 train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10k.config --train_dir=trained/rfcn_resnet101_coco_2018_01_28_10k

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=/home/abhineet/acamp/acamp_code/tf_api/configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10k.config --train_dir=/home/abhineet/acamp/acamp_code/tf_api/trained/rfcn_resnet101_coco_2018_01_28_10k

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10k.config --train_dir=trained/rfcn_resnet101_coco_2018_01_28_10k

<a id="10k_ar___trai_n_"></a>
## 10k_ar       @ train

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/batch_6/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10k_ar.config --train_dir=trained/batch_6/rfcn_resnet101_coco_2018_01_28_10k_ar

<a id="10ka___trai_n_"></a>
## 10ka       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/617_w18/Assbatch_6ignment2/models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10ka.config --train_dir=trained/rfcn_resnet101_coco_2018_01_28_10ka

<a id="1k_3a___trai_n_"></a>
## 1K_3a       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_1k3a.config --train_dir=trained/rfcn_resnet101_1k3a --n_steps=500000 --save_interval_secs=600

<a id="mp___1k_3a_trai_n_"></a>
### mp       @ 1K_3a/train

TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=1 CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_1k3a.config --train_dir=trained/rfcn_resnet101_1k3a_mp --n_steps=500000 --save_interval_secs=600 --enable_mixed_precision=1

<a id="1k_3a_sampled___trai_n_"></a>
## 1K_3a_sampled       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_1k3a_sampled.config --train_dir=trained/rfcn_resnet101_1k3a_sampled --n_steps=500000 --save_interval_secs=600

<a id="500_static3a___trai_n_"></a>
## 500_static3a       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_500_static3a.config --train_dir=trained/rfcn_resnet101_500_static3a --n_steps=500000 --save_interval_secs=600

<a id="200_static3a___trai_n_"></a>
## 200_static3a       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_200_static3a.config --train_dir=trained/rfcn_resnet101_200_static3a --n_steps=500000 --save_interval_secs=600

<a id="20k_vid3a___trai_n_"></a>
## 20K_vid3a       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20K_vid3a.config --train_dir=trained/rfcn_resnet101_20K_vid3a --n_steps=100000 --save_interval_secs=600

<a id="10k6_vid_entire_seq___trai_n_"></a>
## 10k6_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10k6_vid_entire_seq.config --train_dir=trained/rfcn_resnet101_10k6_vid_entire_seq --n_steps=1000000 --save_interval_secs=600

<a id="20k6_5___trai_n_"></a>
## 20K6_5       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k6_5.config --train_dir=trained/rfcn_resnet101_20k6_5 --n_steps=1000000 --save_interval_secs=600

<a id="20k6_60___trai_n_"></a>
## 20K6_60       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k6_60.config --train_dir=trained/rfcn_resnet101_20k6_60 --n_steps=1000000 --save_interval_secs=600

<a id="1k8_vid___trai_n_"></a>
## 1k8_vid       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_1k8_vid.config --train_dir=trained/rfcn_resnet101_1k8_vid --n_steps=500000 --save_interval_secs=600


<a id="1k8_vid_entire_seq___trai_n_"></a>
## 1k8_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_1k8_vid_entire_seq.config --train_dir=trained/rfcn_resnet101_1k8_vid_entire_seq --n_steps=1000000 --save_interval_secs=600

<a id="1k8_vid_even_min_1___trai_n_"></a>
## 1k8_vid_even_min_1       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_1k8_vid_even_min_1.config --train_dir=trained/rfcn_resnet101_1k8_vid_even_min_1 --n_steps=1000000 --save_interval_secs=600


<a id="tf_api_eval_"></a>
# tf_api_eval

<a id="500_static3a___tf_api_eva_l_"></a>
## 500_static3a       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="class_agnostic___500_static3a_tf_api_eval_"></a>
### class_agnostic       @ 500_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1

<a id="nms_thresh_0___500_static3a_tf_api_eval_"></a>
### nms_thresh_0       @ 500_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0

<a id="200_static3a___tf_api_eva_l_"></a>
## 200_static3a       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_200_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_200_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.01

<a id="20k_vid3a___tf_api_eva_l_"></a>
## 20K_vid3a       @ tf_api_eval

<a id="acamp200_static3a_inverted___20k_vid3a_tf_api_eva_l_"></a>
### acamp200_static3a_inverted       @ 20K_vid3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_20K_vid3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20K_vid3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp200_static3a_inverted load_det=0 score_thresholds=0:1:0.01

<a id="1k3a_sampled___tf_api_eva_l_"></a>
## 1k3a_sampled       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_1k3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_1k3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="10k6_vid_entire_seq___tf_api_eva_l_"></a>
## 10k6_vid_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=2 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_"></a>
### nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=2 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=1 score_thresholds=0:1:0.001 allow_seq_skipping=1 inference_dir=inference_nms_thresh_0

<a id="20k6_5___tf_api_eva_l_"></a>
## 20k6_5       @ tf_api_eval

<a id="inverted___20k6_5_tf_api_eval_"></a>
### inverted       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001

<a id="inverted_only_video___20k6_5_tf_api_eval_"></a>
### inverted_only_video       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/rfcn_resnet101_20k6_5/100000_on_inverted out_postfix=inv_only_vid load_det=1 score_thresholds=0:1:0.001

<a id="nms_thresh_0___inverted_only_video_20k6_5_tf_api_eval_"></a>
#### nms_thresh_0       @ inverted_only_video/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=36 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/rfcn_resnet101_20k6_5/100000_on_nms_thresh_0_inv_only_vid out_postfix=nms_thresh_0_inv_only_vid load_det=1 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0

<a id="combine_sequences_640x360___20k6_5_tf_api_eval_"></a>
### combine_sequences_640x360       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=76 show_img=0 n_classes=6  eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 combine_sequences=1 input_size=640x360

<a id="combine_sequences_416x416___20k6_5_tf_api_eval_"></a>
### combine_sequences_416x416       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 combine_sequences=1 input_size=416x416

<a id="20k6_60___tf_api_eva_l_"></a>
## 20k6_60       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_20k6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=36 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k6_60.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_train inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 frozen_graph_path=trained/rfcn_resnet101_20k6_60/inference_563391/frozen_inference_graph.pb trained_checkpoint_prefix=563391

<a id="acamp_1_per_class_6_class___20k6_60_tf_api_eva_l_"></a>
### acamp_1_per_class_6_class       @ 20k6_60/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_20k6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=36 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_1_per_class_6_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k6_60.config sleep_time=10 write_summary=1 save_det=1 frozen_graph_path=trained/rfcn_resnet101_20k6_60/inference_563391/frozen_inference_graph.pb trained_checkpoint_prefix=563391


<a id="1k8_vid___tf_api_eva_l_"></a>
## 1k8_vid       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_1k8_vid labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=36 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_1k8_vid.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.01

<a id="1k8_vid_entire_seq___tf_api_eva_l_"></a>
## 1k8_vid_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_1k8_vid_entire_seq labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_1k8_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="1k8_vid_even_min_1___tf_api_eva_l_"></a>
## 1k8_vid_even_min_1       @ tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=32 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1



<a id="eva_l_"></a>
# eval

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/eval.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_5k.config --checkpoint_dir=trained/rfcn_resnet101_coco_2018_01_28_5k --eval_dir=trained/rfcn_resnet101_coco_2018_01_28_5k/eval --num_examples=0


<a id="20k7___eval_"></a>
## 20k7       @ eval

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k7.config --train_dir=trained/rfcn_resnet101_coco_2018_01_28_20k7

<a id="20k5___eval_"></a>
## 20k5       @ eval

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k5.config --train_dir=trained/rfcn_resnet101_coco_2018_01_28_20k5

<a id="40k3_coco___eval_"></a>
## 40k3_coco       @ eval

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_40k3_coco.config --train_dir=trained/rfcn_resnet101_coco_2018_01_28_40k3_coco



-------------------------------------------------------------
<a id="__export_inference_graph_"></a>
# export_inference_graph
-------------------------------------------------------------


    
CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/rfcn_resnet101_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix trained/rfcn_resnet101_coco_2018_01_28/model.ckpt-200000 --output_directory trained/rfcn_resnet101_coco_2018_01_28/inference


<a id="pre_trained___export_inference_graph_"></a>
## pre_trained       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/rfcn_resnet101_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix pre_trained_models/rfcn_resnet101_coco_2018_01_28 --output_directory pre_trained_models/rfcn_resnet101_coco_2018_01_28/inference

<a id="5k___export_inference_graph_"></a>
## 5k       @ export_inference_graph

<a id="300k___5k_export_inference_grap_h_"></a>
### 300K       @ 5k/export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_5k.config --trained_checkpoint_prefix trained/rfcn_resnet101_coco_2018_01_28_5k/model.ckpt-300000 --output_directory trained/rfcn_resnet101_coco_2018_01_28_5k/inference

<a id="500k___5k_export_inference_grap_h_"></a>
### 500K       @ 5k/export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_5k.config --trained_checkpoint_prefix trained/rfcn_resnet101_coco_2018_01_28_5k/model.ckpt-500000 --output_directory trained/rfcn_resnet101_coco_2018_01_28_5k/inference_500K

<a id="10k___export_inference_graph_"></a>
## 10k       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10k.config --trained_checkpoint_prefix trained/rfcn_resnet101_coco_2018_01_28_10k/model.ckpt-500000 --output_directory trained/rfcn_resnet101_coco_2018_01_28_10k/inference_500K

<a id="10k_ar___export_inference_graph_"></a>
## 10k_ar       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/batch_6/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10k_ar.config --trained_checkpoint_prefix trained/batch_6/rfcn_resnet101_coco_2018_01_28_10k_ar/model.ckpt-500000 --output_directory trained/batch_6/rfcn_resnet101_coco_2018_01_28_10k_ar/inference_500K

<a id="10ka___export_inference_graph_"></a>
## 10ka       @ export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10ka.config --trained_checkpoint_prefix trained/rfcn_resnet101_coco_2018_01_28_10ka/model.ckpt-500000 --output_directory trained/rfcn_resnet101_coco_2018_01_28_10ka/inference_500K

<a id="40k3_coco___export_inference_graph_"></a>
## 40k3_coco       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_40k3_coco.config --trained_checkpoint_prefix trained/rfcn_resnet101_coco_2018_01_28_40k3_coco 


<a id="fiasco___40k3_coco_export_inference_graph_"></a>
### fiasco       @ 40k3_coco/export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_5k.config --trained_checkpoint_prefix trained/rfcn_resnet101_coco_2018_01_28_5k/model.ckpt-179754 --output_directory trained/rfcn_resnet101_coco_2018_01_28_5k/inference_179754


-------------------------------------------------------------
<a id="__test_"></a>
# test
-------------------------------------------------------------


<a id="pretrained___test_"></a>
## pretrained       @ test

<a id="camera___pretrained_tes_t_"></a>
### camera       @ pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=900 n_classes=90 write_det=0

<a id="only_person___camera_pretrained_test_"></a>
#### only_person       @ camera/pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0 classes_to_include=person,

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0 classes_to_include=person,


<a id="pt_grey___camera_pretrained_test_"></a>
#### Pt_Grey       @ camera/pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1224 vis_height=1024 n_classes=90 write_det=0 classes_to_include=person, use_ptgrey=1 threaded_mode=1 rgb_mode=1 video_mode=1

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1224 vis_height=1024 n_classes=90 write_det=0 classes_to_include=person, use_ptgrey=1 threaded_mode=1 rgb_mode=1 video_mode=1

<a id="mtf___camera_pretrained_test_"></a>
#### mtf       @ camera/pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1224 vis_height=1024 n_classes=90 write_det=0 classes_to_include=person, use_mtf=1 mtf_args=pipeline,v,img_source,u,vp_usb_res,1920x1080

<a id="100_on_5k___test_"></a>
## 100_on_5k       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt root_dir=/data/acamp/acamp5k/test/images save_dir=results/rfcn_resnet101_coco_2018_01_28_100_on_5k/ n_frames=0 batch_size=25

<a id="5k___test_"></a>
## 5K       @ test

<a id="300k___5k_tes_t_"></a>
### 300K       @ 5K/test

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28_5k/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=data/acamp5k_test.txt root_dir=/data/acamp/acamp5k/test/images save_dir=results/rfcn_resnet101_coco_2018_01_28_5k n_frames=0 batch_size=25

<a id="500k___5k_tes_t_"></a>
### 500K       @ 5K/test

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28_5k/inference_500K/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=data/acamp5k_test.txt root_dir=/data/acamp/acamp5k/test/images save_dir=results/rfcn_resnet101_coco_2018_01_28_5k_500K n_frames=0 batch_size=25

<a id="10k___test_"></a>
## 10K       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28_10k/inference_500K/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp10k_test.txt save_dir=results/rfcn_resnet101_coco_2018_01_28_10k_500K n_frames=0 n_classes=5

<a id="camera___10k_test_"></a>
### camera       @ 10K/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28_10k/inference_500K/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=720 n_classes=5 write_det=0

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28_10k/inference_500K/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=camera_1 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=720 n_classes=5 write_det=0


<a id="10k_ar___test_"></a>
## 10K_ar       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/batch_6/rfcn_resnet101_coco_2018_01_28_10k_ar/inference_500K/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp10k_test.txt save_dir=results/rfcn_resnet101_coco_2018_01_28_10k_ar_500K n_frames=0 n_classes=5

<a id="40k3_coco___test_"></a>
## 40k3_coco       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py model_list=trained_models.txt model_id=15 list_file_name=acamp40k3_test_no_human.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=25 n_classes=3


<a id="40k3_coco_on_20k4_static___test_"></a>
## 40k3_coco_on_20K4_static       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py model_list=trained_models.txt model_id=15 list_file_name=acamp20k4_static_test.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=1 n_classes=3

====================================================================
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="<a-id="-1"></a>"></a>
<a id="_"></a>
<a id="-1"></a>
====================================================================
====================================================================
<a id="__1"></a>
<a id="-2"></a>
====================================================================
====================================================================
<a id="__2"></a>
<a id="-3"></a>
====================================================================
====================================================================
