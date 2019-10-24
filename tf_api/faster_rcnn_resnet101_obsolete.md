<!-- MarkdownTOC -->

- [export_inference_graph](#export_inference_grap_h_)
    - [pre_trained       @ export_inference_graph](#pre_trained___export_inference_graph_)
    - [5K       @ export_inference_graph](#5k___export_inference_graph_)
    - [10k       @ export_inference_graph](#10k___export_inference_graph_)
    - [10k_ar       @ export_inference_graph](#10k_ar___export_inference_graph_)
    - [10ka       @ export_inference_graph](#10ka___export_inference_graph_)
    - [20k3       @ export_inference_graph](#20k3___export_inference_graph_)
    - [20k3_hs       @ export_inference_graph](#20k3_hs___export_inference_graph_)
    - [20k3_coco       @ export_inference_graph](#20k3_coco___export_inference_graph_)
        - [python_2       @ 20k3_coco/export_inference_graph](#python_2___20k3_coco_export_inference_graph_)
    - [20k3_hs_auto       @ export_inference_graph](#20k3_hs_auto___export_inference_graph_)
    - [20k4       @ export_inference_graph](#20k4___export_inference_graph_)
    - [20k_hs       @ export_inference_graph](#20k_hs___export_inference_graph_)
    - [20k_coco       @ export_inference_graph](#20k_coco___export_inference_graph_)
    - [20k7       @ export_inference_graph](#20k7___export_inference_graph_)
    - [20k5       @ export_inference_graph](#20k5___export_inference_graph_)
    - [25k7       @ export_inference_graph](#25k7___export_inference_graph_)
    - [25k5       @ export_inference_graph](#25k5___export_inference_graph_)
    - [25k3       @ export_inference_graph](#25k3___export_inference_graph_)
        - [regularly_spaced       @ 25k3/export_inference_graph](#regularly_spaced___25k3_export_inference_grap_h_)
    - [1600_static3       @ export_inference_graph](#1600_static3___export_inference_graph_)
    - [40k3_coco       @ export_inference_graph](#40k3_coco___export_inference_graph_)
        - [regularly_spaced       @ 40k3_coco/export_inference_graph](#regularly_spaced___40k3_coco_export_inference_graph_)
        - [fiasco       @ 40k3_coco/export_inference_graph](#fiasco___40k3_coco_export_inference_graph_)
- [test](#tes_t_)
    - [list       @ test](#list___test_)
        - [point_grey       @ list/test](#point_grey___list_tes_t_)
            - [ffmpeg       @ point_grey/list/test](#ffmpeg___point_grey_list_test_)
    - [pretrained       @ test](#pretrained___test_)
        - [camera       @ pretrained/test](#camera___pretrained_tes_t_)
            - [only_person       @ camera/pretrained/test](#only_person___camera_pretrained_test_)
            - [Pt_Grey       @ camera/pretrained/test](#pt_grey___camera_pretrained_test_)
        - [10kh       @ pretrained/test](#10kh___pretrained_tes_t_)
    - [100_on_5k       @ test](#100_on_5k___test_)
    - [100_on_20k4       @ test](#100_on_20k4___test_)
    - [5K       @ test](#5k___test_)
    - [5K_on_10ka       @ test](#5k_on_10ka___test_)
        - [5k_on_20k4       @ 5K_on_10ka/test](#5k_on_20k4___5k_on_10ka_tes_t_)
    - [5k_on_train       @ test](#5k_on_train___test_)
    - [10k       @ test](#10k___test_)
        - [10k_on_20k3_hs       @ 10k/test](#10k_on_20k3_hs___10k_test_)
        - [camera       @ 10k/test](#camera___10k_test_)
        - [unlabeled       @ 10k/test](#unlabeled___10k_test_)
    - [10k_ar_on_10k       @ test](#10k_ar_on_10k___test_)
    - [10ka       @ test](#10ka___test_)
        - [moose_21_1       @ 10ka/test](#moose_21_1___10ka_tes_t_)
        - [10ka_on_5k       @ 10ka/test](#10ka_on_5k___10ka_tes_t_)
        - [10ka_on_20k4       @ 10ka/test](#10ka_on_20k4___10ka_tes_t_)
    - [10ka_on_train       @ test](#10ka_on_train___test_)
    - [20k3       @ test](#20k3___test_)
        - [camera       @ 20k3/test](#camera___20k3_tes_t_)
    - [20k3_hs       @ test](#20k3_hs___test_)
        - [20k3_hs_only_h       @ 20k3_hs/test](#20k3_hs_only_h___20k3_hs_test_)
        - [camera       @ 20k3_hs/test](#camera___20k3_hs_test_)
            - [only_person       @ camera/20k3_hs/test](#only_person___camera_20k3_hs_tes_t_)
    - [20k3_coco       @ test](#20k3_coco___test_)
        - [acamp20k_test_180719       @ 20k3_coco/test](#acamp20k_test_180719___20k3_coco_test_)
        - [camera       @ 20k3_coco/test](#camera___20k3_coco_test_)
            - [mtf       @ camera/20k3_coco/test](#mtf___camera_20k3_coco_tes_t_)
    - [20k3_on_40k3_coco       @ test](#20k3_on_40k3_coco___test_)
    - [20k3_on_20k3_missing       @ test](#20k3_on_20k3_missing___test_)
    - [20k_hs       @ test](#20k_hs___test_)
        - [camera       @ 20k_hs/test](#camera___20k_hs_tes_t_)
    - [20k_coco       @ test](#20k_coco___test_)
        - [camera       @ 20k_coco/test](#camera___20k_coco_tes_t_)
    - [20k4       @ test](#20k4___test_)
    - [20k4_on_5k       @ test](#20k4_on_5k___test_)
    - [20k4_on_train       @ test](#20k4_on_train___test_)
    - [20k7       @ test](#20k7___test_)
        - [20k7_on_20k4       @ 20k7/test](#20k7_on_20k4___20k7_tes_t_)
        - [unlabeled       @ 20k7/test](#unlabeled___20k7_tes_t_)
    - [20k5       @ test](#20k5___test_)
        - [20k5_on_20k4       @ 20k5/test](#20k5_on_20k4___20k5_tes_t_)
        - [unlabeled       @ 20k5/test](#unlabeled___20k5_tes_t_)
    - [25k3_on_40k3       @ test](#25k3_on_40k3___test_)
    - [25k3_on_training       @ test](#25k3_on_training___test_)
    - [40k3_coco       @ test](#40k3_coco___test_)
    - [40k3_coco_on_20K4_static       @ test](#40k3_coco_on_20k4_static___test_)
    - [40k3_coco_on_training       @ test](#40k3_coco_on_training___test_)
    - [1600_static3_on_40K3       @ test](#1600_static3_on_40k3___test_)
    - [1600_static3_on_train       @ test](#1600_static3_on_train___test_)
    - [1600_static3_on_20K4_static       @ test](#1600_static3_on_20k4_static___test_)
    - [1600_static3_on_20k3__no_human       @ test](#1600_static3_on_20k3_no_human___test_)
    - [1600_static3_on_training       @ test](#1600_static3_on_training___test_)

<!-- /MarkdownTOC -->

<a id="export_inference_grap_h_"></a>
# export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt-1467802 --output_directory trained/faster_rcnn_resnet101_coco_2018_01_28/inference

<a id="pre_trained___export_inference_graph_"></a>
## pre_trained       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix pre_trained_models/faster_rcnn_resnet101_coco_2018_01_28 --output_directory pre_trained_models/faster_rcnn_resnet101_coco_2018_01_28/inference


<a id="5k___export_inference_graph_"></a>
## 5K       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_5k.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_5k/model.ckpt-536475 --output_directory trained/faster_rcnn_resnet101_coco_2018_01_28_5k/inference_536475

<a id="10k___export_inference_graph_"></a>
## 10k       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_10k/model.ckpt-746846 --output_directory trained/faster_rcnn_resnet101_coco_2018_01_28_10k/inference_746846

<a id="10k_ar___export_inference_graph_"></a>
## 10k_ar       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/batch_6/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10k_ar.config --trained_checkpoint_prefix trained/batch_6/faster_rcnn_resnet101_coco_2018_01_28_10k_ar/model.ckpt-205110 --output_directory trained/batch_6/faster_rcnn_resnet101_coco_2018_01_28_10k_ar/inference_205110

<a id="10ka___export_inference_graph_"></a>
## 10ka       @ export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_10ka.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_10ka/model.ckpt-559541 --output_directory trained/faster_rcnn_resnet101_coco_2018_01_28_10ka/inference_559541

<a id="20k3___export_inference_graph_"></a>
## 20k3       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k3.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_20k3/model.ckpt-462851 --output_directory trained/faster_rcnn_resnet101_coco_2018_01_28_20k3/inference_462851

<a id="20k3_hs___export_inference_graph_"></a>
## 20k3_hs       @ export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k3_hs.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs/model.ckpt-474099 --output_directory trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs/inference_474099

<a id="20k3_coco___export_inference_graph_"></a>
## 20k3_coco       @ export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k3_coco.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco --output_directory trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco


<a id="python_2___20k3_coco_export_inference_graph_"></a>
### python_2       @ 20k3_coco/export_inference_graph

CUDA_VISIBLE_DEVICES=0 python2 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k3_coco.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco/inference_801515 --output_directory trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco/inference_801515

<a id="20k3_hs_auto___export_inference_graph_"></a>
## 20k3_hs_auto       @ export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k3_hs.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs/ 

<a id="20k4___export_inference_graph_"></a>
## 20k4       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k4.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_20k4/ 

<a id="20k_hs___export_inference_graph_"></a>
## 20k_hs       @ export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k_hs.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_20k_hs/

<a id="20k_coco___export_inference_graph_"></a>
## 20k_coco       @ export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k_coco.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_20k_coco/

<a id="20k7___export_inference_graph_"></a>
## 20k7       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k7.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_20k7 

<a id="20k5___export_inference_graph_"></a>
## 20k5       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k5.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_20k5

<a id="25k7___export_inference_graph_"></a>
## 25k7       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_25k7.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_25k7 

<a id="25k5___export_inference_graph_"></a>
## 25k5       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_25k5.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_25k5

<a id="25k3___export_inference_graph_"></a>
## 25k3       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_25k3.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_25k3

<a id="regularly_spaced___25k3_export_inference_grap_h_"></a>
### regularly_spaced       @ 25k3/export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph2.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_25k3.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_25k3 --export_every=200000 --save_ckpt=0

<a id="1600_static3___export_inference_graph_"></a>
## 1600_static3       @ export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph2.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1600_static3_train.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_1600_static3_train --export_every=200000 --save_ckpt=0

<a id="40k3_coco___export_inference_graph_"></a>
## 40k3_coco       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph2.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_40k3_coco.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_40k3_coco 

<a id="regularly_spaced___40k3_coco_export_inference_graph_"></a>
### regularly_spaced       @ 40k3_coco/export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph2.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_40k3_coco.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_40k3_coco --export_every=200000 --save_ckpt=0

<a id="fiasco___40k3_coco_export_inference_graph_"></a>
### fiasco       @ 40k3_coco/export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_5k.config --trained_checkpoint_prefix trained/faster_rcnn_resnet101_coco_2018_01_28_5k/model.ckpt-620648 --output_directory trained/faster_rcnn_resnet101_coco_2018_01_28/inference5k


<a id="tes_t_"></a>
# test

<a id="list___test_"></a>
## list       @ test

```
model_id    model_type
0   faster_rcnn_resnet101_20K_3_class_only_videos_462K_steps
1   faster_rcnn_resnet101_20K_3_class_INRIA_VOC_COCO_474K_steps
2   faster_rcnn_resnet101_20K_3_class_COCO_801K_steps
3   faster_rcnn_resnet101_pretrained
4   rfcn_pretrained
5   faster_rcnn_resnet101_20K_4_animals_1050K_steps
6   faster_rcnn_resnet101_20K_7_class_1033K_steps
7   faster_rcnn_resnet101_20K_5_class_780K_steps
8   rfcn_20K_7_class_500K_steps
9   rfcn_20K_5_class_500K_steps
10   faster_rcnn_resnet101_25K_7_class_421K_steps
11   faster_rcnn_resnet101_25K_5_class_765K_steps
12   faster_rcnn_resnet101_25K_7_class_1081K_steps
13   faster_rcnn_resnet101_25K_5_class_1559K_steps
14   faster_rcnn_resnet101_40K_3_class_1182K_steps
15   rfcn_40K_3_class_1243K_steps
16   ssd_40K_3_class_215K_steps
17   faster_rcnn_resnet101_25K_3_class_200K_steps
18   faster_rcnn_resnet101_25K_3_class_400K_steps
19   faster_rcnn_resnet101_25K_3_class_600K_steps
20   faster_rcnn_resnet101_25K_3_class_800K_steps
21   faster_rcnn_resnet101_25K_3_class_1000K_steps
22   faster_rcnn_resnet101_25K_static_3_class_400K_steps
23   faster_rcnn_resnet101_25K_static_3_class_800K_steps
24   faster_rcnn_resnet101_25K_static_3_class_1000K_steps
```

<a id="point_grey___list_tes_t_"></a>
### point_grey       @ list/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1224 vis_height=1024 write_det=0 use_ptgrey=1 threaded_mode=1 rgb_mode=2 video_mode=1 write_data=40 save_video=1 fps=12 model_id=0

<a id="ffmpeg___point_grey_list_test_"></a>
#### ffmpeg       @ point_grey/list/test

CUDA_VISIBLE_DEVICES=0 python2 tf_api_test.py model_list=_trained_models.txt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1224 vis_height=1024 write_det=0 use_ffmpeg=1 rgb_mode=2 video_mode=1 write_data=40 save_video=1 fps=12 model_id=0



<a id="pretrained___test_"></a>
## pretrained       @ test

<a id="camera___pretrained_tes_t_"></a>
### camera       @ pretrained/test

CUDA_VISIBLE_DEVICES=2 python3 tf_api_test.py ckpt_path=pre_trained_models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=900 n_classes=90 write_det=0

<a id="only_person___camera_pretrained_test_"></a>
#### only_person       @ camera/pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=pre_trained_models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0 classes_to_include=person,

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=pre_trained_models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0 classes_to_include=person,


<a id="pt_grey___camera_pretrained_test_"></a>
#### Pt_Grey       @ camera/pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=pre_trained_models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1224 vis_height=1024 n_classes=90 write_det=0 classes_to_include=person, use_ptgrey=1 threaded_mode=1 rgb_mode=1 video_mode=1

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=pre_trained_models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1224 vis_height=1024 n_classes=90 write_det=0 classes_to_include=person, use_ptgrey=1 threaded_mode=1 rgb_mode=1 video_mode=1

<a id="10kh___pretrained_tes_t_"></a>
### 10kh       @ pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=pre_trained_models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt list_file_name=acamp10kh_test.txt n_frames=0 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_pretrained/ n_frames=0 batch_size=25 n_classes=90

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=pre_trained_models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt list_file_name=acamp10k_test_mot17.txt n_frames=0 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_pretrained/ n_frames=0 batch_size=25 n_classes=90


<a id="100_on_5k___test_"></a>
## 100_on_5k       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt root_dir=/data/acamp/acamp5k/test/images save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_100_on_5k/ n_frames=0 batch_size=25

<a id="100_on_20k4___test_"></a>
## 100_on_20k4       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=acamp20k4_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_100_on_20k4/ n_frames=0 batch_size=1

<a id="5k___test_"></a>
## 5K       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_5k/inference_536475/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt root_dir=/data/acamp/acamp5k/test/images save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_5k/inference_536475 n_frames=0 batch_size=25

<a id="5k_on_10ka___test_"></a>
## 5K_on_10ka       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_5k/inference_536475/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=acamp10ka_test.txt save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_5k_on_10ka/inference_536475 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_5k/inference_536475/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=acamp10k_test_moose_21_1.txt save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_5k_on_10ka/inference_536475 n_frames=0 batch_size=25

<a id="5k_on_20k4___5k_on_10ka_tes_t_"></a>
### 5k_on_20k4       @ 5K_on_10ka/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_5k/inference_536475/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=acamp20k4_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_5k_on_20k4/ n_frames=0 batch_size=25

<a id="5k_on_train___test_"></a>
## 5k_on_train       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_5k/inference_536475/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=acamp5k_train.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_5k_on_train_inference_536475/ n_frames=0 batch_size=25 n_classes=4

<a id="10k___test_"></a>
## 10k       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10k/inference_746846/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp10k_test.txt save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_10k/ n_frames=0 batch_size=25 n_classes=5

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10k/inference_746846/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp10k_test_mot17.txt save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_10k/ n_frames=0 batch_size=25 n_classes=5

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10k/inference_746846/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp10k_test_moose_21_1.txt save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_10k/ n_frames=0 batch_size=25 n_classes=5

<a id="10k_on_20k3_hs___10k_test_"></a>
### 10k_on_20k3_hs       @ 10k/test

CUDA_VISIBLE_DEVICES=1 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10k/inference_746846/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp20k3_hs_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_10k_on_20k3_hs n_frames=0 batch_size=25 n_classes=5

CUDA_VISIBLE_DEVICES=1 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10k/inference_746846/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp20k3_hs_test_only_h.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_10k_on_20k3_hs n_frames=0 batch_size=25 n_classes=5


<a id="camera___10k_test_"></a>
### camera       @ 10k/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10k/inference_746846/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=5 write_det=0

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10k/inference_746846/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=camera_1 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=5 write_det=0


<a id="unlabeled___10k_test_"></a>
### unlabeled       @ 10k/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10k/inference_746846/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=. n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=5 write_det=0 input_type=videos

<a id="10k_ar_on_10k___test_"></a>
## 10k_ar_on_10k       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/batch_6/faster_rcnn_resnet101_coco_2018_01_28_10k_ar/inference_205110/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp10k_test.txt save_dir=results/batch_6/faster_rcnn_resnet101_coco_2018_01_28_10k_ar/ n_frames=0 batch_size=25 n_classes=5

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/batch_6/faster_rcnn_resnet101_coco_2018_01_28_10k_ar/inference_205110/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp10k_test_moose_21_1.txt save_dir=results/batch_6/faster_rcnn_resnet101_coco_2018_01_28_10k_ar/ n_frames=0 batch_size=25 n_classes=5

zr batch_6_faster_rcnn_resnet101_coco_2018_01_28_10k_ar batch_6/faster_rcnn_resnet101_coco_2018_01_28_10k_ar


<a id="10ka___test_"></a>
## 10ka       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10ka/inference_559541/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=acamp10ka_test.txt save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_10ka/ n_frames=0 batch_size=25

<a id="moose_21_1___10ka_tes_t_"></a>
### moose_21_1       @ 10ka/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10ka/inference_559541/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=acamp10k_test_moose_21_1.txt save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_10ka/ n_frames=0 batch_size=25

<a id="10ka_on_5k___10ka_tes_t_"></a>
### 10ka_on_5k       @ 10ka/test

CUDA_VISIBLE_DEVICES=1 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10ka/inference_559541/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt root_dir=/data/acamp/acamp5k/test/images save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_5k/ n_frames=0 batch_size=25

<a id="10ka_on_20k4___10ka_tes_t_"></a>
### 10ka_on_20k4       @ 10ka/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10ka/inference_559541/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=acamp20k4_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_20k4/ n_frames=0 batch_size=25

<a id="10ka_on_train___test_"></a>
## 10ka_on_train       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_10ka/inference_559541/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=acamp10ka_train.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_10ka_on_train/ n_frames=0 batch_size=25


<a id="20k3___test_"></a>
## 20k3       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3/inference_462851/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt list_file_name=acamp20k3_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851/ n_frames=0 batch_size=25 n_classes=3 show_img=0

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3/inference_462851/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt list_file_name=acamp20k_test_180720.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k3_inference_462851/ n_frames=0 batch_size=25 n_classes=3 show_img=0

<a id="camera___20k3_tes_t_"></a>
### camera       @ 20k3/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3/inference_462851/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=5 write_det=0

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3/inference_462851/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt root_dir=camera_1 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=5 write_det=0


<a id="20k3_hs___test_"></a>
## 20k3_hs       @ test

CUDA_VISIBLE_DEVICES=1 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs/inference_474099/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt list_file_name=acamp20k3_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs_inference_474099/ n_frames=0 batch_size=25 n_classes=3

<a id="20k3_hs_only_h___20k3_hs_test_"></a>
### 20k3_hs_only_h       @ 20k3_hs/test

CUDA_VISIBLE_DEVICES=1 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs/inference_474099/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt list_file_name=acamp20k3_hs_test_only_h.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs_inference_474099_20k3_hs_only_h/ n_frames=0 batch_size=1 n_classes=3

<a id="camera___20k3_hs_test_"></a>
### camera       @ 20k3_hs/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs/inference_474099/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=3 write_det=0

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs/inference_474099/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt root_dir=camera_1 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=3 write_det=0

<a id="only_person___camera_20k3_hs_tes_t_"></a>
#### only_person       @ camera/20k3_hs/test


CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs/inference_474099/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=3 write_det=0 classes_to_include=person,

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_hs/inference_474099/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt root_dir=camera_1 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=3 write_det=0 classes_to_include=person,


<a id="20k3_coco___test_"></a>
## 20k3_coco       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco/inference_801515/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt list_file_name=acamp20k3_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco_inference_801515/ n_frames=0 batch_size=25 n_classes=3

<a id="acamp20k_test_180719___20k3_coco_test_"></a>
### acamp20k_test_180719       @ 20k3_coco/test

CUDA_VISIBLE_DEVICES=1 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco/inference_801515/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt list_file_name=acamp20k_test_180719.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco_inference_801515/ n_frames=0 batch_size=25 n_classes=3

<a id="camera___20k3_coco_test_"></a>
### camera       @ 20k3_coco/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco/inference_801515/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=3 write_det=0

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco/inference_801515/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt root_dir=camera_1 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=3 write_det=0

<a id="mtf___camera_20k3_coco_tes_t_"></a>
#### mtf       @ camera/20k3_coco/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k3_coco/inference_801515/frozen_inference_graph.pb labels_path=data/wildlife_label_map_20k3.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 fullscreen=1 n_classes=90 write_det=0 classes_to_include=person, use_mtf=1 mtf_args=pipeline,v,img_source,u,vp_usb_res,1920x1080

<a id="20k3_on_40k3_coco___test_"></a>
## 20k3_on_40k3_coco       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=2 list_file_name=acamp40k3_test_no_human.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=25 n_classes=3

<a id="20k3_on_20k3_missing___test_"></a>
## 20k3_on_20k3_missing       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=2 list_file_name=acamp20k3_test_missing.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=25 n_classes=3


<a id="20k_hs___test_"></a>
## 20k_hs       @ test

CUDA_VISIBLE_DEVICES=1 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k_hs/inference_765623/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp20k_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k_hs_inference_765623/ n_frames=0 batch_size=25 n_classes=5

<a id="camera___20k_hs_tes_t_"></a>
### camera       @ 20k_hs/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k_hs/inference_765623/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=5 write_det=0

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k_hs/inference_765623/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=camera_1 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=5 write_det=0


<a id="20k_coco___test_"></a>
## 20k_coco       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k_coco/inference_1046148/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp20k_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k_coco_inference_1046148/ n_frames=0 batch_size=25 n_classes=5

<a id="camera___20k_coco_tes_t_"></a>
### camera       @ 20k_coco/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k_coco/inference_1046148/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=5 write_det=0

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k_coco/inference_1046148/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=camera_1 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=5 write_det=0


<a id="20k4___test_"></a>
## 20k4       @ test

CUDA_VISIBLE_DEVICES=1 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k4/inference_1050850/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp20k4_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k4_inference_1050850/ n_frames=0 batch_size=25 n_classes=4


<a id="20k4_on_5k___test_"></a>
## 20k4_on_5k       @ test

CUDA_VISIBLE_DEVICES=1 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k4/inference_1050850/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=/data/acamp/acamp5k/test/images save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k4_inference_1050850/ n_frames=0 batch_size=25 n_classes=4

<a id="20k4_on_train___test_"></a>
## 20k4_on_train       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k4/inference_1050850/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt list_file_name=acamp20k4_train.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k4_on_train_inference_1050850/ n_frames=0 batch_size=25 n_classes=4

<a id="20k7___test_"></a>
## 20k7       @ test

<a id="20k7_on_20k4___20k7_tes_t_"></a>
### 20k7_on_20k4       @ 20k7/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k7 labels_path=data/wildlife_label_map_20k7.pbtxt list_file_name=acamp20k4_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k7_on_20k4/ n_frames=0 batch_size=25 n_classes=7

<a id="unlabeled___20k7_tes_t_"></a>
### unlabeled       @ 20k7/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k7 labels_path=data/wildlife_label_map_20k7.pbtxt root_dir=/data/acamp/test_1808162222 n_frames=0 batch_size=25 show_img=0 save_video=1 n_classes=7 write_det=0 input_type=videos

<a id="20k5___test_"></a>
## 20k5       @ test

<a id="20k5_on_20k4___20k5_tes_t_"></a>
### 20k5_on_20k4       @ 20k5/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k5 labels_path=data/wildlife_label_map_20k5.pbtxt list_file_name=acamp20k4_test.txt root_dir=/data/acamp/acamp20k save_dir=results/faster_rcnn_resnet101_coco_2018_01_28_20k5_on_20k4/ n_frames=0 batch_size=25 n_classes=7

<a id="unlabeled___20k5_tes_t_"></a>
### unlabeled       @ 20k5/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28_20k5 labels_path=data/wildlife_label_map_20k5.pbtxt root_dir=/data/acamp/test_1808162222 n_frames=0 batch_size=25 show_img=0 save_video=1 n_classes=7 write_det=0 input_type=videos

<a id="25k3_on_40k3___test_"></a>
## 25k3_on_40k3       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=17 list_file_name=acamp40k3_test_no_human.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=25 n_classes=3

<a id="25k3_on_training___test_"></a>
## 25k3_on_training       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=17 list_file_name=acamp25k3_train.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=1 n_classes=3

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=21 list_file_name=acamp25k3_train.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=1 n_classes=3

<a id="40k3_coco___test_"></a>
## 40k3_coco       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=14 list_file_name=acamp40k3_test_no_human.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=25 n_classes=3

<a id="40k3_coco_on_20k4_static___test_"></a>
## 40k3_coco_on_20K4_static       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=14 list_file_name=acamp20k4_static_test.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=25 n_classes=3

<a id="40k3_coco_on_training___test_"></a>
## 40k3_coco_on_training       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=14 list_file_name=acamp40k3_coco_train.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=1 n_classes=3

<a id="1600_static3_on_40k3___test_"></a>
## 1600_static3_on_40K3       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=24 list_file_name=acamp40k3_test_no_human.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=1 n_classes=3

<a id="1600_static3_on_train___test_"></a>
## 1600_static3_on_train       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=24 list_file_name=acamp1600_static3_train.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=25 n_classes=3

<a id="1600_static3_on_20k4_static___test_"></a>
## 1600_static3_on_20K4_static       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=24 list_file_name=acamp20k4_static_test.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=1 n_classes=3

<a id="1600_static3_on_20k3_no_human___test_"></a>
## 1600_static3_on_20k3__no_human       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=24 list_file_name=acamp20k3_test_no_human.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=25 n_classes=3

<a id="1600_static3_on_training___test_"></a>
## 1600_static3_on_training       @ test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py model_list=_trained_models.txt model_id=24 list_file_name=acamp1600_static3_train.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=1 n_classes=3

