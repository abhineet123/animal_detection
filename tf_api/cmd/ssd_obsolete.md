<!-- MarkdownTOC -->

- [export_inference_graph](#export_inference_grap_h_)
    - [5k       @ export_inference_graph](#5k___export_inference_graph_)
        - [200K       @ 5k/export_inference_graph](#200k___5k_export_inference_grap_h_)
    - [10k       @ export_inference_graph](#10k___export_inference_graph_)
        - [damaged       @ 10k/export_inference_graph](#damaged___10k_export_inference_graph_)
    - [40k3_coco       @ export_inference_graph](#40k3_coco___export_inference_graph_)
- [test](#tes_t_)
    - [pretrained       @ test](#pretrained___test_)
        - [camera       @ pretrained/test](#camera___pretrained_tes_t_)
            - [only_person       @ camera/pretrained/test](#only_person___camera_pretrained_test_)
            - [Pt_Grey       @ camera/pretrained/test](#pt_grey___camera_pretrained_test_)
    - [100_on_5k       @ test](#100_on_5k___test_)
    - [5K       @ test](#5k___test_)
        - [200K       @ 5K/test](#200k___5k_tes_t_)
    - [10K       @ test](#10k___test_)
        - [camera       @ 10K/test](#camera___10k_test_)
    - [40k3_coco       @ test](#40k3_coco___test_)
    - [40k3_coco_on_20K4_static       @ test](#40k3_coco_on_20k4_static___test_)
        - [grizzly_bear       @ 40k3_coco_on_20K4_static/test](#grizzly_bear___40k3_coco_on_20k4_static_tes_t_)
            - [batch_tests       @ grizzly_bear/40k3_coco_on_20K4_static/test](#batch_tests___grizzly_bear_40k3_coco_on_20k4_static_test_)
            - [more_bear_videos       @ grizzly_bear/40k3_coco_on_20K4_static/test](#more_bear_videos___grizzly_bear_40k3_coco_on_20k4_static_test_)
                - [zip       @ more_bear_videos/grizzly_bear/40k3_coco_on_20K4_static/test](#zip___more_bear_videos_grizzly_bear_40k3_coco_on_20k4_static_tes_t_)
        - [moose       @ 40k3_coco_on_20K4_static/test](#moose___40k3_coco_on_20k4_static_tes_t_)
        - [deer       @ 40k3_coco_on_20K4_static/test](#deer___40k3_coco_on_20k4_static_tes_t_)
        - [coyote       @ 40k3_coco_on_20K4_static/test](#coyote___40k3_coco_on_20k4_static_tes_t_)
- [evaluation](#evaluatio_n_)
        - [marcin_180608       @ evaluation/](#marcin_180608___evaluation__)
            - [bear       @ marcin_180608/evaluation/](#bear___marcin_180608_evaluation__)
            - [moose       @ marcin_180608/evaluation/](#moose___marcin_180608_evaluation__)
            - [coyote       @ marcin_180608/evaluation/](#coyote___marcin_180608_evaluation__)
    - [deer       @ evaluation](#deer___evaluation_)

<!-- /MarkdownTOC -->


<a id="export_inference_grap_h_"></a>
# export_inference_graph
    
CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/ssd_inception_v2_coco_2017_11_17/pipeline.config --trained_checkpoint_prefix trained/ssd_inception_v2_coco_2017_11_17/model.ckpt-200000 --output_directory trained/ssd_inception_v2_coco_2017_11_17/inference



<a id="5k___export_inference_graph_"></a>
## 5k       @ export_inference_graph

<a id="200k___5k_export_inference_grap_h_"></a>
### 200K       @ 5k/export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_5k.config --trained_checkpoint_prefix trained/ssd_inception_v2_coco_2017_11_17_5k/model.ckpt-200000 --output_directory trained/ssd_inception_v2_coco_2017_11_17_5k/inference_200K

<a id="10k___export_inference_graph_"></a>
## 10k       @ export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_10k.config --trained_checkpoint_prefix trained/ssd_inception_v2_coco_2017_11_17_10k/model.ckpt-200000 --output_directory trained/ssd_inception_v2_coco_2017_11_17_10k/inference_200K

zr ssd_inception_v2_coco_2017_11_17_10k_inference_200K trained/ssd_inception_v2_coco_2017_11_17_10k/inference_200K

<a id="damaged___10k_export_inference_graph_"></a>
### damaged       @ 10k/export_inference_graph

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_10k.config --trained_checkpoint_prefix trained/ssd_inception_v2_coco_2017_11_17_10k_damaged/model.ckpt-200000 --output_directory trained/ssd_inception_v2_coco_2017_11_17_10k_damaged/inference_200K

zr ssd_inception_v2_coco_2017_11_17_10k_damaged_inference_200K trained/ssd_inception_v2_coco_2017_11_17_10k_damaged/inference_200K

<a id="40k3_coco___export_inference_graph_"></a>
## 40k3_coco       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_40k3.config --trained_checkpoint_prefix trained/ssd_inception_v2_coco_2017_11_17_40k3_coco

<a id="tes_t_"></a>
# test

<a id="pretrained___test_"></a>
## pretrained       @ test

<a id="camera___pretrained_tes_t_"></a>
### camera       @ pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=900 n_classes=90 write_det=0

<a id="only_person___camera_pretrained_test_"></a>
#### only_person       @ camera/pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0 classes_to_include=person,

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0 classes_to_include=person,


<a id="pt_grey___camera_pretrained_test_"></a>
#### Pt_Grey       @ camera/pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1224 vis_height=1024 n_classes=90 write_det=0 classes_to_include=person, use_ptgrey=1 threaded_mode=1 rgb_mode=2 video_mode=1

<a id="100_on_5k___test_"></a>
## 100_on_5k       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt root_dir=/data/acamp/acamp5k/test/images save_dir=results/ssd_inception_v2_coco_2017_11_17_100_on_5k/ n_frames=0 batch_size=1

<a id="5k___test_"></a>
## 5K       @ test

<a id="200k___5k_tes_t_"></a>
### 200K       @ 5K/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17_5k/inference_200K/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=data/acamp5k_test.txt root_dir=/data/acamp/acamp5k/test/images save_dir=results/ssd_inception_v2_coco_2017_11_17_5k n_frames=0 batch_size=25

<a id="10k___test_"></a>
## 10K       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17_10k/inference_200K/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=/data/acamp/acamp5k/test/images save_dir=results/ssd_inception_v2_coco_2017_11_17_10k n_frames=0 batch_size=25

<a id="camera___10k_test_"></a>
### camera       @ 10K/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17_10k/inference_200K/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=720 n_classes=5 write_det=0

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17_10k/inference_200K/frozen_inference_graph.pb labels_path=data/wildlife_label_map_10k.pbtxt root_dir=camera_1 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=720 n_classes=5 write_det=0


<a id="40k3_coco___test_"></a>
## 40k3_coco       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py model_list=trained_models.txt model_id=16 list_file_name=acamp40k3_test_no_human.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=25 n_classes=3

<a id="40k3_coco_on_20k4_static___test_"></a>
## 40k3_coco_on_20K4_static       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py model_list=trained_models.txt model_id=16 list_file_name=acamp20k4_static_test.txt root_dir=/data/acamp/acamp20k n_frames=0 batch_size=1 n_classes=3

<a id="grizzly_bear___40k3_coco_on_20k4_static_tes_t_"></a>
### grizzly_bear       @ 40k3_coco_on_20K4_static/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_1_det.mp4 n_frames=0

<a id="batch_tests___grizzly_bear_40k3_coco_on_20k4_static_test_"></a>
#### batch_tests       @ grizzly_bear/40k3_coco_on_20K4_static/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_1_det_batch_1.mp4 n_frames=0 batch_size=1

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_1_det_batch_2.mp4 n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_1_det_batch_5.mp4 n_frames=0 batch_size=5

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=10

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_1_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_1_det_batch_30.mp4 n_frames=0 batch_size=30

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_1_det_batch_40.mp4 n_frames=0 batch_size=40

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_1_det_batch_50.mp4 n_frames=0 batch_size=50

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_1_det_batch_55.mp4 n_frames=0 batch_size=55

<a id="more_bear_videos___grizzly_bear_40k3_coco_on_20k4_static_test_"></a>
#### more_bear_videos       @ grizzly_bear/40k3_coco_on_20K4_static/test

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_2.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_2_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_3.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_3_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_4.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_4_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_5.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_5_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_6.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/grizzly_bear_6_det_batch_25.mp4 n_frames=0 batch_size=25

<a id="zip___more_bear_videos_grizzly_bear_40k3_coco_on_20k4_static_tes_t_"></a>
##### zip       @ more_bear_videos/grizzly_bear/40k3_coco_on_20K4_static/test

zip -r grizzly_bear_2_6_det_batch_25_1805091402.zip grizzly_bear_2_det_batch_25.mp4 grizzly_bear_3_det_batch_25.mp4 grizzly_bear_4_det_batch_25.mp4 grizzly_bear_5_det_batch_25.mp4 grizzly_bear_6_det_batch_25.mp4


<a id="moose___40k3_coco_on_20k4_static_tes_t_"></a>
### moose       @ 40k3_coco_on_20K4_static/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/moose_1_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_2.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/moose_2_det.mp4 n_frames=0 batch_size=25

<a id="deer___40k3_coco_on_20k4_static_tes_t_"></a>
### deer       @ 40k3_coco_on_20K4_static/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/deer_1_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_2.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/deer_2_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_3.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/deer_3_det.mp4 n_frames=0 batch_size=25

<a id="coyote___40k3_coco_on_20k4_static_tes_t_"></a>
### coyote       @ 40k3_coco_on_20K4_static/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_1.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/coyote_1_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_2.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/coyote_2_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_3.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/coyote_3_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_4.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/coyote_4_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_5.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/coyote_5_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_6.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/coyote_6_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_7.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/coyote_7_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_8.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/coyote_8_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_9.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/coyote_9_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_10.mp4 save_file_name=results/ssd_inception_v2_coco_2017_11_17/coyote_10_det.mp4 n_frames=0 batch_size=25


<a id="evaluatio_n_"></a>
# evaluation

CUDA_VISIBLE_DEVICES=1 python3 eval.py --logtostderr --pipeline_config_path=ssd_inception_v2_coco_2017_11_17_pipeline.config --checkpoint_dir=results/ssd_inception_v2_coco_2017_11_17 --eval_dir=results/ssd_inception_v2_coco_2017_11_17/eval --run_once=1

CUDA_VISIBLE_DEVICES=1 python3 wildlife_detection_eval.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt test_path=images/test save_path=images/test_vis n_frames=0 batch_size=25


<a id="marcin_180608___evaluation__"></a>
### marcin_180608       @ evaluation/

<a id="bear___marcin_180608_evaluation__"></a>
#### bear       @ marcin_180608/evaluation/

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_3 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_4 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_5 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_6 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_3 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_4 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_5 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/grizzly_bear_6_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

<a id="moose___marcin_180608_evaluation__"></a>
#### moose       @ marcin_180608/evaluation/

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_3 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_3 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_4 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_5 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_6 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_7 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_8 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_9 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_10 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_12 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_13 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_3 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_7_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_8_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_9_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_11_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_3 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_13_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

<a id="coyote___marcin_180608_evaluation__"></a>
#### coyote       @ marcin_180608/evaluation/

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_3 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=25


<a id="deer___evaluation_"></a>
## deer       @ evaluation

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_3 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_4 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_2_2
/data/acamp/marcin_180613/deer_2_4 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_3 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_4_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_4_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_5_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_3 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_4 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_7_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_8_1 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/ssd_inception_v2_coco_2017_11_17/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_8_2 save_dir=results/ssd_inception_v2_coco_2017_11_17/ n_frames=0 batch_size=2

