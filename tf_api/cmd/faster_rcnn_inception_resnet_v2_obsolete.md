<!-- MarkdownTOC -->

- [eval](#eva_l_)
- [export_inference_graph](#export_inference_grap_h_)
    - [5k       @ export_inference_graph](#5k___export_inference_graph_)
    - [failed       @ export_inference_graph](#failed___export_inference_graph_)
        - [evaluation_frozen_graphs       @ failed/export_inference_graph](#evaluation_frozen_graphs___failed_export_inference_grap_h_)
        - [617/A2       @ failed/export_inference_graph](#617_a2___failed_export_inference_grap_h_)
    - [first_training_f-rcnn_inception_v2       @ export_inference_graph](#first_training_f_rcnn_inception_v2___export_inference_graph_)
- [test](#tes_t_)
    - [pretrained       @ test](#pretrained___test_)
        - [camera       @ pretrained/test](#camera___pretrained_tes_t_)
            - [only_person       @ camera/pretrained/test](#only_person___camera_pretrained_test_)
            - [Pt_Grey       @ camera/pretrained/test](#pt_grey___camera_pretrained_test_)
    - [100_on_5k       @ test](#100_on_5k___test_)
    - [5k       @ test](#5k___test_)
    - [evaluation_frozen_graphs       @ test](#evaluation_frozen_graphs___test_)
    - [faster_rcnn_inception_v2_coco_2017_11_08       @ test](#faster_rcnn_inception_v2_coco_2017_11_08___test_)
    - [first_training_f-rcnn_inception_v2       @ test](#first_training_f_rcnn_inception_v2___test_)
        - [grizzly_bear       @ first_training_f-rcnn_inception_v2/test](#grizzly_bear___first_training_f_rcnn_inception_v2_tes_t_)
            - [batch_tests       @ grizzly_bear/first_training_f-rcnn_inception_v2/test](#batch_tests___grizzly_bear_first_training_f_rcnn_inception_v2_test_)
            - [more_bear_videos       @ grizzly_bear/first_training_f-rcnn_inception_v2/test](#more_bear_videos___grizzly_bear_first_training_f_rcnn_inception_v2_test_)
                - [zoo_180529       @ more_bear_videos/grizzly_bear/first_training_f-rcnn_inception_v2/test](#zoo_180529___more_bear_videos_grizzly_bear_first_training_f_rcnn_inception_v2_tes_t_)
                - [zip       @ more_bear_videos/grizzly_bear/first_training_f-rcnn_inception_v2/test](#zip___more_bear_videos_grizzly_bear_first_training_f_rcnn_inception_v2_tes_t_)
        - [moose       @ first_training_f-rcnn_inception_v2/test](#moose___first_training_f_rcnn_inception_v2_tes_t_)
        - [deer       @ first_training_f-rcnn_inception_v2/test](#deer___first_training_f_rcnn_inception_v2_tes_t_)
            - [zoo_180529       @ deer/first_training_f-rcnn_inception_v2/test](#zoo_180529___deer_first_training_f_rcnn_inception_v2_test_)
        - [coyote       @ first_training_f-rcnn_inception_v2/test](#coyote___first_training_f_rcnn_inception_v2_tes_t_)
            - [doc       @ coyote/first_training_f-rcnn_inception_v2/test](#doc___coyote_first_training_f_rcnn_inception_v2_test_)
        - [human       @ first_training_f-rcnn_inception_v2/test](#human___first_training_f_rcnn_inception_v2_tes_t_)
- [evaluation](#evaluatio_n_)
        - [marcin_180608       @ evaluation/](#marcin_180608___evaluation__)
            - [bear       @ marcin_180608/evaluation/](#bear___marcin_180608_evaluation__)
            - [moose       @ marcin_180608/evaluation/](#moose___marcin_180608_evaluation__)
            - [coyote       @ marcin_180608/evaluation/](#coyote___marcin_180608_evaluation__)
- [deer](#dee_r_)

<!-- /MarkdownTOC -->

<a id="eva_l_"></a>
# eval

CUDA_VISIBLE_DEVICES=1 python3 models/research/object_detection/eval.py --logtostderr --pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k/faster_rcnn_inception_5k.config --checkpoint_dir=trained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k --eval_dir=trained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k/eval --num_examples=0

<a id="export_inference_grap_h_"></a>
# export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path trained/first_training_f-rcnn_inception_v2/pipeline.config --trained_checkpoint_prefix trained/first_training_f-rcnn_inception_v2/model.ckpt-140000 --output_directory trained/first_training_f-rcnn_inception_v2/inference

<a id="5k___export_inference_graph_"></a>
## 5k       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_5k.config --trained_checkpoint_prefix trained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k/model.ckpt-200000 --output_directory trained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k/inference_200K


<a id="failed___export_inference_graph_"></a>
## failed       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 export_inference_graph.py  --input_type image_tensor --pipeline_config_path faster_rcnn_inception_resnet_v2_atrous_oid.config --trained_checkpoint_prefix trained/evaluation_frozen_graphs/F-RCNN_Inceptionv2/model.ckpt --output_directory evaluation_frozen_graphs/F-RCNN_Inceptionv2
    `
    NotFoundError (see above for traceback): Key FirstStageFeatureExtractor/InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/beta not found in checkpoint 
    `
    
<a id="evaluation_frozen_graphs___failed_export_inference_grap_h_"></a>
### evaluation_frozen_graphs       @ failed/export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 export_inference_graph.py  --input_type image_tensor --pipeline_config_path trained/faster_rcnn_inception_v2_coco_2017_11_08/pipeline.config --trained_checkpoint_prefix trained/evaluation_frozen_graphs/F-RCNN_Inceptionv2/model.ckpt --output_directory evaluation_frozen_graphs/F-RCNN_Inceptionv2

    `
    InvalidArgumentError (see above for traceback): Assign requires shapes of both tensors to match. lhs shape= [91] rhs shape= [5]
    `
    
<a id="617_a2___failed_export_inference_grap_h_"></a>
### 617/A2       @ failed/export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 models/research/object_detection/export_inference_graph.py  --input_type image_tensor --pipeline_config_path trained/faster_rcnn_inception_v2_coco_2017_11_08/pipeline.config --trained_checkpoint_prefix trained/evaluation_frozen_graphs/F-RCNN_Inceptionv2/model.ckpt --output_directory evaluation_frozen_graphs/F-RCNN_Inceptionv2

<a id="first_training_f_rcnn_inception_v2___export_inference_graph_"></a>
## first_training_f-rcnn_inception_v2       @ export_inference_graph

CUDA_VISIBLE_DEVICES=0 python3 export_inference_graph.py  --input_type image_tensor --pipeline_config_path trained/first_training_f-rcnn_inception_v2/pipeline.config --trained_checkpoint_prefix trained/first_training_f-rcnn_inception_v2/model.ckpt-140000 --output_directory trained/first_training_f-rcnn_inception_v2/inference/

`
ValueError: Protocol message RewriterConfig has no "optimize_tensor_layout" field.
`

<a id="tes_t_"></a>
# test

<a id="pretrained___test_"></a>
## pretrained       @ test

<a id="camera___pretrained_tes_t_"></a>
### camera       @ pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=900 n_classes=90 write_det=0

<a id="only_person___camera_pretrained_test_"></a>
#### only_person       @ camera/pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0 classes_to_include=person,

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0 classes_to_include=person,


<a id="pt_grey___camera_pretrained_test_"></a>
#### Pt_Grey       @ camera/pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1224 vis_height=1024 n_classes=90 write_det=0 classes_to_include=person, use_ptgrey=1 threaded_mode=1 rgb_mode=1 video_mode=1

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=pre_trained_models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1224 vis_height=1024 n_classes=90 write_det=0 classes_to_include=person, use_ptgrey=1 threaded_mode=1 rgb_mode=1 video_mode=1

<a id="100_on_5k___test_"></a>
## 100_on_5k       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt root_dir=/data/acamp/acamp5k/test/images save_dir=results/first_training_f-rcnn_inception_v2_100_on_5k/ n_frames=0 batch_size=25

zr first_training_f-rcnn_inception_v2_on_5k results/first_training_f-rcnn_inception_v2

<a id="5k___test_"></a>
## 5k       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k/inference_200K/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt root_dir=/data/acamp/acamp5k/test/images save_dir=results/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k/inference_200K n_frames=0 batch_size=10

zr faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k_200K results/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28_5k/inference_200K


<a id="evaluation_frozen_graphs___test_"></a>
## evaluation_frozen_graphs       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/evaluation_frozen_graphs/F-RCNN_Inceptionv2/saved_model/saved_model.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det.mp4 n_frames=0

<a id="faster_rcnn_inception_v2_coco_2017_11_08___test_"></a>
## faster_rcnn_inception_v2_coco_2017_11_08       @ test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_inception_v2_coco_2017_11_08/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_det_100.mp4 n_frames=100

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_inception_v2_coco_2017_11_08/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det.mp4 n_frames=0

<a id="first_training_f_rcnn_inception_v2___test_"></a>
## first_training_f-rcnn_inception_v2       @ test

<a id="grizzly_bear___first_training_f_rcnn_inception_v2_tes_t_"></a>
### grizzly_bear       @ first_training_f-rcnn_inception_v2/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det.mp4 n_frames=0

<a id="batch_tests___grizzly_bear_first_training_f_rcnn_inception_v2_test_"></a>
#### batch_tests       @ grizzly_bear/first_training_f-rcnn_inception_v2/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=1

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=5

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=10

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det_batch_30.mp4 n_frames=0 batch_size=30

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det_batch_40.mp4 n_frames=0 batch_size=40

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det_batch_50.mp4 n_frames=0 batch_size=50

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_1_det_batch_55.mp4 n_frames=0 batch_size=55

<a id="more_bear_videos___grizzly_bear_first_training_f_rcnn_inception_v2_test_"></a>
#### more_bear_videos       @ grizzly_bear/first_training_f-rcnn_inception_v2/test

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_2.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_2_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_3.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_3_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_4.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_4_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_5.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_5_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_6.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/grizzly_bear_6_det_batch_25.mp4 n_frames=0 batch_size=25

<a id="zoo_180529___more_bear_videos_grizzly_bear_first_training_f_rcnn_inception_v2_tes_t_"></a>
##### zoo_180529       @ more_bear_videos/grizzly_bear/first_training_f-rcnn_inception_v2/test

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_0.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_0.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_1.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_1.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_2.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_2.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_3.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_3.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_4.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_4.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_8.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_8.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_9.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_9.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_10.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_10.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_11.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_11.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_12.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_12.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_13.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_13.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_14.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_14.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_15.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_15.mkv n_frames=0 batch_size=25


CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_17.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_17.mkv n_frames=0 batch_size=25


CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_18.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_18.mkv n_frames=0 batch_size=25


CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_19.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_19.mkv n_frames=0 batch_size=25


CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/bear_zoo_180529_21.MPG save_file_name=results/first_training_f-rcnn_inception_v2/bear_zoo_180529_21.mkv n_frames=0 batch_size=25

<a id="zip___more_bear_videos_grizzly_bear_first_training_f_rcnn_inception_v2_tes_t_"></a>
##### zip       @ more_bear_videos/grizzly_bear/first_training_f-rcnn_inception_v2/test

zip -r grizzly_bear_2_6_det_batch_25_1805091402.zip grizzly_bear_2_det_batch_25.mp4 grizzly_bear_3_det_batch_25.mp4 grizzly_bear_4_det_batch_25.mp4 grizzly_bear_5_det_batch_25.mp4 grizzly_bear_6_det_batch_25.mp4


<a id="moose___first_training_f_rcnn_inception_v2_tes_t_"></a>
### moose       @ first_training_f-rcnn_inception_v2/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_det.mp4 n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_2.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/moose_2_det.mp4 n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_1.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_1.mkv n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_2.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_2.mkv n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_3.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_3.mkv n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_4.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_4.mkv n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_5.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_5.mkv n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_6.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_6.mkv n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_7.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_7.mkv n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_8.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_8.mkv n_frames=0

<a id="deer___first_training_f_rcnn_inception_v2_tes_t_"></a>
### deer       @ first_training_f-rcnn_inception_v2/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/deer_1_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_2.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/deer_2_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_3.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/deer_3_det.mp4 n_frames=0 batch_size=25

<a id="zoo_180529___deer_first_training_f_rcnn_inception_v2_test_"></a>
#### zoo_180529       @ deer/first_training_f-rcnn_inception_v2/test

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_0.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_0.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_1.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_1.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_2.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_2.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_3.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_3.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_4.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_4.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_5.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_5.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_6.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_6.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_7.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_7.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_8.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_8.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_9.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_9.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_10.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_10.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_zoo_180529_11.MPG save_file_name=results/first_training_f-rcnn_inception_v2/deer_zoo_180529/deer_zoo_180529_11.mkv n_frames=0 batch_size=25

<a id="coyote___first_training_f_rcnn_inception_v2_tes_t_"></a>
### coyote       @ first_training_f-rcnn_inception_v2/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_1.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/coyote_1_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_2.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/coyote_2_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_3.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/coyote_3_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_4.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/coyote_4_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_5.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/coyote_5_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_6.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/coyote_6_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_7.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/coyote_7_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_8.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/coyote_8_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_9.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/coyote_9_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_10.mp4 save_file_name=results/first_training_f-rcnn_inception_v2/coyote_10_det.mp4 n_frames=0 batch_size=25

<a id="doc___coyote_first_training_f_rcnn_inception_v2_test_"></a>
#### doc       @ coyote/first_training_f-rcnn_inception_v2/test

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_1_1.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_1_1.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_1_2.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_1_2.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_1_3.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_1_3.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_3_1.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_3_1.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_3_2.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_3_2.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_3_3.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_3_3.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_3_4.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_3_4.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_4_1.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_4_1.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_4_2.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_4_2.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_4_3.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_4_3.mkv n_frames=0 batch_size=25


<a id="human___first_training_f_rcnn_inception_v2_tes_t_"></a>
### human       @ first_training_f-rcnn_inception_v2/test

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/human save_file_name=results/first_training_f-rcnn_inception_v2/human n_frames=0 batch_size=25


<a id="evaluatio_n_"></a>
# evaluation

CUDA_VISIBLE_DEVICES=1 python3 eval.py --logtostderr --pipeline_config_path=first_training_f-rcnn_inception_v2_pipeline.config --checkpoint_dir=results/first_training_f-rcnn_inception_v2 --eval_dir=results/first_training_f-rcnn_inception_v2/eval --run_once=1

CUDA_VISIBLE_DEVICES=1 python3 wildlife_detection_eval.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt test_path=images/test save_path=images/test_vis n_frames=0 batch_size=25

<a id="marcin_180608___evaluation__"></a>
### marcin_180608       @ evaluation/

<a id="bear___marcin_180608_evaluation__"></a>
#### bear       @ marcin_180608/evaluation/

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_3 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_4 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_5 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_6 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_3 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_4 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_5 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/grizzly_bear_6_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

<a id="moose___marcin_180608_evaluation__"></a>
#### moose       @ marcin_180608/evaluation/

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_3 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_3 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_4 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_5 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_6 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_7 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_8 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_9 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_10 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_12 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_13 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_3 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_7_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_8_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_9_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_11_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_3 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_13_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

<a id="coyote___marcin_180608_evaluation__"></a>
#### coyote       @ marcin_180608/evaluation/

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_3 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=25


<a id="dee_r_"></a>
# deer

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_3 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_4 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_2_2
/data/acamp/marcin_180613/deer_2_4 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_3 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_4_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_4_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_5_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_3 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_4 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_7_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_8_1 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_8_2 save_dir=results/first_training_f-rcnn_inception_v2/ n_frames=0 batch_size=2
