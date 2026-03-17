====================================================
====================================================

### ResNet101

====================================================
====================================================



### grizzly_bear

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_1_det.mp4 n_frames=0

#### batch tests

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=1

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=5

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=10

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_25.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_30.mp4 n_frames=0 batch_size=30

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_40.mp4 n_frames=0 batch_size=40

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_50.mp4 n_frames=0 batch_size=50

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_55.mp4 n_frames=0 batch_size=55

#### more bear videos

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_2.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_2_det_batch_25.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_3.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_3_det_batch_25.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_4.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_4_det_batch_25.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_5.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_5_det_batch_25.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_6.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/grizzly_bear_6_det_batch_25.mp4 n_frames=0 batch_size=255

##### zip

zip -r grizzly_bear_2_6_det_batch_25_1805091402.zip grizzly_bear_2_det_batch_25.mp4 grizzly_bear_3_det_batch_25.mp4 grizzly_bear_4_det_batch_25.mp4 grizzly_bear_5_det_batch_25.mp4 grizzly_bear_6_det_batch_25.mp4


### moose

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/moose_1_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_2.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/moose_2_det.mp4 n_frames=0 batch_size=255

### deer

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/deer_1_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_2.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/deer_2_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_3.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/deer_3_det.mp4 n_frames=0 batch_size=255

### coyote

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_1.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/coyote_1_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_2.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/coyote_2_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_3.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/coyote_3_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_4.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/coyote_4_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_5.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/coyote_5_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_6.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/coyote_6_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_7.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/coyote_7_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_8.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/coyote_8_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_9.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/coyote_9_det.mp4 n_frames=0 batch_size=255

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_10.mp4 save_file_name=results/faster_rcnn_resnet101_coco_2018_01_28/coyote_10_det.mp4 n_frames=0 batch_size=255


# evaluation

CUDA_VISIBLE_DEVICES=1 python3 eval.py --logtostderr --pipeline_config_path=faster_rcnn_resnet101_coco_2018_01_28_pipeline.config --checkpoint_dir=results/faster_rcnn_resnet101_coco_2018_01_28 --eval_dir=results/faster_rcnn_resnet101_coco_2018_01_28/eval --run_once=1

CUDA_VISIBLE_DEVICES=1 python3 wildlife_detection_eval.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt test_path=images/test save_path=images/test_vis n_frames=0 batch_size=255


### marcin_180608

#### bear

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_3 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_4 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_5 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_6 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_3 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_4 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_5 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/grizzly_bear_6_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

#### moose

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_3 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_3 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_4 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_5 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_6 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_7 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_8 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_9 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_10 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_12 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_13 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_3 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_7_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_8_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_9_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_11_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_3 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_13_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

#### coyote

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_3 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

#### deer

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_3 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_4 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_2_2
/data/acamp/marcin_180613/deer_2_4 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_3 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_4_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_4_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_5_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_3 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_4 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_7_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_8_1 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/faster_rcnn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_8_2 save_dir=results/faster_rcnn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25



====================================================
====================================================

### RFCN

====================================================
====================================================


### fiasco

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28_5k/inference_179754/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt list_file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/marcin_180608_list.txt root_dir=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin save_dir=results/rfcn_resnet101_coco_2018_01_28_5k/inference_179754 n_frames=0 batch_size=25

zr rfcn_resnet101_coco_2018_01_28_5k_inference_179754 rfcn_resnet101_coco_2018_01_28_5k/inference_179754

### grizzly_bear

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_det.mp4 n_frames=0

#### batch tests

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=1

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=5

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_10.mp4 n_frames=0 batch_size=10

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_30.mp4 n_frames=0 batch_size=30

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_40.mp4 n_frames=0 batch_size=40

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_50.mp4 n_frames=0 batch_size=50

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_det_batch_55.mp4 n_frames=0 batch_size=55

#### more bear videos

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_2.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_2_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_3.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_3_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_4.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_4_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_5.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_5_det_batch_25.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_6.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_6_det_batch_25.mp4 n_frames=0 batch_size=25

##### zip

zip -r grizzly_bear_2_6_det_batch_25_1805091402.zip grizzly_bear_2_det_batch_25.mp4 grizzly_bear_3_det_batch_25.mp4 grizzly_bear_4_det_batch_25.mp4 grizzly_bear_5_det_batch_25.mp4 grizzly_bear_6_det_batch_25.mp4


### moose

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/moose_1_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_2.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/moose_2_det.mp4 n_frames=0 batch_size=25

### deer

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/deer_1_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_2.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/deer_2_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_3.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/deer_3_det.mp4 n_frames=0 batch_size=25

### coyote

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_1_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_2.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_2_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_3.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_3_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_4.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_4_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_5.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_5_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_6.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_6_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_7.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_7_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_8.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_8_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_9.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_9_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_10.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_10_det.mp4 n_frames=0 batch_size=25


# evaluation

CUDA_VISIBLE_DEVICES=1 python3 eval.py --logtostderr --pipeline_config_path=rfcn_resnet101_coco_2018_01_28_pipeline.config --checkpoint_dir=results/rfcn_resnet101_coco_2018_01_28 --eval_dir=results/rfcn_resnet101_coco_2018_01_28/eval --run_once=1

CUDA_VISIBLE_DEVICES=1 python3 wildlife_detection_eval.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt test_path=images/test save_path=images/test_vis n_frames=0 batch_size=25

# new

## grizzly_bear_1_1

python3 visualize_detections.py src_path=videos/grizzly_bear_1_1 det_path=videos/grizzly_bear_1_1/annotations.csv save_path=videos/grizzly_bear_1_1_vis.avi save_fmt=2

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1_1 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_1.avi csv_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_1.csv map_folder=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_1_map n_frames=0 batch_size=1

parser.add_argument('--gt_path', type=str, help="folder containing GT")
parser.add_argument('--det_path', type=str, help="folder containing detections")
parser.add_argument('--img_path', type=str, help="folder containing images")

rfcn_resnet101_coco_2018_01_28_grizzly_bear_1_1_map

python3 main.py --det_path=../object_detection/results/rfcn/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_1_map --img_path=../object_detection/videos/grizzly_bear_1_1 --gt_path="../object_detection/videos/grizzly_bear_1_1/mAP"

## grizzly_bear_1_2

python3 visualize_detections.py src_path=videos/grizzly_bear_1_2 det_path=videos/grizzly_bear_1_2/annotations.csv save_path=videos/grizzly_bear_1_2_vis.avi save_fmt=2

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1_2 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_2.avi csv_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_2.csv map_folder=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_2_map n_frames=0 batch_size=1

parser.add_argument('--gt_path', type=str, help="folder containing GT")
parser.add_argument('--det_path', type=str, help="folder containing detections")
parser.add_argument('--img_path', type=str, help="folder containing images")

rfcn_resnet101_coco_2018_01_28_grizzly_bear_1_2_map

python3 main.py --det_path=../object_detection/results/rfcn/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_2_map --img_path=../object_detection/videos/grizzly_bear_1_2 --gt_path="../object_detection/videos/grizzly_bear_1_2/mAP"

## grizzly_bear_1_3

python3 visualize_detections.py src_path=videos/grizzly_bear_1_3 det_path=videos/grizzly_bear_1_3/annotations.csv save_path=videos/grizzly_bear_1_3_vis.avi save_fmt=2

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1_3 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_3.avi csv_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_3.csv map_folder=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_3_map n_frames=0 batch_size=1

parser.add_argument('--gt_path', type=str, help="folder containing GT")
parser.add_argument('--det_path', type=str, help="folder containing detections")
parser.add_argument('--img_path', type=str, help="folder containing images")

rfcn_resnet101_coco_2018_01_28_grizzly_bear_1_3_map

python3 main.py --det_path=../object_detection/results/rfcn/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_3_map --img_path=../object_detection/videos/grizzly_bear_1_3 --gt_path="../object_detection/videos/grizzly_bear_1_3/mAP"

## grizzly_bear_1_4

python3 visualize_detections.py src_path=videos/grizzly_bear_1_4 det_path=videos/grizzly_bear_1_4/annotations.csv save_path=videos/grizzly_bear_1_4_vis.avi save_fmt=2

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/grizzly_bear_1_4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_4.avi csv_file_name=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_4.csv map_folder=results/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_4_map n_frames=0 batch_size=1

parser.add_argument('--gt_path', type=str, help="folder containing GT")
parser.add_argument('--det_path', type=str, help="folder containing detections")
parser.add_argument('--img_path', type=str, help="folder containing images")

rfcn_resnet101_coco_2018_01_28_grizzly_bear_1_4_map

python3 main.py --det_path=../object_detection/results/rfcn/rfcn_resnet101_coco_2018_01_28/grizzly_bear_1_4_map --img_path=../object_detection/videos/grizzly_bear_1_4 --gt_path="../object_detection/videos/grizzly_bear_1_4/mAP"

## deer_2_4

python3 visualize_detections.py src_path=videos/deer_2_4 det_path=videos/deer_2_4/annotations.csv save_path=videos/deer_2_4_vis.avi save_fmt=2


CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_2_4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/deer_2_4.avi csv_file_name=results/rfcn_resnet101_coco_2018_01_28/deer_2_4.csv map_folder=results/rfcn_resnet101_coco_2018_01_28/deer_2_4 n_frames=0 batch_size=1


## coyote_doc_1_1

python3 visualize_detections.py src_path=videos/coyote_doc_1_1 det_path=videos/coyote_doc_1_1/annotations.csv save_path=videos/coyote_doc_1_1_vis.avi save_fmt=2

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_1_1 save_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_doc_1_1.avi csv_file_name=results/rfcn_resnet101_coco_2018_01_28/coyote_doc_1_1.csv map_folder=results/rfcn_resnet101_coco_2018_01_28/coyote_doc_1_1_map n_frames=0 batch_size=1

parser.add_argument('--gt_path', type=str, help="folder containing GT")
parser.add_argument('--det_path', type=str, help="folder containing detections")
parser.add_argument('--img_path', type=str, help="folder containing images")

rfcn_resnet101_coco_2018_01_28_coyote_doc_1_1_map

python3 main.py --det_path=../object_detection/results/rfcn/rfcn_resnet101_coco_2018_01_28/coyote_doc_1_1_map --img_path=../object_detection/videos/coyote_doc_1_1 --gt_path="../object_detection/videos/coyote_doc_1_1/mAP"

## horse_1_1

python3 visualize_detections.py src_path=videos/horse_1_1 det_path=videos/horse_1_1/annotations.csv save_path=videos/horse_1_1_vis.avi save_fmt=2

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/horse_1_1 save_file_name=results/rfcn_resnet101_coco_2018_01_28/horse_1_1.avi csv_file_name=results/rfcn_resnet101_coco_2018_01_28/horse_1_1.csv map_folder=results/rfcn_resnet101_coco_2018_01_28/horse_1_1_map n_frames=0 batch_size=1

parser.add_argument('--gt_path', type=str, help="folder containing GT")
parser.add_argument('--det_path', type=str, help="folder containing detections")
parser.add_argument('--img_path', type=str, help="folder containing images")

rfcn_resnet101_coco_2018_01_28_horse_1_1_map

python3 main.py --det_path=../object_detection/results/rfcn/rfcn_resnet101_coco_2018_01_28/horse_1_1_map --img_path=../object_detection/videos/horse_1_1 --gt_path="../object_detection/videos/horse_1_1/mAP"

## cow_1

python3 visualize_detections.py src_path=videos/cow_1 det_path=videos/cow_1/annotations.csv save_path=videos/cow_1_vis.avi save_fmt=2

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/cow_1 save_file_name=results/rfcn_resnet101_coco_2018_01_28/cow_1.avi csv_file_name=results/rfcn_resnet101_coco_2018_01_28/cow_1.csv map_folder=results/rfcn_resnet101_coco_2018_01_28/cow_1_map n_frames=0 batch_size=1

parser.add_argument('--gt_path', type=str, help="folder containing GT")
parser.add_argument('--det_path', type=str, help="folder containing detections")
parser.add_argument('--img_path', type=str, help="folder containing images")

rfcn_resnet101_coco_2018_01_28_cow_1_map

python3 main.py --det_path=../object_detection/results/rfcn/rfcn_resnet101_coco_2018_01_28/cow_1_map --img_path=../object_detection/videos/cow_1 --gt_path="../object_detection/videos/cow_1/mAP"


### marcin_180608

#### bear

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_3 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_4 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_5 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_6 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_3 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_4 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_5 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/grizzly_bear_6_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

#### moose

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_3 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_3 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_4 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_5 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_6 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_7 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_8 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_9 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_10 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_12 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_13 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_3 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_7_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_8_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_9_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_11_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_3 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_13_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

#### coyote

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_3 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=25


# deer

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_3 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_1_4 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_2_2
/data/acamp/marcin_180613/deer_2_4 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2



CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_3_3 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_4_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_4_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_5_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_3 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_6_4 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_7_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_8_1 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=/data/acamp/marcin_180613/deer_8_2 save_dir=results/rfcn_resnet101_coco_2018_01_28/ n_frames=0 batch_size=2
