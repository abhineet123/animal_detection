### moose

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/moose_1_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_2.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/moose_2_det.mp4 n_frames=0 batch_size=25

### deer

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_1.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/deer_1_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_2.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/deer_2_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/rfcn_resnet101_coco_2018_01_28/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/deer_3.mp4 save_file_name=results/rfcn_resnet101_coco_2018_01_28/deer_3_det.mp4 n_frames=0 batch_size=25