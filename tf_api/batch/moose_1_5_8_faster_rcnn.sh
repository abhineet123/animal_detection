
CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_5.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_5.mkv n_frames=0

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_6.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_6.mkv n_frames=0

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_7.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_7.mkv n_frames=0

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_8.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_8.mkv n_frames=0
