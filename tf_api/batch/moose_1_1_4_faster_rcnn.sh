
CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_1.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_1.mkv n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_2.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_2.mkv n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_3.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_3.mkv n_frames=0

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/moose_1_4.mkv save_file_name=results/first_training_f-rcnn_inception_v2/moose_1_4.mkv n_frames=0
