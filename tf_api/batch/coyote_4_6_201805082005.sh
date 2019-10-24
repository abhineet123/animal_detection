CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=results/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_4.mp4 save_file_name=results/coyote_4_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=results/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_5.mp4 save_file_name=results/coyote_5_det.mp4 n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=1 python3 test.py ckpt_path=results/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_6.mp4 save_file_name=results/coyote_6_det.mp4 n_frames=0 batch_size=25