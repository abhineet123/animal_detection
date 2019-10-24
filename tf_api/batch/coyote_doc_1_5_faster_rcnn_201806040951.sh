
CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_1_1.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_1_1.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_1_2.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_1_2.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_1_3.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_1_3.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_3_1.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_3_1.mkv n_frames=0 batch_size=25

CUDA_VISIBLE_DEVICES=0 python3 test.py ckpt_path=trained/first_training_f-rcnn_inception_v2/inference/frozen_inference_graph.pb labels_path=data/wildlife_label_map.pbtxt file_name=videos/coyote_doc_3_2.mkv save_file_name=results/first_training_f-rcnn_inception_v2/coyote_doc_3_2.mkv n_frames=0 batch_size=25
