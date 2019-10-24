# 1k8_vid_even_min_1 grs

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1 allow_seq_skipping=1

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1 allow_seq_skipping=1

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=32 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1 allow_seq_skipping=1

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1 allow_seq_skipping=1 n_threads=0

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=128 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1 allow_seq_skipping=1 n_threads=4

CUDA_VISIBLE_DEVICES=2 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_4k8.pbtxt n_frames=0 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_8_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/1k8_vid_even_min_1_24_mp/latest_on_1k8_vid_even_min_1_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv score_thresholds=0:1:0.001 class_agnostic=1 allow_seq_skipping=1 n_threads=1







