# 1k8_vid_even_min_1 orca 

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=6 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1 allow_seq_skipping=1 

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=3 show_img=0 n_classes=8  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1 allow_seq_skipping=1 n_threads=1



