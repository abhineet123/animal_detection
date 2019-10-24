# grs 10k6_vid_entire_seq

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=0 allow_seq_skipping=1 n_threads=4

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=2 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=0 allow_seq_skipping=1 n_threads=4

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=0 allow_seq_skipping=1 n_threads=4

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/latest_on_acamp10k6_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 class_agnostic=0 allow_seq_skipping=1 n_threads=4






