CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/ssd_inception_20k6_5/51446_on_nms_thresh_0_inv_only_vid out_postfix=nms_thresh_0_inv_only_vid load_det=1 n_threads=4 score_thresholds=0:1:0.001 class_agnostic=1 inference_dir=inference_nms_thresh_0

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_coco_2018_03_29_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 load_dir=trained/ssd_mobilenet_v2_coco_2018_03_29_20k6_5/213946_on_nms_thresh_0_inv_only_vid out_postfix=nms_thresh_0_inv_only_vid load_det=1 n_threads=4 score_thresholds=0:1:0.001 class_agnostic=1 inference_dir=inference_nms_thresh_0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 n_threads=4 load_dir=yolov3_weights/acamp20k6_5_24/backup270_on_acamp20k6_5_inverted_yolov3_pt/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid out_postfix=inv_only_vid score_thresholds=0:1:0.001 class_agnostic=1





