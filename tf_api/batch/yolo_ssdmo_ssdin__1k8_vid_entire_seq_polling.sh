set -x

### yolo_ssdmo_ssdin       @ 1k8_vid_entire_seq/polling

#### type_0       @ yolo_ssdmo_ssdin/1k8_vid_entire_seq/polling

python3 polling.py csv_paths=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest_on_acamp1k8_vid_entire_seq_inv,trained/ssd_inception_1k8_vid_entire_seq/347637_on_inverted,trained/ssd_mobilenet_v2_1k8_vid_entire_seq/154686_on_inverted root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv  class_names_path=../labelling_tool/data/predefined_classes_4k8.txt output_path=trained/1k8_vid_entire_seq_yolo_ssdmo_ssdin_polled allow_missing_detections=0 only_sampling=0 conf_thresh=2.9,30.3,34.8 polling_type=0

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py load_dir=trained/1k8_vid_entire_seq_yolo_ssdmo_ssdin_polled_type_0 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inv out_prefix=1k8_vid_entire_seq load_det=1 save_animation=0 score_thresholds=0:1:0.001 

#### type_1       @ yolo_ssdmo_ssdin/1k8_vid_entire_seq/polling

python3 polling.py csv_paths=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest_on_acamp1k8_vid_entire_seq_inv,trained/ssd_inception_1k8_vid_entire_seq/347637_on_inverted,trained/ssd_mobilenet_v2_1k8_vid_entire_seq/154686_on_inverted root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv  class_names_path=../labelling_tool/data/predefined_classes_4k8.txt output_path=trained/1k8_vid_entire_seq_yolo_ssdmo_ssdin_polled allow_missing_detections=0 only_sampling=0 conf_thresh=2.9,30.3,34.8 discard_below_thresh=1 polling_type=1

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py load_dir=trained/1k8_vid_entire_seq_yolo_ssdmo_ssdin_polled_type_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inv out_prefix=1k8_vid_entire_seq load_det=1 save_animation=0 score_thresholds=0:1:0.001 

#### type_2       @ yolo_ssdmo_ssdin/1k8_vid_entire_seq/polling

python3 polling.py csv_paths=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest_on_acamp1k8_vid_entire_seq_inv,trained/ssd_inception_1k8_vid_entire_seq/347637_on_inverted,trained/ssd_mobilenet_v2_1k8_vid_entire_seq/154686_on_inverted root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv  class_names_path=../labelling_tool/data/predefined_classes_4k8.txt output_path=trained/1k8_vid_entire_seq_yolo_ssdmo_ssdin_polled allow_missing_detections=0 only_sampling=0 conf_thresh=2.9,30.3,34.8 discard_below_thresh=1 polling_type=2 

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py load_dir=trained/1k8_vid_entire_seq_yolo_ssdmo_ssdin_polled_type_2 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inv out_prefix=1k8_vid_entire_seq load_det=1 save_animation=0 score_thresholds=0:1:0.001 
