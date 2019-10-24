set -x

### yolo_ssdmo_ssdin       @ 500_static3a/polling

#### type_0       @ yolo_ssdmo_ssdin/500_static3a/polling

python3 polling.py csv_paths=yolov3_weights/acamp500_static3a/latest_on_acamp500_static3a_train_inverted,trained/ssd_mobilenet_v2_coco_2018_03_29_500_static3a/152290_on_inverted,trained/ssd_inception_500_static3a/239435_on_inverted root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted  class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt output_path=trained/500_static3a_yolo_ssdmo_ssdin_polled allow_missing_detections=0 only_sampling=0 conf_thresh=2.9,30.3,34.8 polling_type=0

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py load_dir=trained/500_static3a_yolo_ssdmo_ssdin_polled_type_0 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inv out_prefix=500_static3a load_det=1 save_animation=0 score_thresholds=0:1:0.001 

#### type_1       @ yolo_ssdmo_ssdin/500_static3a/polling

python3 polling.py csv_paths=yolov3_weights/acamp500_static3a/latest_on_acamp500_static3a_train_inverted,trained/ssd_mobilenet_v2_coco_2018_03_29_500_static3a/152290_on_inverted,trained/ssd_inception_500_static3a/239435_on_inverted root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted  class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt output_path=trained/500_static3a_yolo_ssdmo_ssdin_polled allow_missing_detections=0 only_sampling=0 conf_thresh=2.9,30.3,34.8 discard_below_thresh=1 polling_type=1

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py load_dir=trained/500_static3a_yolo_ssdmo_ssdin_polled_type_1 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inv out_prefix=500_static3a load_det=1 save_animation=0 score_thresholds=0:1:0.001 

#### type_2       @ yolo_ssdmo_ssdin/500_static3a/polling

python3 polling.py csv_paths=yolov3_weights/acamp500_static3a/latest_on_acamp500_static3a_train_inverted,trained/ssd_mobilenet_v2_coco_2018_03_29_500_static3a/152290_on_inverted,trained/ssd_inception_500_static3a/239435_on_inverted root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted  class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt output_path=trained/500_static3a_yolo_ssdmo_ssdin_polled allow_missing_detections=0 only_sampling=0 conf_thresh=2.9,30.3,34.8 discard_below_thresh=1 polling_type=2 

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py load_dir=trained/500_static3a_yolo_ssdmo_ssdin_polled_type_2 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inv out_prefix=500_static3a load_det=1 save_animation=0 score_thresholds=0:1:0.001 
