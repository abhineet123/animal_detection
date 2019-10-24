# grs

## faster_rcnn_inception_resnet_v2_500_static3a

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=1 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0 n_threads=4

## faster_rcnn_resnet101_500_static3a

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=1 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0 n_threads=4

## ssd_mobilenet_v2_coco_2018_03_29_500_static3a

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=1 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0 n_threads=4

## faster_rcnn_nas_500_static3a

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_nas_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/faster_rcnn_nas_coco_2018_01_28/faster_rcnn_nas_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=1 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0 n_threads=4
