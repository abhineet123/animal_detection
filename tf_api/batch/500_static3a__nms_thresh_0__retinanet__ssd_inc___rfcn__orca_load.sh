# orca

## retinanet_resnet50_v1_500_static3a

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=1 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0 n_threads=4

## ssd_inception_500_static3a

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=1 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0 n_threads=4

## rfcn_resnet101_500_static3a

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=1 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0 n_threads=4

