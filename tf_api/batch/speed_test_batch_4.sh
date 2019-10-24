CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=4 --weights=yolov3_weights/acamp20k6_5_24/backup270.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_inverted_yolov3_pt.txt

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=4 show_img=0 n_classes=6  eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 load_dir=trained/retinanet_resnet50_v1_20k6_5/8320_on_inverted combine_sequences=1 input_size=416x416

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_mobilenet_v2_coco_2018_03_29_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=4 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001  combine_sequences=1 input_size=416x416

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/ssd_inception_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=4 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_inception_v2_coco_2017_11_17/ssd_inception_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 combine_sequences=1 input_size=416x416

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/rfcn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=4 show_img=0 n_classes=6  eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/rfcn_resnet101_coco_2018_01_28/rfcn_resnet101_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 combine_sequences=1 input_size=416x416

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_resnet101_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=4 show_img=0 n_classes=6  eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_resnet101_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 combine_sequences=1 input_size=416x416

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/faster_rcnn_inception_resnet_v2_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=4 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/faster_rcnn_inception_resnet_v2_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 combine_sequences=1 input_size=416x416








