import glob
import os, sys
import shutil
import pickle, copy
import time
import multiprocessing
import functools
from contextlib import closing

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pformat, pprint

from prettytable import PrettyTable
from tabulate import tabulate

if str(sys.version).startswith('2'):
    import map_utils as utils
else:
    sys.path.append("..")
    import mAP.map_utils as utils

sys.path.append('..')
from tf_api.utilities import resizeAR, sortKey, computeThreshRecPrec


def evaluate(seq_paths, gt_classes, gt_path_list=(), det_path_list=(),
             _gt_data_dict=None, raw_det_data_dict=None, pkl_files_path='pkl_files', save_dir='',
             vis_out_fname='', out_fname='', save_file_res='1920x1080', vid_fmt='H264,30,mkv', iou_thresh=0,
             save_animation=0, show_animation=0, show_text=0, show_stats=0, show_gt=0, show_only_tp=0,
             draw_plot=0, delete_tmp_files=0, write_summary=1, eval_result_dict=None, compute_opt=0,
             score_thresholds=(0.5,), results_path='', rec_ratios=(), wt_avg=0, n_threads=0):
    n_score_thresholds = len(score_thresholds)
    score_thresh = score_thresholds[0]

    score_thresholds = np.asarray(score_thresholds).squeeze()

    save_w, save_h = [int(x) for x in save_file_res.split('x')]

    if not results_path:
        if out_fname:
            results_path = os.path.join(os.path.dirname(out_fname), os.path.splitext(os.path.basename(out_fname))[0])
        else:
            results_path = 'results'

    codec, fps, vid_ext = vid_fmt.split(',')
    fps = int(fps)
    fourcc = cv2.VideoWriter_fourcc(*codec)

    enable_animation = save_animation or show_animation

    n_seq = len(seq_paths)

    if _gt_data_dict is None:
        if n_seq != len(seq_paths):
            raise IOError(
                'Mismatch between the no. of image ({})  and GT ({}) sequences'.format(n_seq, len(seq_paths)))

    if raw_det_data_dict is None:
        if n_seq < len(det_path_list):
            print(
                'Mismatch between the no. of image ({}) and detection ({}) sequences'.format(n_seq, len(det_path_list)))
            det_path_list = det_path_list[:n_seq]

    seq_name_list = [os.path.basename(x) for x in seq_paths]

    if not os.path.exists(pkl_files_path):  # if it doesn't exist already
        os.makedirs(pkl_files_path)

    plots_out_dir = results_path
    if draw_plot:
        print('Saving plots to: {}'.format(plots_out_dir))
        if not os.path.isdir(plots_out_dir):
            os.makedirs(plots_out_dir)
    # if save_animation:
    #     print('Saving animation to: {}'.format(anim_out_dir))
    #     os.makedirs(anim_out_dir)

    # print('gt_path_list: ', gt_path_list)
    # print('det_path_list: ', det_path_list)
    # gt_counter_per_class_loaded = 0
    # gt_counter_per_class_path = os.path.join(pkl_files_path, 'gt_counter_per_class.pkl')
    # if os.path.exists(gt_counter_per_class_path):
    #     with open(gt_counter_per_class_path, 'rb') as f:
    #         gt_counter_per_class = pickle.load(f)
    #         gt_counter_per_class_loaded = 1

    # max_gt_path_len = max([len(x) for x in gt_path_list])
    gt_data_dict_path = ''
    if _gt_data_dict is not None:
        gt_data_dict_loaded = 1
        gt_data_dict = copy.deepcopy(_gt_data_dict)
        # for seq_name in gt_data_dict:
        #     if seq_name == 'counter_per_class':
        #         continue
        #     seq_gt_data_dict = gt_data_dict[seq_name]
        #     print('seq_gt_data_dict: ', seq_gt_data_dict)
        #     for filename in seq_gt_data_dict:
        #         for _dict in seq_gt_data_dict[filename]:
        #             print('_dict: ', _dict)
        #             _dict["used"] = False
        # print('gt_data_dict: ', gt_data_dict)
    else:
        # gt_data_dict_path = os.path.join(pkl_files_path, 'gt_data_dict.pkl')
        # if os.path.exists(gt_data_dict_path):
        #     print('Loading GT data from {}'.format(gt_data_dict_path))
        #     with open(gt_data_dict_path, 'rb') as f:
        #         gt_data_dict = pickle.load(f)
        #         gt_data_dict_loaded = 1
        # else:
        gt_data_dict = {}
        gt_data_dict_loaded = 0
        print('Generating GT data...')

    # df_dets_path = os.path.join(pkl_files_path, 'df_dets.pkl')
    # df_dets_loaded = 0
    # if os.path.exists(df_dets_path):
    #     print('Loading detection raw data from {}'.format(df_dets_path))
    #     with open(df_dets_path, 'rb') as f:
    #         df_dets = pickle.load(f)
    #         df_dets_loaded = 1
    # else:
    #     df_dets = {}

    if raw_det_data_dict is not None:
        raw_det_data_dict_loaded = 1
        # det_data_dict_path = ''
    else:
        raw_det_data_dict = {}
        raw_det_data_dict_loaded = 0
        # det_data_dict_path = os.path.join(pkl_files_path, 'det_data_dict.pkl')
    # max_det_path_len = max([len(x) for x in det_path_list])

    # if os.path.exists(det_data_dict_path):
    #     print('Loading detection data from {}'.format(det_data_dict_path))
    #     with open(det_data_dict_path, 'rb') as f:
    #         det_data_dict = pickle.load(f)
    #         det_data_dict_loaded = 1
    #         raw_det_data_dict_loaded = 1
    # else:
    det_data_dict = {}
    # det_data_dict_loaded = 0

    if not raw_det_data_dict_loaded:
        print('Generating detection data...')

    # print('raw_det_data_dict: ', raw_det_data_dict)
    # print('raw_det_data_dict_loaded: ', raw_det_data_dict_loaded)

    _pause = 0
    gt_counter_per_class = {}
    gt_start_t = time.time()
    for seq_idx, _gt_path in enumerate(gt_path_list):

        if gt_data_dict_loaded and raw_det_data_dict_loaded:
            break

        gt_path = os.path.join(_gt_path, 'annotations.csv')

        if not os.path.isfile(gt_path):
            raise IOError('GT file: {} does not exist'.format(gt_path))

        seq_gt_data_dict = {}

        sys.stdout.write('\rProcessing sequence {:d}/{:d}: {:s} '.format(
            seq_idx + 1, n_seq, gt_path))
        sys.stdout.flush()

        seq_name = seq_name_list[seq_idx]
        seq_path = seq_paths[seq_idx]
        gt_seq_name = os.path.basename(_gt_path)

        if not gt_data_dict_loaded:
            df_gt = pd.read_csv(gt_path)
            seq_gt_data_dict['csv'] = gt_path
            for _, row in df_gt.iterrows():
                filename = row['filename']

                file_path = os.path.join(seq_path, filename)

                xmin = float(row['xmin'])
                ymin = float(row['ymin'])
                xmax = float(row['xmax'])
                ymax = float(row['ymax'])
                gt_class = row['class']

                # bbox = str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax)
                bbox = [xmin, ymin, xmax, ymax]

                if file_path not in seq_gt_data_dict:
                    seq_gt_data_dict[file_path] = []

                seq_gt_data_dict[file_path].append(
                    {'class_name': gt_class, 'bbox': bbox, 'used': False, 'matched': False})

                if gt_class in gt_counter_per_class:
                    gt_counter_per_class[gt_class] += 1
                else:
                    gt_counter_per_class[gt_class] = 1
            gt_data_dict[seq_path] = seq_gt_data_dict

        if not raw_det_data_dict_loaded:
            det_path = det_path_list[seq_idx]
            det_seq_name = os.path.splitext(os.path.basename(det_path))[0]
            if det_seq_name != seq_name or gt_seq_name != seq_name:
                raise IOError('Mismatch between GT, detection and image sequences: {}, {}, {}'.format(
                    gt_seq_name, det_seq_name, seq_name))

            df_det = pd.read_csv(det_path)
            bounding_boxes = []
            for _, row in df_det.iterrows():
                filename = row['filename']

                file_path = os.path.join(seq_path, filename)
                xmin = float(row['xmin'])
                ymin = float(row['ymin'])
                xmax = float(row['xmax'])
                ymax = float(row['ymax'])
                gt_class = row['class']

                try:
                    confidence = row['confidence']
                except:
                    confidence = 1.0

                bounding_boxes.append(
                    {"class": gt_class,
                     "confidence": confidence,
                     "file_path": file_path,
                     "bbox": [xmin, ymin, xmax, ymax]}
                )
            raw_det_data_dict[seq_path] = bounding_boxes

    sys.stdout.write('\n')
    sys.stdout.flush()

    if not gt_data_dict_loaded:
        gt_data_dict['counter_per_class'] = gt_counter_per_class

    gt_end_t = time.time()
    if not (gt_data_dict_loaded and raw_det_data_dict_loaded):
        print('Time taken: {} sec'.format(gt_end_t - gt_start_t))

    gt_counter_per_class = gt_data_dict['counter_per_class']

    # print('gt_data_dict_loaded: ', gt_data_dict_loaded)
    # print('det_data_dict_loaded: ', det_data_dict_loaded)

    # print('before: gt_counter_per_class: ', gt_counter_per_class)

    for _class_name in gt_classes:
        if _class_name not in gt_counter_per_class.keys():
            gt_counter_per_class[_class_name] = 0

    # print('after: gt_counter_per_class: ', gt_counter_per_class)

    total_gt = 0
    for _class_name in gt_counter_per_class.keys():
        total_gt += gt_counter_per_class[_class_name]

    gt_fraction_per_class = {}
    gt_fraction_per_class_list = []
    for _class_name in gt_counter_per_class.keys():
        try:
            _gt_fraction = float(gt_counter_per_class[_class_name]) / float(total_gt)
        except ZeroDivisionError:
            print('gt_counter_per_class: ', gt_counter_per_class)
            print('total_gt: ', total_gt)
            _gt_fraction = 0

        gt_fraction_per_class[_class_name] = _gt_fraction
        gt_fraction_per_class_list.append(_gt_fraction)

    # gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    print('gt_classes: ', gt_classes)
    print('n_classes: ', n_classes)
    print('gt_counter_per_class: ', gt_counter_per_class)

    # print(gt_classes)
    # print(gt_counter_per_class)
    # if not gt_counter_per_class_loaded:
    #     print('Saving GT count data to {}'.format(gt_counter_per_class_path))
    #     with open(gt_counter_per_class_path, 'wb') as f:
    #         pickle.dump(gt_counter_per_class, f, pickle.HIGHEST_PROTOCOL)

    # if gt_data_dict_path and not gt_data_dict_loaded:
    #     print('Saving GT data to {}'.format(gt_data_dict_path))
    #     with open(gt_data_dict_path, 'wb') as f:
    #         pickle.dump(gt_data_dict, f, pickle.HIGHEST_PROTOCOL)

    # if not df_dets_loaded:
    #     with open(df_dets_path, 'wb') as f:
    #         pickle.dump(df_dets, f, pickle.HIGHEST_PROTOCOL)

    # """
    #  Check format of the flag --set-class-iou (if used)
    #   e.g. check if class exists
    # """
    # if specific_iou_flagged:
    #     n_args = len(set_class_iou)
    #     error_msg = \
    #         '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    #     if n_args % 2 != 0:
    #         raise SystemError('Error, missing arguments. Flag usage:' + error_msg)
    #     # [class_1] [IoU_1] [class_2] [IoU_2]
    #     # specific_iou_classes = ['class_1', 'class_2']
    #     specific_iou_classes = set_class_iou[::2]  # even
    #     # iou_list = ['IoU_1', 'IoU_2']
    #     iou_list = set_class_iou[1::2]  # odd
    #     if len(specific_iou_classes) != len(iou_list):
    #         raise SystemError('Error, missing arguments. Flag usage:' + error_msg)
    #     for tmp_class in specific_iou_classes:
    #         if tmp_class not in gt_classes:
    #             raise SystemError('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
    #     for num in iou_list:
    #         if not utils.is_float_between_0_and_1(num):
    #             raise SystemError('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)
    log_dir = 'pprint_log'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # with open(os.path.join(log_dir, 'gt_data_dict.txt'), 'w') as logFile:
    #     pprint(gt_data_dict, logFile)
    #
    # with open(os.path.join(log_dir, 'raw_det_data_dict.txt'), 'w') as logFile:
    #     pprint(raw_det_data_dict, logFile)

    print('Discarding detections with score < {}'.format(score_thresh))

    for gt_class_idx, gt_class in enumerate(gt_classes):

        # if det_data_dict_loaded:
        #     break

        class_det_data_dict = {}
        print('Processing class {:d}: {:s}'.format(gt_class_idx, gt_class))

        det_start_t = time.time()

        for seq_idx in range(n_seq):
            sys.stdout.write('\rPost processing sequence {:d}/{:d} '.format(
                seq_idx + 1, n_seq))
            sys.stdout.flush()

            seq_name = seq_name_list[seq_idx]
            seq_path = seq_paths[seq_idx]

            seq_gt = gt_data_dict[seq_path]
            seq_det = raw_det_data_dict[seq_path]

            det_exists = {}
            bounding_boxes = []
            for _data in seq_det:
                det_class = _data['class']
                file_path = _data['file_path']
                confidence = _data['confidence']
                bbox = _data['bbox']

                if file_path not in det_exists:
                    det_exists[file_path] = 0

                if bbox is None or det_class != gt_class or confidence < score_thresh:
                    continue

                bounding_boxes.append(
                    {"confidence": confidence, "file_id": file_path, "bbox": bbox})
                det_exists[file_path] = 1

            for file_path in det_exists:
                if det_exists[file_path]:
                    continue

                try:
                    gt_data = seq_gt[file_path]
                except KeyError as e:
                    if not os.path.isdir('pprint_log'):
                        os.makedirs('pprint_log')
                    print('seq_path: {}'.format(seq_path))
                    print('file_path: {}'.format(file_path))

                    with open('map_pprint_log.txt', 'w') as _fid:
                        pprint(('seq_det', seq_det), _fid)
                        pprint(('seq_gt', seq_gt), _fid)

                    raise KeyError(e)

                for obj in gt_data:
                    if obj['class_name'] == gt_class:
                        bounding_boxes.append({'confidence': None, 'file_id': file_path, 'bbox': None})
                        break

            class_det_data_dict[seq_path] = bounding_boxes

        det_end_t = time.time()
        det_data_dict[gt_class] = class_det_data_dict
        sys.stdout.write('\n')
        sys.stdout.flush()

        print('Time taken: {} sec'.format(det_end_t - det_start_t))

    # if det_data_dict_path and not det_data_dict_loaded:
    #     print('Saving detection data to {}'.format(det_data_dict_path))
    #     with open(det_data_dict_path, 'wb') as f:
    #         pickle.dump(det_data_dict, f, pickle.HIGHEST_PROTOCOL)

    if not vis_out_fname:
        seq_name = seq_name_list[0]
        vis_out_fname = os.path.join(save_dir, '{}.{}'.format(seq_name, vid_ext))
    _save_dir = os.path.dirname(vis_out_fname)
    if _save_dir and not os.path.isdir(_save_dir):
        os.makedirs(_save_dir)

    """
     Calculate the AP for each class
    """
    weighted_mAP = sum_AP = 0.0
    weighted_prec = weighted_rec = weighted_rec_prec = wm_score = 0.0
    sum_prec = sum_rec = sum_rec_prec = sum_score = 0.0

    # ap_dictionary = {}
    print('Calculating the AP for each class...')

    # colors (OpenCV works with BGR)
    white = (255, 255, 255)
    blue = (255, 0, 0)
    light_blue = (255, 200, 100)
    green = (0, 255, 0)
    light_red = (30, 30, 255)
    magenta = (255, 0, 255)
    # 1st line
    margin = 10
    # Add bottom border to image
    bottom_border = 60
    BLACK = [0, 0, 0]

    video_out = None
    win_name = "s: next sequence c: next class q/escape: quit"

    if save_animation:
        video_h = save_h
        video_w = save_w
        if show_text:
            video_h += bottom_border
        video_out = cv2.VideoWriter(vis_out_fname, fourcc, fps, (video_w, video_h))
        if not video_out:
            raise SystemError('Visualizations video file: {} could not be opened'.format(vis_out_fname))
        print('Saving visualizations to {}\n'.format(vis_out_fname))

    count_true_positives = {}
    tp_sum_overall = 0
    fp_sum_overall = 0
    fn_sum_overall = 0
    gt_overall = 0

    if write_summary:
        if not out_fname:
            if not os.path.isdir('results'):
                os.makedirs('results')
            out_fname = os.path.join('results', '{:s}.txt'.format(
                os.path.splitext(os.path.basename(vis_out_fname))[0]))
        print('Writing result summary to {}'.format(out_fname))

    out_template = os.path.join(os.path.dirname(out_fname), os.path.splitext(os.path.basename(out_fname))[0]).replace(
        os.sep, '_')
    out_text = out_template
    text = 'class\tAP(%)\tPrecision(%)\tRecall(%)\tR=P(%)\tScore(%)\tTP\tFN\tFP\tGT'
    out_text += '\n' + text + '\n'

    text_table = PrettyTable(text.split('\t'))

    return_eval_dict = 0
    if eval_result_dict is None:
        return_eval_dict = 1
        eval_result_dict = {}

    # print(text)
    # if write_summary:
    #     out_file.write(text + '\n')

    if cv2.__version__.startswith('2'):
        font_line_type = cv2.CV_AA
    else:
        font_line_type = cv2.LINE_AA

    rec_thresh_all = np.zeros((n_score_thresholds, n_classes))
    prec_thresh_all = np.zeros((n_score_thresholds, n_classes))

    cmb_summary_data = {}

    class_rec_prec = np.zeros((n_score_thresholds, n_classes * 2), dtype=np.float32)
    class_rec_prec_header = ['score_thresh', ]

    class_rec_prec[:, 0] = score_thresholds

    out_text_class = ''

    for gt_class_idx, gt_class in enumerate(gt_classes):

        print('Processing class {:d}/{:d}: {:s}'.format(gt_class_idx + 1, n_classes, gt_class))

        count_true_positives[gt_class] = 0
        end_class = 0
        """
         Load predictions of that class
        """
        fp_class = []
        tp_class = []
        conf_class = []
        tp_sum = 0
        fp_sum = 0
        fn_sum = 0
        n_used_gt = 0
        n_total_gt = 0

        all_considered_gt = []
        # absolute_all_gt = []
        duplicate_gt = []
        det_file_ids = []

        vis_frames = {}

        for seq_idx in range(n_seq):
            seq_path = seq_paths[seq_idx]
            seq_name = seq_name_list[seq_idx]

            seq_gt_data_dict = gt_data_dict[seq_path]

            seq_det_data = det_data_dict[gt_class][seq_path]

            seq_det_data.sort(key=lambda x: x['file_id'])
            # seq_det_data.sort(key=lambda x: x['confidence'], reverse=True)

            """
             Assign predictions to ground truth objects
            """
            missing_detections = []
            n_detections = len(seq_det_data)
            tp = [0] * n_detections
            fp = [0] * n_detections
            fn = [0] * n_detections
            conf = [0] * n_detections
            det_idx = 0
            img = None
            while True:
                if enable_animation and img is not None and \
                        (status == "MATCH!" or not show_only_tp):
                    # _h, _w = img.shape[:2]
                    # print('_h: ', _h)
                    # print('_w: ', _w)
                    if save_animation:
                        video_out.write(img)

                    if show_animation:
                        # vis_frames[file_id] = img

                        cv2.imshow(win_name, img)
                        k = cv2.waitKey(1 - _pause)
                        if k == ord('q') or k == 27:
                            if save_animation:
                                video_out.release()
                            cv2.destroyAllWindows()
                            sys.exit(0)
                        elif k == ord('c'):
                            end_class = 1
                            break
                        elif k == ord('s'):
                            break
                        elif k == 32:
                            _pause = 1 - _pause
                    else:
                        sys.stdout.write('\rDone detection {:d}/{:d} for seq {:d}/{:d}: {:s}, class {:s}'.format(
                            det_idx + 1, n_detections, seq_idx + 1, n_seq, seq_name, gt_class))
                        sys.stdout.flush()

                if det_idx >= n_detections:
                    break

                frame_det_data = seq_det_data[det_idx]

                file_id = frame_det_data["file_id"]
                conf[det_idx] = frame_det_data["confidence"]

                det_file_ids.append(file_id)

                if enable_animation:
                    ground_truth_img = file_id
                    img_full_path = os.path.join(seq_path, ground_truth_img)
                    img = cv2.imread(img_full_path)
                    if img is None:
                        raise IOError('Image {} could not be read'.format(img_full_path))

                # gt_file = os.path.join(pkl_files_path, seq_name, '{}_ground_truth.json'.format(file_id))
                # frame_gt_data = json.load(open(gt_file))
                # if gt_file not in loaded_json:
                #     loaded_json.add(gt_file)
                #     for obj in frame_gt_data:
                #         obj["used"] = False

                if enable_animation and missing_detections:
                    # print('\nfile_id: ', file_id)
                    # print('missing_detections:\n ', missing_detections)
                    for det in missing_detections:
                        bb_det = [int(x) for x in det['bbox']]
                        cv2.rectangle(img, (bb_det[0], bb_det[1]), (bb_det[2], bb_det[3]), magenta, 2)

                    img = resizeAR(img, save_w, save_h)
                    img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                    height, _ = img.shape[:2]
                    v_pos = int(height - margin - (bottom_border / 2))
                    text = "{}: {} ".format(seq_name, ground_truth_img)
                    img, line_width = utils.draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    text = "Class [" + str(gt_class_idx + 1) + "/" + str(n_classes) + "]: " + gt_class + " "
                    img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue,
                                                               line_width)

                    color = light_red
                    status = "MISSING DETECTION"

                    text = "Result: {}".format(status)
                    img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), color,
                                                               line_width)

                    if show_stats:
                        v_pos += int(bottom_border / 2)
                        try:
                            _recall = float(tp_sum) / float(tp_sum + fn_sum) * 100.0
                        except ZeroDivisionError:
                            _recall = 0
                        try:
                            _prec = float(tp_sum) / float(tp_sum + fp_sum) * 100.0
                        except ZeroDivisionError:
                            _prec = 0
                        text = 'tp: {:d} fn: {:d} fp: {:d} recall: {:5.2f}% prec: {:5.2f}%'.format(
                            tp_sum, fn_sum, fp_sum, _recall, _prec)
                        img, line_width = utils.draw_text_in_image(img, text, (margin, v_pos), white, line_width)
                    missing_detections = []
                    det_idx += 1
                    continue

                try:
                    frame_gt_data = seq_gt_data_dict[file_id]
                except KeyError as e:
                    print('seq_name: {}'.format(seq_name))
                    raise KeyError(e)

                # _duplicate_gt = [obj for obj in frame_gt_data if
                #                  obj['class_name'] == gt_class and obj in absolute_all_gt]
                # duplicate_gt += _duplicate_gt
                #
                # _absolute_all_gt = [obj for obj in frame_gt_data if obj['class_name'] == gt_class]
                # absolute_all_gt += _absolute_all_gt

                if frame_det_data["bbox"] is None:
                    fn[det_idx] = 1
                    missing_detections += [obj for obj in frame_gt_data if obj['class_name'] == gt_class]
                    fn_sum += len(missing_detections)

                    used_gt = [obj for obj in frame_gt_data if obj["class_name"] == gt_class and obj['used']]
                    n_used_gt += len(used_gt)

                    all_gt = [obj for obj in frame_gt_data if obj['class_name'] == gt_class]
                    n_total_gt += len(all_gt)

                    all_considered_gt += all_gt

                    # print('None det {} :: {} :: n_total_gt: {} n_used_gt: {} tp_sum: {} fn_sum: {}'.format(
                    #     seq_name, file_id, n_total_gt, n_used_gt, tp_sum, fn_sum))

                    if n_total_gt != n_used_gt + fn_sum:
                        print('missing_detections:\n{}'.format(pformat(missing_detections)))
                        print('all_gt:\n{}'.format(pformat(all_gt)))

                        raise SystemError('{} : {} :: Mismatch between n_total_gt: {} and n_used_gt+fn_sum: {}'.format(
                            gt_class, file_id, n_total_gt, n_used_gt + fn_sum
                        ))

                    if n_used_gt != tp_sum:
                        raise SystemError('{} : {} :: Mismatch between n_used_gt: {} and tp_sum: {}'.format(
                            gt_class, file_id, n_used_gt, tp_sum
                        ))

                    if enable_animation:
                        if missing_detections:
                            img = None
                    else:
                        missing_detections = []
                        det_idx += 1
                    continue

                ovmax = -1
                gt_match = -1
                # load frame_det_data bounding-box
                bb_det = frame_det_data["bbox"]
                # bb_det = [float(x) for x in frame_det_data["bbox"].split()]
                for obj in frame_gt_data:
                    # look for a gt_class match
                    if obj["class_name"] != gt_class:
                        continue
                    bb_gt = obj["bbox"]
                    # bb_gt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb_det[0], bb_gt[0]), max(bb_det[1], bb_gt[1]), min(bb_det[2], bb_gt[2]),
                          min(bb_det[3], bb_gt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1

                    if iw <= 0 or ih <= 0:
                        continue
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb_det[2] - bb_det[0] + 1) * (bb_det[3] - bb_det[1] + 1) + (bb_gt[2] - bb_gt[0]
                                                                                      + 1) * (
                                 bb_gt[3] - bb_gt[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = obj

                # assign frame_det_data as true positive or false positive
                if enable_animation:
                    status = "NO MATCH FOUND!"  # status is only used in the animation
                # set minimum overlap
                min_overlap = iou_thresh
                # if specific_iou_flagged:
                #     if gt_class in specific_iou_classes:
                #         index = specific_iou_classes.index(gt_class)
                #         min_overlap = float(iou_list[index])

                if ovmax >= min_overlap:
                    if not gt_match['used']:
                        # true positive
                        tp[det_idx] = 1
                        tp_sum += 1
                        gt_match['used'] = True
                        count_true_positives[gt_class] += 1

                        # update the ".json" file
                        # with open(gt_file, 'w') as f:
                        #     f.write(json.dumps(frame_gt_data))

                        if enable_animation:
                            status = "MATCH!"
                    else:
                        # false positive (multiple detection)
                        fp[det_idx] = 1
                        fp_sum += 1
                        if enable_animation:
                            status = "REPEATED MATCH!"
                else:
                    # false positive
                    fp[det_idx] = 1
                    fp_sum += 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

                """
                 Draw image to show animation
                """
                if enable_animation:
                    if status == "MATCH!":
                        box_color = green
                    else:
                        box_color = light_red
                    cv2.rectangle(img, (int(bb_det[0]), int(bb_det[1])), (int(bb_det[2]), int(bb_det[3])), box_color, 2)

                    if show_gt and ovmax > 0:  # if there is intersections between the bounding-boxes
                        bb_gt = gt_match["bbox"]
                        # bb_gt = [float(x) for x in gt_match["bbox"].split()]
                        cv2.rectangle(img, (int(bb_gt[0]), int(bb_gt[1])), (int(bb_gt[2]), int(bb_gt[3])), light_blue,
                                      2)

                    img, resize_factor, start_row, start_col = resizeAR(img, save_w, save_h, return_factors=True)

                    if not show_text:
                        _xmin = (bb_det[0] + start_col) * resize_factor
                        _xmax = (bb_det[2] + start_col) * resize_factor
                        _ymin = (bb_det[1] + start_row) * resize_factor
                        _ymax = (bb_det[3] + start_row) * resize_factor

                        _bb = [_xmin, _ymin, _xmax, _ymax]
                        if _bb[1] > 10:
                            y_loc = int(_bb[1] - 5)
                        else:
                            y_loc = int(_bb[3] + 5)
                        box_label = '{}: {:.2f}%'.format(gt_class, float(frame_det_data["confidence"]) * 100)
                        cv2.putText(img, box_label, (int(_bb[0] - 1), y_loc),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, font_line_type)
                    else:
                        img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                        height, _ = img.shape[:2]
                        v_pos = int(height - margin - (bottom_border / 2))
                        text = "{}: {} ".format(seq_name, ground_truth_img)
                        img, line_width = utils.draw_text_in_image(img, text, (margin, v_pos), white, 0)
                        text = "Class [" + str(gt_class_idx + 1) + "/" + str(n_classes) + "]: " + gt_class + " "
                        img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue,
                                                                   line_width)
                        color = light_red
                        if status == "MATCH!":
                            color = green
                        text = "Result: " + status + " "
                        img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), color,
                                                                   line_width)

                        v_pos += int(bottom_border / 2)
                        # rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                        # text = "Prediction #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                        #     float(frame_det_data["confidence"]) * 100)

                        text = ''
                        if show_stats:
                            try:
                                _recall = float(tp_sum) / float(tp_sum + fn_sum) * 100.0
                            except ZeroDivisionError:
                                _recall = 0
                            try:
                                _prec = float(tp_sum) / float(tp_sum + fp_sum) * 100.0
                            except ZeroDivisionError:
                                _prec = 0

                            text += 'tp: {:d} fn: {:d} fp: {:d} recall: {:5.2f}% prec: {:5.2f}% '.format(
                                tp_sum, fn_sum, fp_sum, _recall, _prec)
                            # text += 'tp: {:d} fn: {:d} fp: {:d} '.format(tp_sum, fn_sum, fp_sum)
                        text += "confidence: {0:.2f}% ".format(float(frame_det_data["confidence"]) * 100)
                        img, line_width = utils.draw_text_in_image(img, text, (margin, v_pos), white, 0)

                        if ovmax != -1:
                            color = light_red
                            if status == "INSUFFICIENT OVERLAP":
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                            else:
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                                color = green
                            img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), color,
                                                                       line_width)

                # if det_idx == n_detections - 1:
                if det_idx == n_detections - 1 or seq_det_data[det_idx + 1]['file_id'] != file_id:
                    missing_detections += [obj for obj in frame_gt_data if
                                           obj["class_name"] == gt_class and not obj['used']]
                    n_missing_detections = len(missing_detections)

                    fn_sum += n_missing_detections

                    used_gt = [obj for obj in frame_gt_data if obj["class_name"] == gt_class and obj['used']]
                    n_used_gt += len(used_gt)

                    all_gt = [obj for obj in frame_gt_data if obj['class_name'] == gt_class]
                    n_total_gt += len(all_gt)

                    all_considered_gt += all_gt

                    # print('{} :: {} :: n_total_gt: {} n_used_gt: {} tp_sum: {} fn_sum: {}'.format(
                    #     seq_name, file_id, n_total_gt, n_used_gt, tp_sum, fn_sum))

                    if n_total_gt != n_used_gt + fn_sum:
                        print('missing_detections:\n{}'.format(pformat(missing_detections)))
                        print('all_gt:\n{}'.format(pformat(all_gt)))

                        raise SystemError('{} : {} :: Mismatch between n_total_gt: {} and n_used_gt+fn_sum: {}'.format(
                            gt_class, file_id, n_total_gt, n_used_gt + fn_sum
                        ))

                    if n_used_gt != tp_sum:
                        raise SystemError('{} : {} :: Mismatch between n_used_gt: {} and tp_sum: {}'.format(
                            gt_class, file_id, n_used_gt, tp_sum
                        ))

                    if enable_animation:
                        if missing_detections:
                            # img = None
                            continue
                    else:
                        missing_detections = []
                det_idx += 1

            # -------------------------------
            # completed processing all frames for one sequence
            # -------------------------------

            if enable_animation and not show_animation:
                sys.stdout.write('\n')
                sys.stdout.flush()

            if end_class:
                break

            fp_class += [x for i, x in enumerate(fp) if fn[i] == 0]
            tp_class += [x for i, x in enumerate(tp) if fn[i] == 0]
            conf_class += [x for i, x in enumerate(conf) if fn[i] == 0]

            # sys.stdout.write('\rDone sequence {:d}/{:d}: {:s}'.format(seq_idx + 1, n_seq, seq_path))
            # sys.stdout.flush()

        # print('fp_class: ', fp_class)
        # print('tp_class: ', tp_class)

        # -------------------------------
        # completed processing all sequences for one class
        # -------------------------------

        # print('Sorting by confidence...')
        sort_idx = np.argsort(conf_class)[::-1]
        fp_class = [fp_class[i] for i in sort_idx]
        tp_class = [tp_class[i] for i in sort_idx]
        conf_class = [conf_class[i] for i in sort_idx]

        if n_score_thresholds > 1:
            if n_threads == 1:
                print('Computing recall and precision '
                      'over {} thresholds without multi threading'.format(
                    n_score_thresholds))
                _start_t = time.time()
                _rec_prec_list = []
                for __thresh_idx in range(n_score_thresholds):
                    __temp = computeThreshRecPrec(
                        __thresh_idx,
                        score_thresholds=score_thresholds,
                        gt_counter_per_class=gt_counter_per_class,
                        conf_class=conf_class,
                        fp_class=fp_class,
                        tp_class=tp_class,
                        gt_class=gt_class, )
                    _rec_prec_list.append(__temp)
            else:
                if n_threads == 0:
                    n_threads = multiprocessing.cpu_count()
                print('Computing recall and precision '
                      'over {} thresholds using {} threads'.format(
                    n_score_thresholds, n_threads))
                _start_t = time.time()
                with closing(multiprocessing.Pool(n_threads)) as pool:
                    _rec_prec_list = pool.map(functools.partial(
                        computeThreshRecPrec,
                        score_thresholds=score_thresholds,
                        gt_counter_per_class=gt_counter_per_class,
                        conf_class=conf_class,
                        fp_class=fp_class,
                        tp_class=tp_class,
                        gt_class=gt_class,
                    ), range(n_score_thresholds))
                rec_thresh_all[:, gt_class_idx] = [_rec_prec[0] for _rec_prec in _rec_prec_list]
                prec_thresh_all[:, gt_class_idx] = [_rec_prec[1] for _rec_prec in _rec_prec_list]
                del _rec_prec_list
            _end_t = time.time()
            print('Time taken: {:.4f}'.format(_end_t - _start_t))
            # print()

        # compute precision/recall
        cumsum = 0
        for det_idx, val in enumerate(fp_class):
            # fp_class[det_idx] Seems to have the number of false positives Encountered if only the 1st det_idx +1
            # detections are considered
            fp_class[det_idx] += cumsum
            cumsum += val
        cumsum = 0
        for det_idx, val in enumerate(tp_class):
            tp_class[det_idx] += cumsum
            cumsum += val
        # print('tp: ', tp)

        # print('fp_class: ', fp_class)
        # print('tp_class: ', tp_class)

        rec = tp_class[:]
        for det_idx, val in enumerate(tp_class):
            if tp_class[det_idx] > 0:
                rec[det_idx] = float(tp_class[det_idx]) / gt_counter_per_class[gt_class]
        # print(rec)
        prec = tp_class[:]
        for det_idx, val in enumerate(tp_class):
            try:
                prec[det_idx] = float(tp_class[det_idx]) / (fp_class[det_idx] + tp_class[det_idx])
            except ZeroDivisionError:
                prec[det_idx] = 0

        # print(prec)

        ap, mrec, mprec = utils.voc_ap(rec, prec)

        if draw_plot:
            fig1 = plt.figure(figsize=(18, 9), dpi=80)
            plt.subplot(1, 2, 1)
            plt.plot(rec, prec, 'b-.')
            plt.fill_between(mrec, 0, mprec, alpha=0.2, edgecolor='r')

            # set window title
            fig1.canvas.set_window_title('AP ' + gt_class)
            # set plot title
            plt.title('class: ' + text)
            plt.grid(1)
            # plt.suptitle('This is a somewhat long figure title', fontsize=16)
            # set axis titles
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # optional - set axes
            axes = plt.gca()  # gca - get current axes
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])  # .05 to give some extra space

            # fig2 = plt.figure()
            plt.subplot(1, 2, 2)
            plt.plot(conf_class, rec, 'r-')
            # plt.hold(1)
            plt.plot(conf_class, prec, 'g-')
            plt.title('Recall and Precision vs Confidence')
            # plt.hold(0)
            plt.grid(1)

            plt.legend(['Recall', 'Precision'])

            plt.xlabel('Confidence')
            plt.ylabel('Recall / Precision')

            axes = plt.gca()  # gca - get current axes
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.0])  # .05 to give some extra space

            # Alternative option -> wait for button to be pressed
            # while not plt.waitforbuttonpress():
            #     pass

            # Alternative option -> normal display
            # plt.show()

            # save the plot
            plot_out_fname = os.path.join(plots_out_dir, gt_class + ".png")
            print('Saving plot to: {}'.format(plot_out_fname))
            fig1.savefig(plot_out_fname)

            plt.close(fig1)

        rec = np.asarray(rec)
        prec = np.asarray(prec)
        conf_class = np.asarray(conf_class)

        _diff = rec - prec

        idx = np.argwhere(np.diff(np.sign(_diff))).flatten()
        if not idx.size:
            # print('rec/prec: {}'.format(pformat(np.vstack((conf_class, rec, prec)).T)))

            idx = np.argmin(np.abs(_diff))
            if idx.size > 1:
                idx = idx[0]
            _txt += 'No intersection between recall and precision found; ' \
                    'min_difference: {} at {} for confidence: {}'.format(
                _diff[idx], (rec[idx], prec[idx]), conf_class[idx]
            )
            print(_txt)
            _rec_prec = (rec[idx] + prec[idx]) / 2.0
            _score = conf_class[idx]
        else:
            _txt = 'Intersection at {} for confidence: {} with idx: {}'.format(rec[idx], conf_class[idx], idx)
            print(_txt)
            _rec_prec = rec[idx[0]]
            _score = conf_class[idx][0]

        if _score < score_thresh:
            print('conf_class: {}'.format(pformat(list(conf_class))))
            raise SystemError('_score: {} < score_thresh: {}'.format(_score, score_thresh))

        if draw_plot:
            out_text_class = _txt + '\n'

            out_text_class += '{}_rec_prec\n{}\n'.format(gt_class,
                pd.DataFrame(data=np.vstack((conf_class, rec * 100, prec * 100)).T,
                             columns=['score_thresh', '{}_recall'.format(gt_class),
                                      '{}_precision'.format(gt_class)]).to_csv(
                    sep='\t', index=False),
            )
            out_text_class += '\n'

            # out_fname_class = out_fname.replace('.txt', '_class.md')
            out_fname_class = out_fname + '.{}'.format(gt_class)
            with open(out_fname_class, 'w') as out_file:
                out_file.write(out_text_class)
            print('Saved {} result summary to {}'.format(gt_class, out_fname_class))

        if tp_sum > 0:
            _rec = float(tp_sum) / gt_counter_per_class[gt_class]
        else:
            _rec = 0
        try:
            _prec = float(tp_sum) / (fp_sum + tp_sum)
        except ZeroDivisionError:
            _prec = 0

        weighted_mAP += ap * gt_fraction_per_class[gt_class]
        weighted_prec += _prec * gt_fraction_per_class[gt_class]
        weighted_rec += _rec * gt_fraction_per_class[gt_class]
        weighted_rec_prec += _rec_prec * gt_fraction_per_class[gt_class]
        wm_score += _score * gt_fraction_per_class[gt_class]

        sum_AP += ap
        sum_prec += _prec
        sum_rec += _rec
        sum_rec_prec += _rec_prec
        sum_score += _score

        no_det_file_ids = [k for k in seq_gt_data_dict if k not in det_file_ids and k != 'csv']
        # if no_det_file_ids:
        #     raise SystemError('{} files have no detections: {}'.format(len(no_det_file_ids),
        #                                                                no_det_file_ids))
        all_class_gt = []
        for k in gt_data_dict:
            if k == 'counter_per_class':
                continue
            for m in gt_data_dict[k]:
                if m == 'csv':
                    continue
                all_class_gt += [obj for obj in gt_data_dict[k][m] if obj['class_name'] == gt_class]

        skipped_gt = [obj for obj in all_class_gt if obj not in all_considered_gt]

        n_all_considered_gt = len(all_considered_gt)
        # n_absolute_all_gt = len(absolute_all_gt)
        # n_duplicate_gt = len(duplicate_gt)
        n_all_class_gt = len(all_class_gt)
        n_skipped_gt = len(skipped_gt)

        if n_all_considered_gt != n_all_class_gt:
            # annoying_gt = [k for k in absolute_all_gt if k not in all_considered_gt]
            # n_annoying_gt = len(annoying_gt)

            # print('annoying_gt:')
            # pprint(annoying_gt)

            # print('duplicate_gt:')
            # pprint(duplicate_gt)

            # print('skipped_gt:\n{}'.format(pformat(skipped_gt)))
            print('gt_counter_per_class:\n{}'.format(pformat(gt_counter_per_class)))

            raise SystemError('{} :: Mismatch between '
                              'n_all_considered_gt: {} '
                              'and n_all_class_gt: {}, '
                              'n_skipped_gt: {} '.format(
                gt_class,
                n_all_considered_gt,
                n_all_class_gt,
                n_skipped_gt
            ))

        if n_total_gt != tp_sum + fn_sum:
            raise SystemError('{} :: Mismatch between n_total_gt: {} and tp_sum+fn_sum: {}, n_used_gt: {}'.format(
                gt_class, n_total_gt, tp_sum, fn_sum, n_used_gt
            ))

        if n_total_gt != gt_counter_per_class[gt_class]:
            raise SystemError('Mismatch between n_total_gt: {} and gt_counter_per_class[{}]: {}'.format(
                n_total_gt, gt_class, gt_counter_per_class[gt_class]
            ))

        # text = "{:s} AP: {:.2f}% prec: {:.2f}% rec: {:.2f}% tp: {:d} fn: {:d} fp: {:d} gt: {:d}\n".format(
        #     gt_class, ap * 100, _prec * 100, _rec * 100, tp_sum, fn_sum, fp_sum, gt_counter_per_class[gt_class])

        text = "{:s}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}\t{:d}\t{:d}\t{:d}".format(
            gt_class, ap * 100, _prec * 100, _rec * 100, _rec_prec * 100, _score * 100, tp_sum, fn_sum, fp_sum,
            gt_counter_per_class[gt_class])

        eval_result_dict[gt_class] = {
            'AP': ap * 100,
            'Precision': _prec * 100,
            'Recall': _rec * 100,
            'R=P': _rec_prec * 100,
            'Score': _score * 100,
            'TP': tp_sum,
            'FN': fn_sum,
            'FP': fp_sum,
            'GT': gt_counter_per_class[gt_class],
        }
        text_table.add_row(text.split('\t'))

        tp_sum_overall += tp_sum
        fp_sum_overall += fp_sum
        fn_sum_overall += fn_sum
        gt_overall += gt_counter_per_class[gt_class]

        cmb_summary_data[gt_class] = [ap * 100, _rec_prec * 100, _score * 100, gt_counter_per_class[gt_class]]

        """
         Write to results.txt
        """
        # rounded_prec = ['%.2f' % elem for elem in prec]
        # rounded_rec = ['%.2f' % elem for elem in rec]

        # results_file.write(
        #     text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

        # print(text)
        out_text += text + '\n'
        # if write_summary:
        #     out_file.write(text + '\n')

        # ap_dictionary[gt_class] = ap
        # sys.stdout.write('\n')
        # sys.stdout.flush()

    if enable_animation and show_animation:
        cv2.destroyAllWindows()

    mAP = sum_AP / n_classes
    m_prec = sum_prec / n_classes
    m_rec = sum_rec / n_classes
    m_rec_prec = sum_rec_prec / n_classes
    m_score = sum_score / n_classes

    if wt_avg:
        avg_txt = 'wt_avg'
        avg_wts = gt_fraction_per_class_list

    else:
        avg_txt = 'avg'
        avg_wts = None

    # text = 'Overall\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}\t{:d}\t{:d}\t{:d}'.format(
    #     mAP * 100, m_prec * 100, m_rec * 100, m_rec_prec * 100, m_score * 100,
    #     tp_sum_overall, fn_sum_overall, fp_sum_overall, gt_overall)
    # text_table.add_row(text.split('\t'))
    # out_text += text + '\n'

    # text = 'mean\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(weighted_mAP * 100, weighted_prec * 100,
    #                                                          weighted_rec * 100, weighted_rec_prec * 100)
    # text_table.add_row(text.split('\t') + [''] * 5)

    text = 'avg\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}\t{:d}\t{:d}\t{:d}'.format(
        mAP * 100, m_prec * 100, m_rec * 100, m_rec_prec * 100, m_score * 100,
        tp_sum_overall, fn_sum_overall, fp_sum_overall, gt_overall)
    text_table.add_row(text.split('\t'))
    out_text += text + '\n'

    text = 'wt_avg\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}\t{:d}\t{:d}\t{:d}'.format(
        weighted_mAP * 100, weighted_prec * 100, weighted_rec * 100, weighted_rec_prec * 100, wm_score * 100,
        tp_sum_overall, fn_sum_overall, fp_sum_overall, gt_overall)
    text_table.add_row(text.split('\t'))
    out_text += text + '\n'

    eval_result_dict['overall'] = {
        '_AP': mAP * 100,
        '_Precision': m_prec * 100,
        '_Recall': m_rec * 100,
        '_R=P': m_rec_prec * 100,
        'AP': weighted_mAP * 100,
        'Precision': weighted_prec * 100,
        'Recall': weighted_rec * 100,
        'R=P': weighted_rec_prec * 100,
        'TP': tp_sum_overall,
        'FN': fn_sum_overall,
        'FP': fp_sum_overall,
        'GT': gt_overall,
    }
    cmb_summary_data['avg'] = [mAP * 100, m_rec_prec * 100, m_score * 100, gt_overall]

    if n_score_thresholds > 1:
        print('Computing combined results over {} thresholds'.format(n_score_thresholds))
        # m_rec_thresh = [0] * n_score_thresholds
        # m_prec_thresh = [0] * n_score_thresholds

        wm_rec_thresh = np.zeros((n_score_thresholds,))
        wm_prec_thresh = np.zeros((n_score_thresholds,))

        # m_rec_thresh = np.zeros((n_score_thresholds,))
        # m_prec_thresh = np.zeros((n_score_thresholds,))

        gt_fraction_per_class_list = np.asarray(gt_fraction_per_class_list).squeeze()

        for thresh_idx, _thresh in enumerate(score_thresholds):
            _rec_thresh, _prec_thresh = rec_thresh_all[thresh_idx, :].squeeze(), \
                                        prec_thresh_all[thresh_idx, :].squeeze()
            # m_rec_thresh[thresh_idx] = np.mean(_rec_thresh)
            # m_prec_thresh[thresh_idx] = np.mean(_prec_thresh)

            # print('_rec_thresh.shape: {}'.format(_rec_thresh.shape))
            # print('_prec_thresh.shape: {}'.format(_prec_thresh.shape))
            # print('gt_fraction_per_class_list.shape: {}'.format(gt_fraction_per_class_list.shape))

            wm_rec_thresh[thresh_idx] = np.average(_rec_thresh, weights=avg_wts)
            wm_prec_thresh[thresh_idx] = np.average(_prec_thresh, weights=avg_wts)

            # m_rec_thresh[thresh_idx] = np.average(_rec_thresh)
            # m_prec_thresh[thresh_idx] = np.average(_prec_thresh)

        overall_ap_thresh, _, _ = utils.voc_ap(wm_rec_thresh[::-1], wm_prec_thresh[::-1])
        wm_diff_thresh = wm_rec_thresh - wm_prec_thresh

        itsc_idx = np.argwhere(np.diff(np.sign(wm_rec_thresh - wm_prec_thresh))).flatten()

        if not itsc_idx.size:
            # print('rec/prec: {}'.format(pformat(np.vstack((conf_class, rec, prec)).T)))
            itsc_idx = np.argmin(np.abs(wm_diff_thresh))
            if itsc_idx.size > 1:
                itsc_idx = itsc_idx[0]
                print('No intersection between recall and precision found; ' \
                        'min_difference: {} at {} for confidence: {}'.format(
                    wm_diff_thresh[itsc_idx], (wm_rec_thresh[itsc_idx], wm_prec_thresh[itsc_idx]),
                    score_thresholds[itsc_idx])
                )
        else:
            print('intersection at {} for confidence: {} with idx: {}'.format(
                wm_rec_thresh[itsc_idx], score_thresholds[itsc_idx], itsc_idx))

        print('overall_ap: {}'.format(overall_ap_thresh))

        if draw_plot:
            fig1 = plt.figure(figsize=(18, 9), dpi=80)
            plt.subplot(1, 2, 1)
            plt.plot(wm_rec_thresh, wm_prec_thresh, 'b-.')

            # set window title
            fig1.canvas.set_window_title('AP ' + gt_class)
            # set plot title
            plt.title('class: ' + text)
            plt.grid(1)
            # plt.suptitle('This is a somewhat long figure title', fontsize=16)
            # set axis titles
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # optional - set axes
            axes = plt.gca()  # gca - get current axes
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])  # .05 to give some extra space

            # fig2 = plt.figure()
            plt.subplot(1, 2, 2)
            plt.plot(score_thresholds, wm_rec_thresh, 'r-')
            # plt.hold(1)
            plt.plot(score_thresholds, wm_prec_thresh, 'g-')
            plt.title('Recall and Precision vs Confidence')
            # plt.hold(0)
            plt.grid(1)

            plt.legend(['Recall', 'Precision'])

            plt.xlabel('Confidence')
            plt.ylabel('Recall / Precision')

            axes = plt.gca()  # gca - get current axes
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.0])  # .05 to give some extra space

            # Alternative option -> wait for button to be pressed
            # while not plt.waitforbuttonpress():
            #     pass

            # Alternative option -> normal display
            # plt.show()

            # save the plot
            plot_out_fname = os.path.join(plots_out_dir, "overall.png")
            print('Saving plot to: {}'.format(plot_out_fname))
            fig1.savefig(plot_out_fname)

            plt.close(fig1)

        try:
            itsc_idx = itsc_idx[0]
        except IndexError:
            pass

        _idx_threshs = [itsc_idx, ]

        if compute_opt:
            diff_thresh = 0.02

            opt_idx = itsc_idx
            opt_data = []

            idx_threshs = range(itsc_idx + 1)[::-1]
            itsc_wm_rec, itsc_wm_prec = wm_rec_thresh[itsc_idx], wm_prec_thresh[itsc_idx]

            for curr_idx in idx_threshs:
                curr_wm_rec, curr_wm_prec = wm_rec_thresh[curr_idx], wm_prec_thresh[curr_idx]
                inc_rec, dec_prec = curr_wm_rec - itsc_wm_rec, itsc_wm_prec - curr_wm_prec

                diff_rec_prec = (curr_wm_rec - curr_wm_prec) / curr_wm_rec

                diff_dec_rec_prec = (inc_rec - dec_prec)

                opt_data.append([k * 100 for k in [score_thresholds[curr_idx], curr_wm_rec, curr_wm_prec,
                                                   inc_rec, dec_prec, diff_rec_prec]])

                if inc_rec < 0 or dec_prec < 0 or diff_rec_prec < 0:
                    # raise SystemError('Something weird going on: idx_threshs:\n{}\n'
                    #                   'curr_wm_rec: {} curr_wm_prec: {} inc_rec: {}, dec_prec: {} diff_rec_prec: {}'.format(
                    #     idx_threshs, curr_wm_rec, curr_wm_prec, inc_rec, dec_prec, diff_rec_prec))
                    break

                if inc_rec < dec_prec and diff_rec_prec > diff_thresh:
                    break

                opt_idx = curr_idx

            opt_score_thresh, opt_wm_rec, opt_wm_prec = score_thresholds[opt_idx], wm_rec_thresh[opt_idx], \
                                                        wm_prec_thresh[opt_idx]

            opt_data = np.asarray(opt_data)
            opt_headers = ['score_thresh', 'recall', 'precision', 'inc_rec', 'dec_prec', 'diff_rec_prec']
            print(tabulate(opt_data, opt_headers, tablefmt="fancy_grid"))

            if opt_idx != itsc_idx:
                _idx_threshs.append(opt_idx)

        # out_text += 'rec_ratio_data\n{}\n'.format(
        #     pd.DataFrame(data=opt_data, columns=opt_headers).to_csv(sep='\t', index=False))

        print('itsc_idx: {}'.format(itsc_idx))
        if isinstance(itsc_idx, list) and not itsc_idx:
            _score_threshold = 0
        else:
            _score_threshold = score_thresholds[itsc_idx]
            print('_score_threshold: {}'.format(_score_threshold))

        cmb_summary_text = '\tClass Specific\t\t\tmRP threshold {:.2f} %\t\t\n'.format(
            _score_threshold * 100)

        cmb_summary_text += 'class\tAP(%)\tRP(%)\tScore(%)\tRecall(%)\tPrecision(%)\tAverage(%)\tGT\n'

        for __i, _idx in enumerate(_idx_threshs):
            _score_threshold = score_thresholds[_idx] * 100

            for _class_id, _class_name in enumerate(gt_classes):
                _header = '{:s} {:.2f}'.format(_class_name, _score_threshold)

                class_rec = rec_thresh_all[:, _class_id].squeeze()
                class_prec = prec_thresh_all[:, _class_id].squeeze()

                class_ap, _, _ = utils.voc_ap(class_rec[_idx:][::-1], class_prec[_idx:][::-1])
                class_ap *= 100
                # print('score_threshold {} :: {} ap: {}'.format(_score_threshold, _class_name, class_ap))

                _curr_rec = class_rec[_idx] * 100
                _curr_prec = class_prec[_idx] * 100

                _curr_rec_prec = (_curr_prec + _curr_rec) / 2.0

                eval_result_dict[_header] = {
                    'AP': class_ap,
                    'Precision': _curr_prec,
                    'Recall': _curr_rec,
                    'Score': _score_threshold,
                }

                text = '{:s}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(_header, class_ap, _curr_prec, _curr_rec,
                                                                             _curr_rec_prec, _score_threshold)
                out_text += text + '\n'

                text_table.add_row([
                    '{:s}'.format(_header),
                    '{:.2f}'.format(class_ap),
                    '{:.2f}'.format(_curr_prec),
                    '{:.2f}'.format(_curr_rec),
                    '{:.2f}'.format(_curr_rec_prec),
                    '{:.2f}'.format(_score_threshold),
                    '',
                    '',
                    '',
                    '',
                ])
                if __i == 0:
                    __ap, __rec_prec, __score, __gt = cmb_summary_data[_class_name]

                    cmb_summary_text += '{:s}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}\n'.format(
                        _class_name, __ap, __rec_prec, __score,
                        _curr_rec, _curr_prec, _curr_rec_prec, __gt
                    )

            overall_ap, _, _ = utils.voc_ap(wm_rec_thresh[_idx:][::-1], wm_prec_thresh[_idx:][::-1])
            overall_ap *= 100
            # print('score_threshold {} :: overall ap: {}'.format(_score_threshold, overall_ap))

            _wm_rec, _wm_prec = wm_rec_thresh[_idx] * 100, wm_prec_thresh[_idx] * 100
            _wm_rec_prec = (_wm_prec + _wm_rec) / 2.0

            _header = '{} {:.2f}'.format(avg_txt, _score_threshold)
            eval_result_dict[_header] = {
                'AP': overall_ap,
                'Precision': _wm_prec,
                'Recall': _wm_rec,
                'Score': _score_threshold,
            }

            text = '{:s}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(_header, overall_ap, _wm_prec, _wm_rec,
                                                                         _wm_rec_prec, _score_threshold)
            out_text += text + '\n'
            text_table.add_row([
                '{:s}'.format(_header),
                '{:.2f}'.format(overall_ap),
                '{:.2f}'.format(_wm_prec),
                '{:.2f}'.format(_wm_rec),
                '{:.2f}'.format(_wm_rec_prec),
                '{:.2f}'.format(_score_threshold),
                '',
                '',
                '',
                '',
            ])
            if __i == 0:
                __ap, __rec_prec, __score, __gt = cmb_summary_data['avg']

                cmb_summary_text += '{:s}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}\n'.format(
                    'average', __ap, __rec_prec, __score,
                    _wm_rec, _wm_prec, _wm_rec_prec, __gt
                )

        if rec_ratios:
            rec_ratio_data = np.zeros((len(rec_ratios), 5))
            for _id, rec_ratio in enumerate(rec_ratios):
                avg_rec_prec = (wm_rec_thresh * rec_ratio + wm_prec_thresh) / (1 + rec_ratio)
                max_id = np.argmax(avg_rec_prec)
                rec_ratio_data[_id, :] = (rec_ratio, score_thresholds[max_id] * 100, wm_rec_thresh[max_id] * 100,
                                          wm_prec_thresh[max_id] * 100, avg_rec_prec[max_id] * 100)
            rec_ratio_headers = ['rec_ratio', 'score_thresh', 'recall', 'precision', 'average']
            print(tabulate(rec_ratio_data, rec_ratio_headers, tablefmt="fancy_grid"))
            out_text += 'rec_ratio_data\n{}\n'.format(
                pd.DataFrame(data=rec_ratio_data, columns=rec_ratio_headers).to_csv(sep='\t', index=False))

        out_text += 'rec_prec\n{}\n'.format(
            pd.DataFrame(data=np.vstack((score_thresholds, wm_rec_thresh * 100, wm_prec_thresh * 100)).T,
                         columns=['score_thresh', 'recall', 'precision']).to_csv(
                sep='\t', index=False),
        )
        out_text += '\n'

    if write_summary:
        cmb_summary_text = '{}\n{}'.format(out_template, cmb_summary_text)
        print(cmb_summary_text)

        print(text_table)
        # out_file.write(text_table.get_string() + '\n')

        with open(out_fname, 'w') as out_file:
            out_file.write(cmb_summary_text)
            out_file.write(out_text)
        print('Saved result summary to {}'.format(out_fname))

    if save_animation:
        video_out.release()
        print('Saved visualizations to {}\n'.format(vis_out_fname))

    # remove the tmp_files directory
    if delete_tmp_files:
        shutil.rmtree(pkl_files_path)

    if return_eval_dict:
        return eval_result_dict, out_text
    else:
        return text_table


def str_to_list(_str, _type=str, _sep=','):
    if _sep not in _str:
        _str += _sep
    return list(map(_type, _str.split(_sep)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
    parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
    parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    # parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
    parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
    parser.add_argument('--labels_path', type=str, help="text file containing class labels")
    parser.add_argument('--gt_path_list_file', type=str, help="file containing list of GT folders", default='')
    parser.add_argument('--det_path_list_file', type=str, help="file containing list of detection folders")
    parser.add_argument('--img_path_list_file', type=str, help="file containing list of image folders")
    parser.add_argument('--img_ext', type=str, help="image extension", default='jpg')
    parser.add_argument('--img_root_dir', type=str, help="folder to save the animation result in", default='')
    parser.add_argument('--gt_root_dir', type=str, help="folder to save the animation result in", default='')
    parser.add_argument('--save_dir', type=str, help="folder to save the animation result in", default='')
    parser.add_argument('--vis_out_fname', type=str, help="file to save the animation result in", default='mAP.mkv')
    parser.add_argument('--save_file_res', type=str, help="resolution of the saved video", default='1280x720')
    parser.add_argument('--pkl_files_path', type=str, help="location of temporary json files", default='tmp_files')
    parser.add_argument('--out_fname', type=str, help="out_fname", default='')
    parser.add_argument('--vid_fmt', type=str, help="comma separated triple to specify the output video format:"
                                                    " codec, FPS, extension", default='H264,30,mkv')
    parser.add_argument('--delete_tmp_files', type=int, default=0)
    parser.add_argument('--save_animation', type=int, default=1)
    parser.add_argument('--show_animation', type=int, default=1)
    parser.add_argument('--show_only_tp', type=int, default=0)
    parser.add_argument('--show_text', type=int, default=1)
    parser.add_argument('--show_stats', type=int, default=1)
    parser.add_argument('--show_gt', type=int, default=1)
    parser.add_argument('--draw_plot', type=int, default=0)
    parser.add_argument('--iou_thresh', type=float, help="iou_thresh", default=0.5)
    parser.add_argument('--score_thresholds', type=str_to_list, help="iou_thresh", default='0.5')

    args = parser.parse_args()

    print('args.gt_path_list_file', args.gt_path_list_file)
    print('args.det_path_list_file', args.det_path_list_file)
    print('args.img_path_list_file', args.img_path_list_file)
    print('args.labels_path', args.labels_path)

    save_dir = args.save_dir
    vis_out_fname = args.vis_out_fname
    vid_fmt = args.vid_fmt
    labels_path = args.labels_path
    iou_thresh = args.iou_thresh
    save_file_res = args.save_file_res
    pkl_files_path = args.pkl_files_path
    delete_tmp_files = args.delete_tmp_files
    save_animation = args.save_animation
    show_animation = args.show_animation
    show_only_tp = args.show_only_tp
    show_text = args.show_text
    show_stats = args.show_stats
    show_gt = args.show_gt
    draw_plot = args.draw_plot
    out_fname = args.out_fname

    gt_path_list_file = args.gt_path_list_file
    img_path_list_file = args.img_path_list_file
    det_path_list_file = args.det_path_list_file

    img_root_dir = args.img_root_dir
    gt_root_dir = args.gt_root_dir

    set_class_iou = args.set_class_iou
    score_thresholds = args.score_thresholds

    if not gt_root_dir:
        gt_root_dir = img_root_dir

    if not save_dir:
        save_dir = 'results'

    # if there are no classes to ignore then replace None by empty list
    # if args.ignore is None:
    #     args.ignore = []

    # specific_iou_flagged = False
    # if args.set_class_iou is not None:
    #     specific_iou_flagged = True

    img_ext = args.img_ext

    if os.path.isdir(img_path_list_file):
        img_path_list = [os.path.join(img_path_list_file, name) for name in os.listdir(img_path_list_file) if
                         os.path.isdir(os.path.join(img_path_list_file, name))]
    else:
        img_path_list = utils.file_lines_to_list(img_path_list_file)
        if not gt_path_list_file:
            gt_path_list_file = img_path_list_file
        if img_root_dir:
            img_path_list = [os.path.join(img_root_dir, name) for name in img_path_list]

    if os.path.isdir(gt_path_list_file):
        gt_path_list = [os.path.join(gt_path_list_file, name) for name in os.listdir(gt_path_list_file)
                        if os.path.isdir(os.path.join(gt_path_list_file, name))]

    else:
        gt_path_list = utils.file_lines_to_list(gt_path_list_file)
        # gt_path_list = [os.path.join(name, 'annotations.csv') for name in gt_path_list]
        if gt_root_dir:
            gt_path_list = [os.path.join(gt_root_dir, name) for name in gt_path_list]

    if os.path.isdir(det_path_list_file):
        det_path_list = [os.path.join(det_path_list_file, name) for name in os.listdir(det_path_list_file) if
                         os.path.isfile(os.path.join(det_path_list_file, name)) and name.endswith('.csv')]
    else:
        det_path_list = utils.file_lines_to_list(det_path_list_file)
        det_path_list = [name + '.csv' for name in det_path_list]

    gt_path_list.sort(key=sortKey)
    img_path_list.sort(key=sortKey)
    det_path_list.sort(key=sortKey)

    gt_classes = open(labels_path, 'r').readlines()
    gt_classes = [x.strip() for x in gt_classes]

    eval_result_dict, _ = evaluate(img_path_list, gt_path_list, det_path_list, gt_classes, pkl_files_path, save_dir,
                                   vis_out_fname, out_fname, save_file_res, vid_fmt, iou_thresh,
                                   save_animation, show_animation, show_text, show_stats, show_gt, show_only_tp,
                                   draw_plot, delete_tmp_files, set_class_iou,
                                   score_thresholds=score_thresholds)


if __name__ == '__main__':
    main()
