import cv2
import numpy as np
import pandas as pd
import time
import sys, os, shutil
from datetime import datetime
from pprint import pprint, pformat
import imageio
import subprocess
import random
import signal
import math
import ast
import multiprocessing
import functools

import logging

logging_fmt = logging.Formatter('%(levelname)s::%(module)s::%(funcName)s::%(lineno)s :  %(message)s')
logging_level = logging.INFO
logging.basicConfig(level=logging_level, format=logging_fmt)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

sys.path.append("..")

from utils import label_map_util
from utils import ops as utils_ops
from utils import visualization_utils as vis_util

import mAP.map_utils as map_utils
from mAP.mAP import evaluate

# from core.post_processing import multiclass_non_max_suppression

from utilities import processArguments, sortKey, resizeAR, readGT, readDetections, loadDetections


def getDetections(seq_paths, seq_to_samples, batch_size, sess, tensor_dict,
                  classes_to_include, category_index, category_index_dict,
                  save_dir='', load_det=0, save_det=0,
                  save_video=0, video_file_name='', codec='H264', vid_fps=30, vid_size='1280x720',
                  csv_file_name='', mask_out_path='', img_ext='jpg', min_score_thresh=0,
                  combine_sequences=0, input_size='', show_img=0, fullscreen=0, allow_seq_skipping=0,
                  iou_thresh=0.5, max_boxes_per_class=10, bgr_to_rgb=1, eval_every=0, start_seq_id=0,
                  class_agnostic=0):
    if tensor_dict is not None:
        image_tensor = tensor_dict['image_tensor']
        if 'detection_masks' in tensor_dict:
            enable_masks = 1
            tensor_dict['detection_masks_orig'] = tensor_dict['detection_masks']
        else:
            enable_masks = 0
    else:
        enable_masks = 0

    if load_det:
        save_det = 0
        enable_masks = 0
        batch_size = 1
        input_size = ''

    if input_size:
        input_size = tuple([int(x) for x in input_size.split('x')])
        print('Resizing all images to {}x{}'.format(*input_size))

    vid_size = tuple([int(x) for x in vid_size.split('x')])

    # if enable_nms:
    #     print('NMS is enabled with iou_thresh: {} and max_boxes_per_class: {}'.format(
    #     iou_thresh, max_boxes_per_class))

    # detection_boxes = tensor_dict['detection_boxes']
    # detection_scores = tensor_dict['detection_scores']
    # detection_classes = tensor_dict['detection_classes']
    # num_detections = tensor_dict['num_detections']
    # tensors = [detection_boxes, detection_scores, detection_classes, num_detections]

    # if 'detection_masks' in tensor_dict:
    #     detection_masks = tensor_dict['detection_masks']
    #     tensors.append(detection_masks)
    # else:
    #     detection_masks = None

    raw_det_data_dict = {}
    n_seq = len(seq_paths)

    if save_det:
        if not save_dir:
            save_dir = 'detections'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    combined_batch_id = 0
    combined_avg_fps = 0

    median_fps = median_t = 0

    _exit = 0
    _pause = 1

    win_title = 'space: toggle pause, q: next sequence, esc: quit'

    if show_img:
        if fullscreen:
            cv2.namedWindow(win_title, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(win_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            # cv2.namedWindow(win_title, cv2.WINDOW_GUI_NORMAL)
            try:
                cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_OPENGL)
            except cv2.error:
                cv2.namedWindow(win_title)

        def mouseHandler(event, x, y, flags=None, param=None):
            pass

        cv2.setMouseCallback(win_title, mouseHandler)

    if load_det:
        n_cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(n_cpus)
        print('Loading csv detections using {} threads'.format(n_cpus))
        _start_t = time.time()
        raw_det_data_list = pool.map(functools.partial(
            loadDetections,
            seq_to_samples=seq_to_samples,
            seq_paths=seq_paths,
            save_dir=save_dir,
            combine_sequences=combine_sequences,
            csv_file_name=csv_file_name,
            n_seq=n_seq,
            allow_seq_skipping=allow_seq_skipping,
            class_agnostic=class_agnostic,
        ), range(n_seq))
        _end_t = time.time()
        total_frames = np.sum(raw_det_data_list[i][1] for i in range(n_seq))
        fps = total_frames / float(_end_t - _start_t)
        print('Done reading data for {} frames at {:.4f} fps'.format(total_frames, fps))
        raw_det_data_dict = {seq_paths[i]: raw_det_data_list[i][0] for i in range(n_seq)}

    fps_list = []
    for seq_idx in range(start_seq_id, n_seq):
        seq_path = seq_paths[seq_idx]

        src_file_list = seq_to_samples[seq_idx]
        n_frames = len(src_file_list)

        seq_name = os.path.splitext(os.path.basename(seq_path))[0]

        if not load_det:
            print('sequence {}/{}: {}: '.format(seq_idx + 1, n_seq, seq_name))

        if not video_file_name:
            video_file_name = os.path.join(save_dir, '{}.mkv'.format(seq_name))

        video_out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_out = cv2.VideoWriter(video_file_name, fourcc, vid_fps, vid_size)
            if not video_out:
                raise SystemError('Output video file: {} could not be opened'.format(video_file_name))
            print('Saving {}x{} visualization video to {}'.format(vid_size[0], vid_size[1], video_file_name))

        if not csv_file_name:
            csv_file_name = os.path.join(save_dir, '{}.csv'.format(seq_name))

        if enable_masks:
            if not mask_out_path:
                mask_out_path = os.path.join(save_dir, '{}_masks'.format(seq_name))
            if os.path.isdir(mask_out_path):
                shutil.rmtree(mask_out_path)
            os.makedirs(mask_out_path)
            print('Saving masks to {}'.format(mask_out_path))

        if load_det:

            # if combine_sequences and not os.path.isfile(csv_file_name):
            #     _start_t = time.time()
            #     _seq_paths = list(set([os.path.dirname(k) for k in src_file_list]))
            #     raw_det_data_dict[seq_path] = []
            #     for _seq_path in _seq_paths:
            #         _seq_name = os.path.splitext(os.path.basename(_seq_path))[0]
            #         _src_file_list = [k for k in src_file_list if os.path.dirname(k) == _seq_path]
            #         _csv_file_name = os.path.join(save_dir, '{}.csv'.format(_seq_name))
            #         print('Loading csv detections from {}'.format(_csv_file_name))
            #         _start_t = time.time()
            #         raw_det_data_dict[seq_path] += readDetections(_seq_path, _csv_file_name, _src_file_list)
            #     _end_t = time.time()
            # else:
            #     print('Loading csv detections from {}'.format(csv_file_name))
            #     _start_t = time.time()
            #     raw_det_data_dict[seq_path] = readDetections(seq_path, csv_file_name, src_file_list)
            #     _end_t = time.time()
            # try:
            #     fps = float(n_frames) / float(_end_t - _start_t)
            #     print('fps: {:.4f} '.format(fps))
            # except ZeroDivisionError:
            #     print('fps: too high')

            mask_out_path = csv_file_name = ''

            if not show_img and not save_video:
                continue

            class_name_to_id = {v: k for k, v in category_index_dict.items()}

            print('class_name_to_id: {}'.format(pformat(class_name_to_id)))

            bounding_boxes_dict = {}
            for k in raw_det_data_dict[seq_path]:
                file_path = k['file_path']
                if file_path not in bounding_boxes_dict:
                    bounding_boxes_dict[file_path] = []
                bounding_boxes_dict[file_path].append(k)

            # print('raw_det_data_dict boxes: {}'.format(pformat(raw_det_data_dict)))
            # print('bounding_boxes_dict boxes: {}'.format(pformat(bounding_boxes_dict)))

        if save_det:
            csv_save_dir = os.path.dirname(csv_file_name)
            if not os.path.isdir(csv_save_dir):
                os.makedirs(csv_save_dir)
            print('Saving csv detections to {}'.format(csv_file_name))

        if not bgr_to_rgb:
            print('\n\nbgr_to_rgb is disabled\n\n')

        batch_id = 0
        avg_fps = 0
        frame_id = 0
        csv_raw = []
        exit_seq = 0

        bounding_boxes = []

        while frame_id < n_frames:
            overall_start_t = time.time()

            norm_factors = []
            # orig_image_sizes = []
            orig_images = []
            images = []
            file_paths = []
            for img_id in range(batch_size):
                if frame_id >= n_frames:
                    break
                cap_start_t = time.time()
                # file_name = os.path.basename(src_file_list[frame_id])
                file_path = src_file_list[frame_id]
                image_np = cv2.imread(src_file_list[frame_id])
                cap_end_t = time.time()
                cap_fps = 1.0 / float(cap_end_t - cap_start_t)

                orig_images.append(image_np)
                # orig_image_sizes.append(image_np.shape[:2])

                if input_size:
                    image_np, resize_factor, start_row, start_col = resizeAR(image_np, input_size[0], input_size[1],
                                                                             return_factors=1)
                    norm_factors.append((resize_factor, start_row, start_col))

                if bgr_to_rgb:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                images.append(image_np)
                file_paths.append(file_path)
                frame_id += 1

            curr_batch_size = len(images)

            if curr_batch_size == 0:
                break

            if load_det:
                boxes, scores, classes = [], [], []
                for img_id in range(curr_batch_size):
                    orig_height, orig_width = orig_images[img_id].shape[:2]
                    curr_boxes = []
                    curr_scores = []
                    curr_classes = []
                    curr_bb = bounding_boxes_dict[file_paths[img_id]]
                    for _obj in curr_bb:
                        _bbox = _obj["bbox"]

                        if _bbox is None:
                            print('None _bbox for {}'.format(file_paths[img_id]))
                            # curr_boxes.append([0, 0, 0, 0])
                            # curr_scores.append(0)
                            # curr_classes.append(0)
                            continue

                        xmin, ymin, xmax, ymax = _bbox
                        class_name = _obj["class"]
                        _score = _obj["confidence"]
                        _class = class_name_to_id[class_name]

                        xmin = float(xmin) / orig_width
                        xmax = float(xmax) / orig_width
                        ymin = float(ymin) / orig_height
                        ymax = float(ymax) / orig_height
                        curr_boxes.append([ymin, xmin, ymax, xmax])
                        curr_scores.append(_score)
                        curr_classes.append(_class)

                    boxes.append(curr_boxes)
                    scores.append(curr_scores)
                    classes.append(curr_classes)

                boxes = np.array(boxes, copy=False)
                scores = np.array(scores, copy=False)
                classes = np.array(classes, copy=False)
                valid_idx = list(range(curr_batch_size))
                # print('loaded boxes: {}'.format(pformat(boxes)))
                # print('loaded scores: {}'.format(pformat(scores)))
            else:
                # Actual detection
                tensor_names = [
                    'detection_boxes',
                    'detection_scores',
                    'detection_classes',
                    'num_detections',
                ]

                batch_failed = (curr_batch_size != batch_size)

                try:
                    if enable_masks:
                        raise ValueError('Masks only supported with single image processing')
                    images = np.array(images, copy=False)
                    _start_t = time.time()
                    tensor_val_dict = sess.run(tensor_dict, feed_dict={image_tensor: images})
                    _end_t = time.time()
                    fps = float(curr_batch_size) / float(_end_t - _start_t)
                except ValueError as e:
                    print('Batch run not possible: {}'.format(e))
                    batch_failed = 1
                    # images of different sizes
                    boxes = []
                    scores = []
                    classes = []
                    masks = []
                    valid_idx = []
                    time_taken = 0
                    for idx, image in enumerate(images):
                        img_h, img_w = image.shape[:2]
                        # if enable_masks:
                        #     detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                        #     detection_masks = tf.squeeze(tensor_dict['detection_masks_orig'], [0])
                        #     real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                        #     detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                        #     detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                        #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        #         detection_masks, detection_boxes, img_h, img_w)
                        #     detection_masks_reframed = tf.cast(
                        #         tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                        #     # Follow the convention by adding back the batch dimension
                        #     tensor_dict['detection_masks'] = tf.expand_dims(
                        #         detection_masks_reframed, 0)
                        _image = np.expand_dims(image, axis=0)

                        # print('Running on image of size {}x{} : {}'.format(img_h, img_w, file_names[idx]))
                        _start_t = time.time()
                        try:
                            tensor_val_dict = sess.run(tensor_dict, feed_dict={image_tensor: _image})
                        except tf.errors.ResourceExhaustedError:
                            print('Ran out of memory while processing image {}'.format(file_paths[idx]))
                            sys.exit()
                            # tensor_val_dict = {k: None for k in tensor_names}
                            # if enable_masks:
                            #     tensor_val_dict['detection_masks'] = None
                        else:
                            valid_idx.append(idx)
                        _end_t = time.time()

                        time_taken += _end_t - _start_t

                        (_boxes, _scores, _classes, num) = [tensor_val_dict[k] for k in tensor_names]

                        boxes.append(_boxes)
                        scores.append(_scores)
                        classes.append(_classes)
                        if enable_masks:
                            _masks = tensor_val_dict['detection_masks']
                            masks.append(_masks)

                    fps = float(curr_batch_size) / float(time_taken)
                else:
                    (boxes, scores, classes, num) = [tensor_val_dict[k] for k in tensor_names]
                    if enable_masks:
                        masks = tensor_val_dict['detection_masks']
                    valid_idx = list(range(curr_batch_size))

                batch_id += 1
                avg_fps += (fps - avg_fps) / float(batch_id)

                fps_list.append(fps)

                median_start_t = time.time()
                median_fps = np.median(fps_list)
                median_end_t = time.time()
                median_t = median_end_t - median_start_t

                if not batch_failed:
                    combined_batch_id += 1
                    combined_avg_fps += (fps - combined_avg_fps) / float(combined_batch_id)

            curr_batch_size = len(boxes)

            for _i in range(len(valid_idx)):
                img_id = valid_idx[_i]
                file_path = file_paths[img_id]
                filename = os.path.basename(file_path)

                curr_image = np.squeeze(images[img_id])
                orig_curr_image = np.squeeze(orig_images[img_id])

                curr_boxes = np.squeeze(boxes[img_id])
                curr_classes = np.squeeze(classes[img_id]).astype(np.int32)
                curr_scores = np.squeeze(scores[img_id])

                min_score = np.amin(curr_scores)
                max_score = np.amax(curr_scores)

                # print()
                # if enable_nms:
                #     _scores = np.asarray(scores[img_id])
                #     _boxes = np.asarray(boxes[img_id])
                #     # print('_scores: {}'.format(_scores))
                #     # print('_scores.shape: {}'.format(_scores.shape))
                #     # print('_boxes.shape: {}'.format(_boxes.shape))
                #     curr_boxes, n_nms_boxes = multiclass_non_max_suppression(_boxes, _scores,
                #                                                              min_score_thresh, iou_thresh,
                #                                                              max_boxes_per_class)

                try:
                    n_raw_objs = len(curr_scores)
                except TypeError:
                    n_raw_objs = 1

                if enable_masks:
                    curr_masks = np.squeeze(masks[img_id])

                # print('boxes: ', boxes)
                # print('curr_boxes: ', curr_boxes)
                # print('curr_boxes.shape: ', curr_boxes.shape)
                # print('classes: ', classes)
                # print('curr_classes: ', curr_classes)
                # print('scores: ', scores)
                # print('curr_scores: ', curr_scores)

                # print('n_raw_objs: ', n_raw_objs)

                height, width = curr_image.shape[:2]
                orig_height, orig_width = orig_curr_image.shape[:2]

                try:
                    n_objs = len(list(curr_classes))
                except TypeError:
                    n_objs = 1
                    curr_boxes = np.expand_dims(curr_boxes, 0)
                    curr_classes = np.expand_dims(curr_classes, 0)
                    curr_scores = np.expand_dims(curr_scores, 0)
                    if enable_masks:
                        curr_masks = np.expand_dims(curr_masks, 0)
                    # sys.exit()

                if classes_to_include:
                    _indices = [x for x in range(n_objs) if
                                category_index_dict[curr_classes[x]] in classes_to_include]
                    curr_scores = np.asarray([curr_scores[x] if x in _indices else 0 for x in range(n_objs)])

                _sort_idx = np.argsort(curr_scores, axis=0)[::-1]
                sort_idx = [x for x in _sort_idx if curr_scores[x] >= min_score_thresh]

                n_objs = len(sort_idx)

                # print('curr_scores: ', curr_scores)
                # print('curr_classes: ', curr_classes)
                # print('n_raw_objs: ', n_raw_objs)
                # print('n_objs: ', n_objs)

                if enable_masks:
                    if n_objs == 0:
                        print('No valid objects found for {}. Retaining the object with the highest score {}'.format(
                            file_path, curr_scores[_sort_idx[0]]))
                        sort_idx = np.asarray([_sort_idx[0], ])
                        n_objs = 1
                    elif n_objs > 1:
                        print('{} valid objects found for {} with scores {}. '
                              'Curtailing to the highest score object'.format(n_objs, file_path, curr_scores[sort_idx]))
                        sort_idx = np.asarray([sort_idx[0], ])
                        n_objs = 1

                    # print('curr_masks: {}'.format(curr_masks.shape))

                    # with open('curr_masks.txt', 'w') as fid:
                    #     pprint(curr_masks, fid)
                    # sys.exit()

                    # curr_masks = np.asarray(
                    #     [curr_masks[x] for x in range(n_objs) if curr_scores[x] > min_score_thresh])

                    curr_masks = curr_masks[sort_idx]

                curr_boxes = curr_boxes[sort_idx]
                curr_classes = curr_classes[sort_idx]
                curr_scores = curr_scores[sort_idx]

                if load_det:
                    _class = [_class + 1 for _class in curr_classes]

                # print('curr_boxes: ', curr_boxes)
                # print('curr_boxes.shape: ', curr_boxes.shape)
                # print('curr_classes: ', curr_classes)
                # print('curr_scores: ', curr_scores)
                # sys.exit()

                if input_size:
                    for _id in range(n_objs):
                        ymin, xmin, ymax, xmax = curr_boxes[_id]

                        xmin = xmin * width
                        xmax = xmax * width
                        ymin = ymin * height
                        ymax = ymax * height

                        resize_factor, start_row, start_col = norm_factors[img_id]
                        xmin = xmin / resize_factor - start_col
                        xmax = xmax / resize_factor - start_col
                        ymin = ymin / resize_factor - start_row
                        ymax = ymax / resize_factor - start_row

                        xmin = xmin / orig_width
                        xmax = xmax / orig_width
                        ymin = ymin / orig_height
                        ymax = ymax / orig_height

                        curr_boxes[_id] = [ymin, xmin, ymax, xmax]

                if save_video or show_img:
                    kw_args = {
                        'use_normalized_coordinates': True,
                        'line_thickness': 4,
                        'min_score_thresh': min_score_thresh,
                    }
                    # if enable_masks:
                    #     kw_args['instance_masks'] = curr_masks

                    vis_util.visualize_boxes_and_labels_on_image_array(
                        orig_curr_image,
                        curr_boxes,
                        curr_classes,
                        curr_scores,
                        category_index,
                        **kw_args
                    )

                    if save_video:
                        video_out.write(resizeAR(orig_curr_image, vid_size[0], vid_size[1]))

                # boxes_drawn = 0

                for _id in range(n_objs):
                    _box, _class, _score = curr_boxes[_id], curr_classes[_id], curr_scores[_id]

                    if _box is None:
                        continue

                    # if _score <= min_score_thresh:
                    #     continue

                    if enable_masks:
                        _mask = curr_masks[_id]

                        def reframe_box_masks_to_image_masks(box_mask, box, h, w):
                            def transform_boxes_relative_to_boxes(box, reference_boxes):
                                box = np.reshape(box, [2, 2])
                                min_corner = np.expand_dims(reference_boxes[0:2], 0)
                                max_corner = np.expand_dims(reference_boxes[2:4], 0)
                                transformed_boxes = (box - min_corner) / (max_corner - min_corner)
                                return np.reshape(transformed_boxes, [4, ])

                            box_mask = np.expand_dims(box_mask, axis=3)
                            unit_box = np.concatenate(
                                [np.zeros([1, 2]), np.ones([1, 2])], axis=1)
                            reverse_box = transform_boxes_relative_to_boxes(unit_box, box)

                            # print('box: {}'.format(box))
                            # print('unit_boxes: {}'.format(unit_boxes))
                            # print('reverse_box: {}'.format(reverse_box))

                            reverse_box = np.expand_dims(reverse_box, axis=0)
                            box_mask = np.expand_dims(box_mask, axis=0)

                            image_masks = tf.image.crop_and_resize(image=box_mask,
                                                                   boxes=reverse_box,
                                                                   box_ind=tf.range(1),
                                                                   crop_size=[h, w],
                                                                   extrapolation_value=0.0)

                            # image_masks = []
                            #
                            # xmin, ymin, xmax, ymax = reverse_box
                            # image_mask = box_mask[ymin:ymax, xmin:xmax]
                            # image_mask = cv2.resize(image_mask, (w, h))
                            # image_masks.append(image_mask)

                            # print('image_masks: {}'.format(image_masks))

                            return np.asarray(image_masks.eval(session=sess), dtype=np.float32)

                        _mask = np.squeeze(reframe_box_masks_to_image_masks(_mask, _box, height, width))

                        _mask = (_mask * 255.0).astype(np.uint8)

                        mask_fname = os.path.splitext(os.path.basename(file_path))[0] + '_{}.png'.format(_id)
                        cv2.imwrite(os.path.join(mask_out_path, mask_fname), _mask)
                        # print('_mask reframed: {}'.format(_mask))

                    try:
                        label = category_index[_class]['name']
                    except KeyError:
                        print('category_index: {}'.format(pformat(category_index)))
                        print('Ignoring box with unknown class: {}'.format(_class))
                        continue

                    ymin, xmin, ymax, xmax = _box

                    xmin = xmin * orig_width
                    xmax = xmax * orig_width
                    ymin = ymin * orig_height
                    ymax = ymax * orig_height

                    # if input_size:
                    #     resize_factor, start_row, start_col = norm_factors[img_id]
                    #     xmin = xmin / resize_factor - start_col
                    #     xmax = xmax / resize_factor - start_col
                    #     ymin = ymin / resize_factor - start_row
                    #     ymax = ymax / resize_factor - start_row
                    #
                    #     orig_h, orig_w = orig_image_sizes[img_id]

                    # orig_h, orig_w = height, width

                    bounding_boxes.append(
                        {"class": label,
                         "confidence": _score,
                         "file_path": file_path,
                         "bbox": [xmin, ymin, xmax, ymax]}
                    )
                    if save_det:
                        raw_data = {
                            'filename': filename,
                            'width': orig_width,
                            'height': orig_height,
                            'class': label,
                            'xmin': int(xmin),
                            'ymin': int(ymin),
                            'xmax': int(xmax),
                            'ymax': int(ymax),
                            'confidence': _score
                        }
                        csv_raw.append(raw_data)

                if show_img:
                    # if not boxes_drawn:
                    #     vis_util.visualize_boxes_and_labels_on_image_array(
                    #         curr_image,
                    #         curr_boxes,
                    #         curr_classes,
                    #         curr_scores,
                    #         category_index,
                    #         use_normalized_coordinates=True,
                    #         line_thickness=4)
                    #     boxes_drawn = 1

                    img_text = "{:s} frame {:d} fps: {:5.2f} avg_fps: {:5.2f}".format(
                        seq_name, frame_id, fps, avg_fps)

                    cv2.putText(orig_curr_image, img_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

                    # print('curr_boxes: {}'.format(pformat(curr_boxes)))

                    cv2.imshow(win_title, orig_curr_image)
                    if load_det:
                        _pause_dur = 10
                    else:
                        _pause_dur = 1

                    k = cv2.waitKey((1 - _pause) * _pause_dur)
                    if k == ord('q'):
                        exit_seq = 1
                        break
                    elif k == 27:
                        _exit = exit_seq = 1
                        break
                    elif k == 32:
                        _pause = 1 - _pause

            overall_end_t = time.time()
            overall_fps = float(curr_batch_size) / float(overall_end_t - overall_start_t)
            sys.stdout.write(
                '\rDone {:d}/{:d} frames fps: {:.4f} ({:.4f}) avg_fps: {:.4f} ({:.4f}) n_objs: {:d},{:d}'.format(
                    frame_id, n_frames, fps, overall_fps, avg_fps, combined_avg_fps, n_raw_objs, n_objs))
            sys.stdout.flush()

            # if show_img:
            #     cv2.destroyWindow(win_title)

            if exit_seq:
                break

        sys.stdout.write('\n')
        sys.stdout.flush()

        if save_video:
            print('Saved {}x{} visualization video to {}'.format(vid_size[0], vid_size[1], video_file_name))

        if enable_masks:
            print('Saved masks to {}'.format(mask_out_path))

        # valid_filenames = [os.path.basename(k) for k in src_file_list]
        if eval_every >= 0:
            det_file_paths = [k['file_path'] for k in bounding_boxes]
            skipped_file_paths = [k for k in src_file_list if k not in det_file_paths]
            for file_path in skipped_file_paths:
                bounding_boxes.append(
                    {"class": None,
                     "confidence": None,
                     "file_path": file_path,
                     "bbox": None}
                )
            raw_det_data_dict[seq_path] = bounding_boxes

        if save_video:
            video_out.release()
            video_file_name = ''

        if save_det:
            df = pd.DataFrame(csv_raw)
            out_file_path = os.path.join(csv_file_name)
            df.to_csv(out_file_path)

        print('combined_avg_fps: {}'.format(combined_avg_fps))
        mask_out_path = csv_file_name = ''

        if _exit:
            break

    # print('raw_det_data_dict: ', raw_det_data_dict)
    return raw_det_data_dict


def main():
    params = {
        'ckpt_path': '',
        'labels_path': '',

        'seq_paths': '',
        'root_dir': '',

        'seq_postfix': '',

        'eval_on_single_image': '',
        'start_frame_id': 0,
        'model_list': '',
        'model_id': 0,
        'save_dir': '',
        'save_file_name': '',
        'csv_file_name': '',

        # 'load_path': '',

        'n_classes': 4,
        'input_type': '',
        'batch_size': 1,
        'show_img': 0,
        'save_video': 0,
        'vid_size': '1280x720',
        'vis_height': 0,
        'n_frames': 0,
        'codec': 'H264',
        'fps': 20,
        'allow_memory_growth': 1,
        'gpu_memory_fraction': 1.0,

        'load_det': 0,
        'save_det': 1,

        'classes_to_include': [],
        'threaded_mode': 1,
        'rgb_mode': 1,
        'video_mode': 1,
        'fullscreen': 0,

        'load_samples_root': '',
        'load_samples': [],
        'sampling_ratio': 1.,
        'random_sampling': 0,
        'even_sampling': 0.,
        'samples_per_class': 0,
        'samples_per_seq': 0,
        'allow_seq_skipping': 0,

        'img_ext': 'jpg',

        'sleep_time': 600,
        'eval_every': 0,
        'pipeline_config_path': '',
        'inference_script_path': '~/models/research/object_detection/export_inference_graph.py',

        'train_pid': 0,
        'eval_dir': '',

        'write_summary': 1,
        'start_seq_id': 0,

        'inference_dir': 'inference',
        'trained_checkpoint_prefix': '',
        'out_postfix': '',
        'out_prefix': '',

        'inverted_sampling': 0,

        'enable_masks': 0,
        'frozen_graph_path': '',

        'save_animation': 0,
        'show_text': 1,
        'show_stats': 1,
        'show_gt': 1,
        'show_only_tp': 0,
        'show_animation': 0,
        'save_file_res': '1280x720',
        'draw_plot': 0,
        'results_path': '',
        'wt_avg_map': 0,

        'vid_ext': 'mkv',

        'load_dir': '',

        'combine_sequences': 0,
        'input_size': '',
        'score_thresholds': [0, ],
        'rec_ratios': [],
        # 'rec_ratios': list(np.arange(1.0,4.1,0.1)),

        # 'enable_nms': 0,
        'iou_thresh': 0.5,
        'max_boxes_per_class': 10,
        'ckpt_iters': 0,
        'n_threads': 0,
        'bgr_to_rgb': 1,

        'class_agnostic': 0,
    }

    processArguments(sys.argv[1:], params)

    seq_paths = params['seq_paths']
    root_dir = params['root_dir']

    seq_postfix = params['seq_postfix']

    _ckpt_path = params['ckpt_path']
    _labels_path = params['labels_path']
    _n_classes = params['n_classes']
    model_list = params['model_list']
    model_id = params['model_id']
    eval_on_single_image = params['eval_on_single_image']
    start_frame_id = params['start_frame_id']
    save_dir = params['save_dir']
    load_dir = params['load_dir']
    save_file_name = params['save_file_name']
    csv_file_name = params['csv_file_name']

    # load_path = params['load_path']

    input_type = params['input_type']
    batch_size = params['batch_size']
    show_img = params['show_img']
    save_video = params['save_video']
    vid_size = params['vid_size']
    n_frames = params['n_frames']
    codec = params['codec']
    fps = params['fps']
    allow_memory_growth = params['allow_memory_growth']
    gpu_memory_fraction = params['gpu_memory_fraction']
    load_det = params['load_det']
    save_det = params['save_det']
    classes_to_include = params['classes_to_include']
    rgb_mode = params['rgb_mode']
    video_mode = params['video_mode']
    fullscreen = params['fullscreen']
    img_ext = params['img_ext']

    load_samples_root = params['load_samples_root']
    load_samples = params['load_samples']
    sampling_ratio = params['sampling_ratio']
    random_sampling = params['random_sampling']
    even_sampling = params['even_sampling']
    samples_per_class = params['samples_per_class']
    samples_per_seq = params['samples_per_seq']
    allow_seq_skipping = params['allow_seq_skipping']

    bgr_to_rgb = params['bgr_to_rgb']

    sleep_time = params['sleep_time']
    eval_every = params['eval_every']
    pipeline_config_path = params['pipeline_config_path']
    inference_script_path = params['inference_script_path']

    train_pid = params['train_pid']
    eval_dir = params['eval_dir']

    write_summary = params['write_summary']
    start_seq_id = params['start_seq_id']

    trained_checkpoint_prefix = params['trained_checkpoint_prefix']
    out_postfix = params['out_postfix']
    out_prefix = params['out_prefix']
    inverted_sampling = params['inverted_sampling']

    enable_masks = params['enable_masks']
    frozen_graph_path = params['frozen_graph_path']

    save_animation = params['save_animation']
    show_text = params['show_text']
    show_stats = params['show_stats']
    show_gt = params['show_gt']
    show_only_tp = params['show_only_tp']
    show_animation = params['show_animation']
    save_file_res = params['save_file_res']
    draw_plot = params['draw_plot']
    results_path = params['results_path']

    vid_ext = params['vid_ext']

    combine_sequences = params['combine_sequences']
    input_size = params['input_size']
    score_thresholds = params['score_thresholds']
    rec_ratios = params['rec_ratios']

    # enable_nms = params['enable_nms']
    iou_thresh = params['iou_thresh']
    max_boxes_per_class = params['max_boxes_per_class']

    ckpt_iters = params['ckpt_iters']

    wt_avg_map = params['wt_avg_map']
    n_threads = params['n_threads']
    inference_dir = params['inference_dir']

    class_agnostic = params['class_agnostic']

    next_exported_iters = eval_every
    prev_ckpt_iters = -1

    n_score_thresholds = len(score_thresholds)
    min_score_thresh = score_thresholds[0]

    # if n_score_thresholds > 1:
    #     print('score_thresholds: {}'.format(score_thresholds))

    if show_img and fullscreen:
        from screeninfo import get_monitors

        monitors = get_monitors()
        curr_monitor = str(monitors[0])
        resolution = curr_monitor.split('(')[1].split('+')[0].split('x')

        _vis_width, _vis_height = [int(x) for x in resolution]

    if model_list:
        if not os.path.exists(model_list):
            raise IOError('Checkpoint list file: {} does not exist'.format(model_list))
        model_list = [x.strip().split(',') for x in open(model_list).readlines() if x.strip()]
        model_list = [[ckpt_path, labels_path, int(n_classes), ckpt_label]
                      for ckpt_path, labels_path, n_classes, ckpt_label in model_list]
    else:
        model_list = [[_ckpt_path, _labels_path, _n_classes, ''], ]
        model_id = 0

    ckpt_path, labels_path, n_classes, ckpt_label = model_list[model_id]
    if not load_det:
        if not os.path.exists(ckpt_path):
            raise IOError('Checkpoint file: {} does not exist'.format(ckpt_path))

    label_map = label_map_util.load_labelmap(labels_path)

    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=n_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    category_index_dict = {k['id']: k['name'] for k in category_index.values()}
    class_names = list(category_index_dict.values())

    print('category_index: {}'.format(pformat(category_index)))
    print('category_index_dict: {}'.format(pformat(category_index_dict)))

    video_exts = ['mp4', 'mkv', 'avi', 'mpg', 'mpeg', 'mjpg']

    if seq_paths:
        if os.path.isfile(seq_paths):
            if not out_postfix:
                out_postfix = os.path.splitext(os.path.basename(seq_paths))[0]
            seq_paths = [x.strip() for x in open(seq_paths).readlines() if x.strip()]
        else:
            seq_paths = seq_paths.split(',')
        if root_dir:
            seq_paths = [os.path.join(root_dir, x) for x in seq_paths]
    elif root_dir:
        if root_dir.startswith('camera'):
            seq_paths = [root_dir, ]
        else:
            seq_paths = []
            if input_type:
                if input_type == 'videos':
                    for ext in video_exts:
                        seq_paths += [os.path.join(root_dir, k) for k in os.listdir(root_dir) if
                                      not os.path.isdir(os.path.join(root_dir, k)) and k.endswith(
                                          '.{:s}'.format(ext))]
                    if len(seq_paths) == 0:
                        file_gen = [[os.path.join(dirpath, f) for f in filenames if
                                     os.path.splitext(f.lower())[1][1:] in video_exts]
                                    for (dirpath, dirnames, filenames) in os.walk(root_dir)]
                        seq_paths = [item for sublist in file_gen for item in sublist]
                        print('Here we are')
                else:
                    seq_paths += [os.path.join(root_dir, k) for k in os.listdir(root_dir) if
                                  not os.path.isdir(os.path.join(root_dir, k)) and k.endswith(
                                      '.{:s}'.format(input_type))]
            else:
                seq_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                             os.path.isdir(os.path.join(root_dir, name))]
            # seq_paths.sort(key=sortKey)
    else:
        raise IOError('Either seq_paths or root_dir must be specified')

    if seq_postfix:
        seq_paths = ['{}_{}'.format(seq_path, seq_postfix) for seq_path in seq_paths]

    seq_paths = [seq_path.replace('\\', '/') for seq_path in seq_paths]

    seq_paths.sort(key=sortKey)

    # if start_seq_id > 0:
    #     seq_paths = seq_paths[start_seq_id:]

    # print('seq_paths: ', seq_paths)
    n_seq = len(seq_paths)
    # print('Running over {} sequence(s)'.format(n_seq))

    gt_paths = [os.path.join(k, 'annotations.csv') for k in seq_paths]
    gt_paths = [gt_path.replace('\\', '/') for gt_path in gt_paths]

    seq_names = [os.path.basename(x) for x in seq_paths]

    sample_from_end = 0
    variable_sampling_ratio = 0

    if samples_per_seq != 0:
        sampling_ratio = 0
        samples_per_class = 0
        variable_sampling_ratio = 1
        if samples_per_seq < 0:
            samples_per_seq = -samples_per_seq
            print('Sampling from end')
            sample_from_end = 1
        print('Using variable sampling ratio to include {} samples per sequence'.format(samples_per_seq))

    if samples_per_class != 0:
        sampling_ratio = 0
        variable_sampling_ratio = 2
        if samples_per_class < 0:
            samples_per_class = -samples_per_class
            print('Sampling from end')
            sample_from_end = 1
        print('Using variable sampling ratio to include {} samples per class'.format(samples_per_class))

    if sampling_ratio < 0:
        print('Sampling from end')
        sample_from_end = 1
        sampling_ratio = -sampling_ratio
        print('sampling_ratio: ', sampling_ratio)

    if even_sampling != 0:
        print('Using evenly spaced sampling')
        random_sampling = 0
        if even_sampling < 0:
            even_sampling = -even_sampling
            inverted_sampling = 1

    if inverted_sampling:
        print('Inverting the sampling')

    assert sampling_ratio <= 1.0 and sampling_ratio >= 0.0, 'sampling_ratio must be between 0 and 1'

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = allow_memory_growth
    session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

    all_summary = []

    total_samples = 0
    total_files = 0

    seq_to_src_files = {}
    class_to_n_files = {_class: 0 for _class in class_names}
    class_to_n_files_per_seq = {_class: {} for _class in class_names}
    seq_to_n_files = {}

    def getClass(seq_path):
        if len(class_names) == 1:
            return class_names[0]
        for _class in class_names:
            if _class in os.path.basename(seq_path):
                return _class
        raise IOError('No class found for {}'.format(seq_path))

    seq_to_sampling_ratio = {k: sampling_ratio for k in seq_paths}
    seq_to_class = {k: getClass(k) for k in seq_paths}

    for seq_idx, seq_path in enumerate(seq_paths):

        seq_name = os.path.splitext(os.path.basename(seq_path))[0]
        if not os.path.isdir(seq_path):
            raise IOError('Image Sequence: {} does not exist'.format(seq_path))

        if eval_on_single_image:
            src_files = [os.path.join(seq_path, eval_on_single_image), ]
        else:
            src_files = [os.path.join(seq_path, k) for k in os.listdir(seq_path) if
                         os.path.splitext(k.lower())[1][1:] == img_ext]
            src_files.sort(key=sortKey)
            if start_frame_id > 0:
                src_files = src_files[start_frame_id:]

        n_files = len(src_files)
        seq_to_n_files[seq_path] = n_files

        src_files = [k.replace(os.sep, '/') for k in src_files]

        seq_to_src_files[seq_path] = src_files

        _class = seq_to_class[seq_path]
        class_to_n_files_per_seq[_class][seq_path] = n_files
        class_to_n_files[_class] += n_files

        total_files += n_files

    print('class_to_n_files:')
    pprint(class_to_n_files)

    if variable_sampling_ratio == 1:
        seq_to_sampling_ratio = {
            k: float(samples_per_seq) / float(seq_to_n_files[k]) for k in seq_paths
        }
    elif variable_sampling_ratio == 2:
        class_to_sampling_ratio = {k: float(samples_per_class) / class_to_n_files[k] for k in class_to_n_files}

        print('class_to_sampling_ratio:')
        pprint(class_to_sampling_ratio)

        seq_to_sampling_ratio = {
            k: class_to_sampling_ratio[seq_to_class[k]] for k in seq_paths
        }

    seq_to_samples = {}

    if len(load_samples) == 1:
        if load_samples[0] == 1:
            load_samples = ['seq_to_samples.txt', ]
        elif load_samples[0] == 0:
            load_samples = []

    if load_samples:
        # if load_samples == '1':
        #     load_samples = 'seq_to_samples.txt'
        print('load_samples: {}'.format(pformat(load_samples)))
        if load_samples_root:
            load_samples = [os.path.join(load_samples_root, k) for k in load_samples]
        print('Loading samples from : {}'.format(load_samples))
        for _f in load_samples:
            if os.path.isdir(_f):
                _f = os.path.join(_f, 'seq_to_samples.txt')
            with open(_f, 'r') as fid:
                curr_seq_to_samples = ast.literal_eval(fid.read())
                for _seq in curr_seq_to_samples:
                    if _seq in seq_to_samples:
                        seq_to_samples[_seq] += curr_seq_to_samples[_seq]
                    else:
                        seq_to_samples[_seq] = curr_seq_to_samples[_seq]

    class_to_n_samples = {_class: 0 for _class in class_names}

    all_sampled_files = []

    if combine_sequences:
        print('Combining all sequences into one')

    for seq_idx, seq_path in enumerate(seq_paths):

        src_files = seq_to_src_files[seq_path]
        n_files = seq_to_n_files[seq_path]
        sampling_ratio = seq_to_sampling_ratio[seq_path]

        if sampling_ratio > 1.0:
            raise IOError('Invalid sampling_ratio: {} for sequence: {} with {} files'.format(
                sampling_ratio, seq_path, n_files))

        if load_samples:
            try:
                sampled_files = seq_to_samples[seq_path]
            except KeyError:
                sampled_files = []
            if inverted_sampling:
                sampled_files = [k for k in src_files if k not in sampled_files]
            n_samples = len(sampled_files)
        else:
            n_samples = int(n_files * sampling_ratio)

            if sampling_ratio != 1.0:
                if n_samples == 0:
                    sampled_files = []
                else:
                    if random_sampling:
                        sampled_files = random.sample(src_files, n_samples)
                    elif even_sampling:
                        if sampling_ratio > even_sampling:
                            raise SystemError('{} :: sampling_ratio: {} is less than even_sampling: {}'.format(
                                seq_path, sampling_ratio, even_sampling))
                        sample_1_of_n = int(math.ceil(even_sampling / sampling_ratio))
                        end_file = int(n_files * even_sampling)

                        if sample_from_end:
                            sub_src_files = src_files[slice(-1, -end_file)]
                        else:
                            sub_src_files = src_files[slice(0, end_file)]

                        sampled_files = sub_src_files[::sample_1_of_n]

                        more_samples_needed = n_samples - len(sampled_files)
                        if more_samples_needed > 0:
                            unsampled_files = [k for k in sub_src_files if k not in sampled_files]
                            sampled_files += unsampled_files[:more_samples_needed]

                    else:
                        if sample_from_end:
                            sampled_files = src_files[-n_samples:]
                        else:
                            sampled_files = src_files[:n_samples]

                if inverted_sampling:
                    sampled_files = [k for k in src_files if k not in sampled_files]
            else:
                sampled_files = src_files

        if not sampled_files:
            msg = 'No sampled files found for {} with {} source files'.format(seq_path, n_files)
            if allow_seq_skipping:
                print('\n{}\n'.format(msg))
            else:
                raise IOError(msg)

        all_sampled_files += sampled_files
        seq_to_samples[seq_idx] = sampled_files
        actual_samples = len(sampled_files)
        class_to_n_samples[seq_to_class[seq_path]] += actual_samples
        total_samples += actual_samples

    print('class_to_n_samples: {}'.format(pformat(class_to_n_samples)))

    # print('seq_to_samples: {}'.format(pformat(seq_to_samples)))

    if load_det:
        if not load_dir and not ckpt_iters:
            if not ckpt_iters:
                # try to extract ckpt_iters from existing ckpt
                if not trained_checkpoint_prefix:
                    _trained_checkpoint_prefix = tf.train.latest_checkpoint(ckpt_path)
                else:
                    _trained_checkpoint_prefix = trained_checkpoint_prefix

                if _trained_checkpoint_prefix is None:
                    if frozen_graph_path:
                        _trained_checkpoint_prefix = os.path.basename(frozen_graph_path).split('_')[-1]
                if _trained_checkpoint_prefix is not None:
                    ckpt_name = os.path.basename(_trained_checkpoint_prefix)
                    try:
                        ckpt_iters = int(ckpt_name.split('-')[-1])
                    except:
                        ckpt_iters = 0

            if not ckpt_iters:
                ckpt_path_dirs = [k for k in os.listdir(ckpt_path) if os.path.isdir(os.path.join(ckpt_path, k))]
                if not ckpt_path_dirs:
                    raise IOError('No candidate folders for loading detections found in {}'.format(ckpt_path))
                load_dir = os.path.join(ckpt_path, ckpt_path_dirs[0])
            else:
                out_dir = '{}'.format(ckpt_iters)
                if out_postfix:
                    out_dir = '{}_on_{}'.format(out_dir, out_postfix)
                load_dir = os.path.join(ckpt_path, out_dir)
        if not os.path.isdir(load_dir):
            print('ckpt_iters: {}'.format(ckpt_iters))
            print('frozen_graph_path: {}'.format(frozen_graph_path))

            raise IOError('load_dir does not exist: {}'.format(load_dir))
        save_dir = load_dir

    if not eval_dir:
        if ckpt_path:
            eval_dir = ckpt_path
            # eval_dir = os.path.join(ckpt_path, 'tf_api_eval')
        elif save_dir:
            eval_dir = save_dir
        else:
            raise IOError('Either ckpt_path or save_dir must be provided')

    print('Evaluating over {} sequences with a total of {}/{} frames'.format(n_seq, total_samples, total_files))
    # proj_summary, proj_hist_summary = tf.py_func(decode_proj_plot, [proj_pred, proj_gt, n_images_to_show],

    eval_result_dict = {}
    gt_data_dict = {}

    # def get_summary(_c, _t):
    #     return np.float32(eval_result_dict[_c][_t])

    detection_graph = tf.Graph()

    summary_types = ['AP', 'Recall', 'Precision', 'TP', 'FN', 'FP']
    summary_classes = class_names + ['overall', ]

    # summary_dict = {}
    # vars = []
    # with detection_graph.as_default():
    #     with tf.Session(graph=detection_graph, config=session_config) as sess:
    #         for _class in summary_classes:
    #             summary_dict[_class] = {}
    #             for _type in summary_types:
    #                 _name = '{}_{}'.format(_class, _type)
    #                 var_name = 'var_{}'.format(_name)
    #                 var = tf.Variable(0.0, tf.float64, name=var_name)
    #                 vars.append(var)
    #                 summary_dict[_class][_type] = var
    #                 all_summary.append(
    #                     tf.summary.scalar(_name, summary_dict[_class][_type]))
    #         tf.initialize_all_variables().run(session=sess)
    # init = tf.variables_initializer(vars)
    # init = tf.global_variables_initializer()
    # sess.run(init)

    summary_writer = tf.summary.FileWriter(eval_dir)

    if frozen_graph_path and not os.path.isfile(frozen_graph_path):
        print('Frozen inference graph does not exist: {}'.format(frozen_graph_path))
        frozen_graph_path = ''

    if frozen_graph_path:
        evsl_every = 0

    prev_ckpt_iters = 0
    while True:
        if not trained_checkpoint_prefix:
            _trained_checkpoint_prefix = tf.train.latest_checkpoint(ckpt_path)
        else:
            _trained_checkpoint_prefix = trained_checkpoint_prefix

        if _trained_checkpoint_prefix is None:
            if frozen_graph_path:
                _trained_checkpoint_prefix = os.path.basename(frozen_graph_path).split('_')[-1]
            elif not load_det:
                raise IOError('No checkpoint found in {}'.format(ckpt_path))

        if _trained_checkpoint_prefix is not None:
            ckpt_name = os.path.basename(_trained_checkpoint_prefix)
            try:
                ckpt_iters = int(ckpt_name.split('-')[-1])
            except:
                pass
            else:
                inference_dir = '{}_{:d}'.format(inference_dir, ckpt_iters)

        if not load_det:
            if not frozen_graph_path:
                if eval_every > 0 and ckpt_iters < next_exported_iters:
                    # next_exported_iters = ckpt_iters + eval_every - (ckpt_iters % eval_every)
                    sys.stdout.write('\r {} :: Waiting for {} iterations ({} done)'.format(
                        eval_dir, next_exported_iters, ckpt_iters))
                    sys.stdout.flush()
                    time.sleep(sleep_time)
                    continue
                print()

                next_exported_iters = ckpt_iters + eval_every

                if prev_ckpt_iters == ckpt_iters:
                    continue

                output_directory = os.path.join(ckpt_path, inference_dir)

                frozen_graph_path = os.path.join(output_directory, 'frozen_inference_graph.pb')

                print('prev_ckpt_iters: ', prev_ckpt_iters)
                print('ckpt_iters: ', ckpt_iters)
                print('frozen_graph_path: ', frozen_graph_path)

                if train_pid:
                    # stop training
                    print('\n\nStopping training in process: {}\n\n'.format(train_pid))
                    os.kill(train_pid, signal.SIGINT)

                if os.path.exists(frozen_graph_path):
                    print('Frozen inference graph already exists: {}'.format(frozen_graph_path))
                else:
                    cmd = 'python3 {} --input_type image_tensor ' \
                          '--trained_checkpoint_prefix={}' \
                          ' --output_directory={} --pipeline_config_path={}'.format(
                        inference_script_path, _trained_checkpoint_prefix, output_directory,
                        pipeline_config_path)
                    print('Running inference:\n{}'.format(cmd))

                    subprocess.check_call(cmd, shell=True)

            prev_ckpt_iters = ckpt_iters

        summary_out_fname = ''
        if out_prefix:
            summary_out_fname = out_prefix
        if ckpt_iters:
            summary_out_fname = '{}_{}'.format(summary_out_fname, ckpt_iters) \
                if summary_out_fname else '{}'.format(ckpt_iters)
        if out_postfix:
            summary_out_fname = '{}_on_{}'.format(summary_out_fname, out_postfix) \
                if summary_out_fname else '{}'.format(out_postfix)

        if class_agnostic:
            print('Using class agnostic mode')
            summary_out_fname = '{}_ca'.format(summary_out_fname)

        summary_out_path = os.path.join(eval_dir, summary_out_fname + '.txt')
        # combined_summary_out_path = os.path.join(eval_dir, summary_out_fname + '_combined.txt')

        vis_out_path = os.path.join(eval_dir, summary_out_fname + '.{}'.format(vid_ext))
        if write_summary:
            print('Saving mAP results summary to {}'.format(summary_out_path))

        if not save_dir:
            if ckpt_label:
                out_dir = ckpt_label
            else:
                out_dir = '{}'.format(ckpt_iters)
            if out_postfix:
                out_dir = '{}_on_{}'.format(out_dir, out_postfix)

            save_dir = os.path.join(eval_dir, out_dir)

        print('save_dir: {}'.format(save_dir))
        print('ckpt_iters: {}'.format(ckpt_iters))
        print('ckpt_label: {}'.format(ckpt_label))
        print('out_postfix: {}'.format(out_postfix))
        print('eval_dir: {}'.format(eval_dir))

        # sys.exit()

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        sampled_fname = os.path.join(save_dir, 'sampled_files.txt')
        print('\nWriting sampled_files list to: {}\n'.format(sampled_fname))
        with open(sampled_fname, 'w') as fid:
            # print('sampled_files:')
            pprint(all_sampled_files, fid)

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph, config=session_config) as sess:
                if not load_det:
                    print('\nLoading inference graph from: {}'.format(frozen_graph_path))

                    od_graph_def = tf.GraphDef()
                    with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
                        serialized_graph = fid.read()
                        od_graph_def.ParseFromString(serialized_graph)
                        tf.import_graph_def(od_graph_def, name='')
                    tensor_names = [
                        'image_tensor',
                        'detection_boxes',
                        'detection_scores',
                        'detection_classes',
                        'num_detections',
                    ]
                    if enable_masks:
                        tensor_names.append('detection_masks')

                    tensor_dict = {}
                    for tensor_name in tensor_names:
                        tensor_dict[tensor_name] = detection_graph.get_tensor_by_name(tensor_name + ':0')
                    #
                    # if enable_masks:
                    #     detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    #     detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    #     real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    #     detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    #     detection_masks = tensor_dict['detection_masks']
                    #     detection_masks = tf.slice(detection_masks, [0, 0, 0], [num, -1, -1])
                    #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    #         detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    #     detection_masks_reframed = tf.cast(
                    #         tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    #     tensor_dict['detection_masks'] = tf.expand_dims(
                    #         detection_masks_reframed, 0)

                    # # Definite input and output Tensors for detection_graph
                    # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # # Each box represents a part of the image where a particular object was detected.
                    # detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # # Each score represent how level of confidence for each of the objects.
                    # # Score is shown on the result image, together with the class label.
                    # detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    # detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    # num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # if enable_masks:
                    #     detection_masks = detection_graph.get_tensor_by_name('detection_masks:0')
                    # else:
                    #     detection_masks = None
                else:
                    tensor_dict = None
                    # image_tensor = detection_boxes = detection_scores = \
                    #     detection_classes = num_detections = None

                if combine_sequences:
                    _seq_paths = ['combined_seq', ]
                    _seq_to_samples = {
                        0: all_sampled_files
                    }
                else:
                    _seq_paths = seq_paths
                    _seq_to_samples = seq_to_samples
                raw_det_data_dict = getDetections(_seq_paths, _seq_to_samples, batch_size, sess,
                                                  tensor_dict,
                                                  classes_to_include, category_index, category_index_dict,
                                                  load_det=load_det,
                                                  save_det=save_det,
                                                  save_dir=save_dir,
                                                  save_video=save_video,
                                                  vid_size=vid_size,
                                                  combine_sequences=combine_sequences,
                                                  input_size=input_size,
                                                  show_img=show_img,
                                                  fullscreen=fullscreen,
                                                  min_score_thresh=min_score_thresh,
                                                  # enable_nms=enable_nms,
                                                  iou_thresh=iou_thresh,
                                                  max_boxes_per_class=max_boxes_per_class,
                                                  allow_seq_skipping=allow_seq_skipping,
                                                  bgr_to_rgb=bgr_to_rgb,
                                                  eval_every=eval_every,
                                                  start_seq_id=start_seq_id,
                                                  class_agnostic=class_agnostic,
                                                  )

        if eval_every < 0:
            break

        if not gt_data_dict:
            gt_data_dict = readGT(gt_paths, seq_paths, seq_to_samples,
                                  class_names, combine_sequences, class_agnostic)

        # scores_to_ap = {}
        # scores_to_recall = {}
        # scores_to_prec = {}
        # scores_to_eval = {}
        # rec_prec_ap = {}
        # out_text = ''

        if class_agnostic:
            _class_names = ['generic', ]
        else:
            _class_names = class_names

        # for score_idx, score_thresh in enumerate((score_thresholds[0], )):
        #     print('Evaluating for score_threshold: {}'.format(score_thresh))
        eval_result_dict, eval_text = evaluate(_seq_paths, _class_names,
                                               _gt_data_dict=gt_data_dict,
                                               raw_det_data_dict=raw_det_data_dict,
                                               save_dir=eval_dir,
                                               write_summary=write_summary,
                                               out_fname=summary_out_path,
                                               vis_out_fname=vis_out_path,
                                               # eval_result_dict=eval_result_dict,
                                               save_animation=save_animation,
                                               show_text=show_text,
                                               show_stats=show_stats,
                                               show_gt=show_gt,
                                               show_only_tp=show_only_tp,
                                               show_animation=show_animation,
                                               save_file_res=save_file_res,
                                               draw_plot=draw_plot,
                                               results_path=results_path,
                                               score_thresholds=score_thresholds,
                                               rec_ratios=rec_ratios,
                                               wt_avg=wt_avg_map,
                                               n_threads=n_threads,
                                               )
        # scores_to_eval[score_thresh] = eval_result_dict
        # out_text += '{}\n{}\n'.format(score_thresh, eval_text)

        # for _class in eval_result_dict:
        #     _eval = eval_result_dict[_class]
        #     rec, prec, ap = _eval['Recall'], _eval['Precision'], _eval['AP']
        #
        #     if _class not in rec_prec_ap:
        #         rec_prec_ap[_class] = np.zeros((n_score_thresholds, 4))
        #     rec_prec_ap[_class][score_idx, :] = [score_thresh, rec, prec, ap]

        # if n_score_thresholds > 1:
        #     # print('rec_prec_ap:\n{}'.format(pformat(rec_prec_ap)))
        #     # print('scores_to_prec:\n{}'.format(pformat(scores_to_prec)))
        #     # print('scores_to_recall:\n{}'.format(pformat(scores_to_recall)))
        #     # print('scores_to_ap:\n{}'.format(pformat(scores_to_ap)))

        # if n_score_thresholds > 1:
        #     overall_rec = rec_prec_ap['overall'][:, 1][::-1]/100.0
        #     overall_prec = rec_prec_ap['overall'][:, 2][::-1]/100.0
        #     _score_thresh = rec_prec_ap['overall'][:, 0][::-1]/100.0
        #     overall_ap, _, _ = map_utils.voc_ap(overall_rec, overall_prec)
        #     idx = np.argwhere(np.diff(np.sign(overall_rec - overall_prec))).flatten()
        #     print('idx: {}'.format(idx))
        #     print('Intersection at {} for confidence: {}'.format(overall_rec[idx], _score_thresh[idx]))
        #     print('overall_ap: {}'.format(overall_ap))
        #
        # out_text += '\n'
        # for _class in rec_prec_ap:
        #     out_text += '{}:\n'.format(_class)
        #     # out_text += 'scores_to_ap\n{}\n'.format(
        #     #     pd.DataFrame(data=scores_to_ap[_class],
        #     #                  columns=['score_thresh', 'AP']).to_csv(sep='\t', , index =False))
        #     out_text += 'rec_prec_ap\n{}\n'.format(
        #         pd.DataFrame(data=rec_prec_ap[_class],
        #                      columns=['score_thresh', 'recall', 'precision', 'AP']).to_csv(
        #             sep='\t', index =False))
        #     out_text += '\n'
        #
        # print('Writing combined summary to: {}'.format(combined_summary_out_path))
        # with open(combined_summary_out_path, 'w') as fid:
        #     fid.write(out_text)

        # print('gt_data_dict: ')
        # pprint(gt_data_dict)
        # print('raw_det_data_dict: ')
        # pprint(raw_det_data_dict)
        # print('eval_result_dict: ')
        # pprint(eval_result_dict)
        # print(eval_table)
        if eval_every > 0:
            all_summary = []
            for _class in summary_classes:
                for _type in summary_types:
                    _name = '{}_{}'.format(_class, _type)
                    all_summary.append(
                        tf.summary.scalar(_name, tf.convert_to_tensor(float(eval_result_dict[_class][_type])))
                    )

            with detection_graph.as_default():
                with tf.Session(graph=detection_graph, config=session_config) as sess:
                    # result = []
                    # tf.initialize_all_variables().run(session=sess)
                    # init = tf.variables_initializer(vars)
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    # for _class in summary_classes:
                    #     for _type in summary_types:
                    #         tf.assign(summary_dict[_class][_type], )
                    # result.append(summary_dict[_class][_type].eval())

                    # result = sess.run(all_summary)
                    # print('summary_dict: {}'.format(summary_dict))
                    # print('all_summary: {}'.format(all_summary))

                    result = all_summary
                    for _summary in result:
                        summary_writer.add_summary(_summary.eval(), ckpt_iters)
        if not eval_every:
            break

        tf.reset_default_graph()

        if train_pid:
            # start training
            print('\n\nResuming training in process: {}\n\n'.format(train_pid))
            os.kill(train_pid, signal.SIGINT)


if __name__ == '__main__':
    main()
