import cv2
import numpy as np
import sys
import pandas as pd
import pickle
import math

sys.path.append("..")

from tf_api.utilities import processArguments, sortKey, resizeAR, map_mask_to_bbox, fix_bbox
from tf_api.utilities import get2DGaussianErrorFunctionArray, get2DGaussianArray, get2DGaussianArray2, hist_match, \
    hist_match2, compareHist, addBorder, getIOU
import os
import random
import time
import inspect
from pprint import pprint
from PIL import Image
from color_transfer import color_transfer
from tracking.Visualizer import ImageWriter
from libs.pascal_voc_io import PascalVocWriter
from libs.shape import Shape

ui_help = {
    'left_button+drag': 'move the object around',
    'right_button': 'flip the object horizontally',
    'shift+right_button': 'flip the object vertically',
    'ctrl+left_button': 'rotate the object anti clockwise',
    'ctrl+right_button': 'rotate the object clockwise',
    'alt+right_button': 'toggle bounding box visibility',
    'wheel': 'change the object size',
    'shift+wheel / up_arrow,down_arrow / 8,2': 'move the object vertically',
    'ctrl+wheel / right_arrow,left_arrow / 4,6': 'move the object horizontally',
    'ctrl+shift+wheel / 7,1': 'move the object diagonally towards the left',
    'ctrl+alt+wheel / ctrl+shift+right_button+wheel / 9,3': 'move the object diagonally towards the right',
    'ctrl+alt+shift+wheel': 'change the magnified window size without changing the magnified patch area',
    'q': 'exit the tool',
    'escape': 'go to the next image',
    'v': 'toggle visualization',
    'space_bar': 'toggle pausing for adjustment in each frame',
}

def generateAugmentedSequence(src_files, src_data_dict, bkg_files, bkg_data_dict, bkg_imgs,
                              static_bkg=1, boxes_per_bkg=1, random_bkg_box=0,
                              only_one_src_obj=0, aug_seq_size=1, fixed_ar=0, border=0,
                              aug_seq_prefix='', save_path='', hist_match_type=0,
                              show_blended=1, show_bbox=0,
                              save_seq=1, visualize=1, vis_size='', raw_mask=0, map_to_bbox=0,
                              apply_mask_contour=0, resize_ratio_diff=0.01, trans_diff=1,
                              rot_diff=5):
    print('aug_seq_size: {}'.format(aug_seq_size))

    if vis_size:
        vis_width, vis_height = [int(x) for x in vis_size.split('x')]
    else:
        vis_width, vis_height = 1920, 1080

    if abs(hist_match_type) == 1:
        _hist_match = hist_match
    elif abs(hist_match_type) == 2:
        _hist_match = hist_match2
    elif abs(hist_match_type) == 3:
        _hist_match = color_transfer
    else:
        _hist_match = None

    src_id = -1
    video_out = None

    pause_after_frame = 1

    frame_id = aug_frame_id = 0
    aug_seq_id = 1

    n_bkgs = len(bkg_files)
    bkg_ids = list(range(n_bkgs))

    n_src_files = len(src_files)
    print('Running over {} source files'.format(n_src_files))

    prev_src_bbox = None

    dst_img = dst_patch = dst_patch_mask = None
    dst_img_list = []

    dst_width = dst_height = dst_xmin = dst_ymin = dst_xmax = dst_ymax = 0

    # only one box per source image for now
    src_box_id = 0
    _boxes_per_bkg = boxes_per_bkg

    win_names = ['src_img', 'augmented_image', 'dst_patch_mask', 'dst_patch']
    if visualize and hist_match_type > 0:
        win_names.append('src_img_matched')

    start_t = time.time()
    time_taken = 0

    offset_x = offset_y = 0
    resize_ratio = 0
    flip_lr = flip_ud = 0
    # rot_cw = rot_acw = 0
    rot_angle = 0

    while src_id < n_src_files - 1:
        src_id += 1

        src_img_path = src_files[src_id]
        src_seq_path = os.path.dirname(src_img_path)
        src_seq_name = os.path.splitext(os.path.basename(src_seq_path))[0]

        src_fname_no_ext = os.path.splitext(os.path.basename(src_img_path))[0]

        # src_img_path = os.path.join(src_path, src_fname)
        # src_img = cv2.imread(src_img_path)

        src_objs = src_data_dict[src_img_path]

        src_file = src_objs['img']

        # print('reading src_img from {}'.format(src_file))

        if isinstance(src_file, str):
            src_img = cv2.imread(src_file)
            if src_img is None:
                raise IOError('img_file could not be read: {}'.format(src_file))
        else:
            src_img = src_file

        img_h, img_w = src_img.shape[:2]
        src_objs = src_objs['data']
        n_objs = len(src_objs)

        if n_objs == 0:
            raise IOError('No boxes found for {}'.format(src_img_path))

        # if only_one_src_obj:
        #     n_objs = 1

        src_obj = src_objs[src_box_id]
        src_bbox = src_obj['bbox']
        class_name = src_obj['class_name']
        target_id = src_obj['target_id']
        seq_id = src_obj['seq_id']
        mask_file = src_obj['mask']

        # print('reading mask from {}'.format(mask_file))

        if isinstance(mask_file, str):
            mask = cv2.imread(mask_file)
            if mask is None:
                raise IOError('mask_file could not be read: {}'.format(mask_file))
        else:
            mask = mask_file

        if apply_mask_contour:
            _pts, _ = Shape.contourPtsFromMask(mask)
            mask, _ = Shape.contourPtsToMask(_pts, mask)

        if visualize:
            if map_to_bbox:
                mask_img = map_mask_to_bbox(src_bbox, mask, img_shape=src_img.shape)
            else:
                mask_img = mask
            if show_blended:
                blended_img = np.asarray(Image.blend(Image.fromarray(src_img), Image.fromarray(mask_img), 0.5))
                if show_bbox:
                    xmin, ymin, xmax, ymax = src_bbox
                    cv2.rectangle(blended_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            else:
                blended_img = np.concatenate((src_img, mask_img), axis=1)
                if show_bbox:
                    xmin, ymin, xmax, ymax = src_bbox
                    cv2.rectangle(blended_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            _src_img = resizeAR(blended_img, vis_width, vis_height)
            cv2.imshow('src_img', _src_img)

        if not raw_mask:
            mask = mask.astype(np.float32) / 255.0

        if static_bkg > 0:
            curr_bkg_imgs = [bkg_imgs[static_bkg - 1], ]
        elif static_bkg == 0:
            curr_bkg_imgs = [bkg_imgs[bkg_ids[src_id]], ]
        else:
            curr_bkg_imgs = bkg_imgs

        frame_id += 1

        n_curr_bkg_imgs = len(curr_bkg_imgs)
        if n_curr_bkg_imgs > 1:
            print('Running over {} backgrounds'.format(n_curr_bkg_imgs))

        for bkg_img_dict in curr_bkg_imgs:
            aug_frame_id += 1

            bkg_img = bkg_img_dict['image']
            bkg_fname = bkg_img_dict['name']

            bkg_resize_factor = bkg_img_dict['resize_factor']
            bkg_orig_shape = bkg_img_dict['orig_shape']
            bkg_img_h, bkg_img_w = bkg_img.shape[:2]
            bkg_boxes = bkg_data_dict[bkg_fname]

            bkg_seq_path = os.path.dirname(bkg_fname)
            bkg_seq_name = os.path.basename(bkg_seq_path)

            bkg_fname_no_ext = os.path.splitext(os.path.basename(bkg_fname))[0]

            # bkg_xmin, bkg_ymin, bkg_xmax, bkg_ymax = src_bbox
            # _src_width, _src_height = bkg_xmax - bkg_xmin, bkg_ymax - bkg_ymin

            # src_bbox = addBorder(src_bbox, src_img, border_ratio, make_square)

            src_xmin, src_ymin, src_xmax, src_ymax = src_bbox
            if src_xmin < 0 or src_ymin < 0 or src_xmax >= img_w or src_ymax >= img_h:
                raise IOError('Invalid bounding box: {} for image of size {}'.format(
                    src_bbox, (img_h, img_w)
                ))
            src_width, src_height = src_xmax - src_xmin, src_ymax - src_ymin
            src_ar = float(src_width) / float(src_height)

            if hist_match_type > 0:
                # bkg_img_matched = np.zeros_like(bkg_img, dtype=np.uint8)
                # for ch_id in range(3):
                #     bkg_img_matched[:, :, ch_id] = _hist_match(bkg_img[:, :, ch_id].squeeze(),
                #                                               src_img[:, :, ch_id].squeeze())
                # bkg_img = bkg_img_matched

                src_img_matched = _hist_match(bkg_img, src_img)
                src_patch = np.copy(src_img_matched[int(src_ymin):int(src_ymax), int(src_xmin):int(src_xmax), :])

                if visualize:
                    cv2.imshow('src_img_matched', src_img_matched)
            else:
                src_patch = np.copy(src_img[int(src_ymin):int(src_ymax), int(src_xmin):int(src_xmax), :])

            n_bkg_boxes = len(bkg_boxes)
            if _boxes_per_bkg <= 0:
                boxes_per_bkg = n_bkg_boxes
            else:
                boxes_per_bkg = _boxes_per_bkg

            if boxes_per_bkg > n_bkg_boxes:
                bkg_ar_list = [(bkg_xmax - bkg_xmin) / (bkg_ymax - bkg_ymin) for
                               bkg_xmin, bkg_ymin, bkg_xmax, bkg_ymax in bkg_boxes]
                n_reps = math.ceil(float(boxes_per_bkg) / n_bkg_boxes)
                bkg_boxes *= n_reps
            bkg_boxes = bkg_boxes[:boxes_per_bkg]

            # if n_curr_bkg_imgs > 1:
            #     offset_x = offset_y = 0
            #     resize_ratio = 0
            #     flip_lr = flip_ud = 0
            #     rot_cw = rot_acw = 0
            #     rot_angle = 5

            for bkg_box_id in range(boxes_per_bkg):

                if save_seq:
                    if video_out is None:
                        if not aug_seq_prefix:
                            _aug_seq_prefix = '{}_augmented_mask_{}'.format(src_seq_name, bkg_seq_name)
                        else:
                            _aug_seq_prefix = aug_seq_prefix

                        aug_seq_name = _aug_seq_prefix
                        if static_bkg > 0:
                            aug_seq_name = '{:s}_static_{:d}'.format(aug_seq_name, static_bkg)

                        aug_seq_name = '{:s}_{:d}'.format(aug_seq_name, aug_seq_id)
                        aug_seq_path = os.path.join(save_path, aug_seq_name)

                        if not os.path.isdir(aug_seq_path):
                            os.makedirs(aug_seq_path)

                        video_out = ImageWriter(aug_seq_path)
                        print('\nSaving augmented sequence {} to {}'.format(aug_seq_id, aug_seq_path))
                        aug_xml_path = os.path.join(aug_seq_path, 'annotations')
                        if not os.path.isdir(aug_xml_path):
                            os.makedirs(aug_xml_path)

                        aug_frame_id = 0
                        dst_id = 0

                    xml_writer = PascalVocWriter(aug_xml_path, None, None)

                    out_prefix = '{}_{}_{}_{}_{}'.format(
                        src_seq_name, src_fname_no_ext, bkg_seq_name, bkg_fname_no_ext, bkg_box_id)
                    if pause_after_frame:
                        print('Saving image to:  {}'.format(out_prefix))

                bkg_bbox_raw = bkg_boxes[bkg_box_id]

                bkg_bbox_orig = [int(k / bkg_resize_factor) for k in bkg_bbox_raw]
                # bkg_bbox = addBorder(bkg_bbox_orig, bkg_img, border_ratio, make_square)
                bkg_bbox = bkg_bbox_orig

                bkg_xmin, bkg_ymin, bkg_xmax, bkg_ymax = [int(x) for x in bkg_bbox]

                bkg_width, bkg_height = bkg_xmax - bkg_xmin, bkg_ymax - bkg_ymin

                if not resize_ratio:
                    # match bkg aspect ratio to src
                    bkg_ar = float(bkg_width) / float(bkg_height)

                    if src_ar < bkg_ar:
                        dst_height = bkg_height
                        dst_width = int(dst_height * src_ar)
                    else:
                        dst_width = bkg_width
                        dst_height = int(dst_width / src_ar)
                    resize_ratio = float(dst_width) / float(src_width)

                def updateAugmentedPatchAndMask():
                    nonlocal dst_width, dst_height, dst_patch, dst_patch_mask
                    # nonlocal rot_cw, rot_acw

                    dst_width = int(src_width * resize_ratio)
                    dst_height = int(src_height * resize_ratio)
                    try:
                        dst_patch = cv2.resize(src_patch, (dst_width, dst_height))
                    except cv2.error as e:
                        print()
                        print('bkg_orig_shape', bkg_orig_shape)
                        print('src_patch.shape', src_patch.shape)
                        print('bkg_bbox_orig', bkg_bbox_orig)
                        print('bkg_bbox', bkg_bbox)
                        print('bkg_resize_factor', bkg_resize_factor)
                        print('dst_width', dst_width)
                        print('dst_height', dst_height)
                        print('cv2.error: {}'.format(e))
                        sys.exit()

                    # if mask_loaded:
                    dst_patch_mask = cv2.resize(mask, (dst_width, dst_height)).astype(np.float64)

                    if rot_angle != 0:
                        dst_patch = np.asarray(
                            Image.fromarray(dst_patch).rotate(rot_angle))
                        dst_patch_mask = np.asarray(
                            Image.fromarray(dst_patch_mask.astype(np.uint8)).rotate(rot_angle), dtype=np.float64)
                        # rot_cw = 0

                    # if rot_acw:
                    #     dst_patch = np.asarray(
                    #         Image.fromarray(dst_patch).rotate(-rot_angle))
                    #     dst_patch_mask = np.asarray(
                    #         Image.fromarray(dst_patch_mask.astype(np.uint8)).rotate(-rot_angle), dtype=np.float64)
                    #     # rot_acw = 0

                    if flip_lr:
                        # print('Flipping horizontally')
                        dst_patch = np.fliplr(dst_patch)
                        dst_patch_mask = np.fliplr(dst_patch_mask)

                    if flip_ud:
                        # print('Flipping vertically')
                        dst_patch = np.flipud(dst_patch)
                        dst_patch_mask = np.flipud(dst_patch_mask)

                    # else:
                    #     dst_mask_pts = [[(x - src_xmin) * resize_ratio, (y - src_ymin) * resize_ratio] for x, y in mask]
                    #     dst_patch_mask = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
                    #
                    #     dst_patch_mask = cv2.fillPoly(dst_patch_mask, np.array([dst_mask_pts, ], dtype=np.int32),
                    #                                   (1, 1, 1)).astype(np.float64)

                def updateAugmentedImage():
                    nonlocal dst_img, dst_xmin, dst_ymin, dst_xmax, dst_ymax
                    # shift box if outside bkg image bounds
                    start_row, start_col = max(0, bkg_ymin + offset_y), max(0, bkg_xmin + offset_x)

                    if prev_src_bbox is not None:
                        prev_src_xmin, prev_src_ymin, _, _ = prev_src_bbox

                        tx = src_xmin - prev_src_xmin
                        ty = src_ymin - prev_src_ymin

                        start_row = int(start_row + ty * resize_ratio)
                        start_col = int(start_col + tx * resize_ratio)

                    end_row, end_col = start_row + dst_height, start_col + dst_width

                    if end_col > bkg_img_w:
                        diff = end_col - bkg_img_w
                        start_col -= diff
                        end_col -= diff

                    if end_row > bkg_img_h:
                        diff = end_row - bkg_img_h
                        start_row -= diff
                        end_row -= diff

                    if start_row < 0 or start_col < 0:
                        print('Invalid start location: {}, {}'.format(start_row, start_col))
                        print('end_row: {}'.format(end_row))
                        print('end_col: {}'.format(end_col))
                        print('bkg_img_w: {}'.format(bkg_img_w))
                        print('bkg_img_h: {}'.format(bkg_img_h))
                        print('offset_y: {}'.format(offset_y))
                        print('offset_x: {}'.format(offset_x))
                        # bkg_box_id = (bkg_box_id + 1) % n_bkg_boxes
                        return False

                    bkg_patch = bkg_img[start_row:end_row, start_col:end_col, :]

                    if hist_match_type < 0:
                        dst_patch_matched = _hist_match(bkg_patch, dst_patch)
                    else:
                        dst_patch_matched = dst_patch

                    blended_patch = cv2.add(
                        cv2.multiply(1.0 - dst_patch_mask, bkg_patch.astype(np.float64)),
                        cv2.multiply(dst_patch_mask, dst_patch_matched.astype(np.float64))).astype(np.uint8)

                    dst_img = np.copy(bkg_img)
                    dst_img[start_row:end_row, start_col:end_col, :] = blended_patch

                    dst_xmin = int(start_col)
                    dst_ymin = int(start_row)

                    dst_xmax = int(dst_xmin + src_width * resize_ratio)
                    dst_ymax = int(dst_ymin + src_height * resize_ratio)

                    return True

                updateAugmentedPatchAndMask()
                if not updateAugmentedImage():
                    continue

                frame_id += 1
                aug_frame_id += 1

                prev_mouse_pt = []

                lbutton_keys_to_flags = {
                    'none': 1,
                    'ctrl': 9,
                    'shift': 17,
                    'ctrl+shift': 25,
                    'alt': 33,
                    'alt+ctrl': 41,
                    'alt+ctrl+shift': 57,
                }

                rbutton_keys_to_flags = {
                    'none': 2,
                    'ctrl': 10,
                    'shift': 18,
                    'ctrl+shift': 26,
                    'alt': 34,
                    'alt+ctrl': 42,
                    'alt+ctrl+shift': 58,
                }
                mouse_whl_keys_to_flags = {
                    'none': (7864320, -7864320),
                    'ctrl': (7864328, -7864312),
                    'alt': (7864352, -7864288),
                    'shift': (7864336, -7864304),
                    'ctrl+alt': (7864360, -7864280),
                    'ctrl+shift': (7864344, -7864296),
                    'ctrl+shift+rbutton': (7864346, -7864294),
                    'alt+shift': (7864368, -7864272),
                    'ctrl+alt+shift': (7864376, -7864264),
                }


                def mouseHandler(event, x, y, flags=None, param=None):
                    nonlocal prev_mouse_pt, offset_x, offset_y, resize_ratio
                    nonlocal dst_img, dst_xmin, dst_ymin, dst_xmax, dst_ymax, visualize
                    nonlocal flip_lr, flip_ud, prev_mouse_pt
                    nonlocal rot_angle, show_bbox

                    if event != cv2.EVENT_MOUSEMOVE:
                        prev_mouse_pt = []

                    if event == cv2.EVENT_LBUTTONDOWN:
                        # print('flags: {}'.format(flags))
                        if flags == lbutton_keys_to_flags['ctrl']:
                            # rot_acw = 1
                            rot_angle += rot_diff
                            showAugmentedImage(update=1)
                            # elif flags == keys_to_flags['ctrl+shift']:
                            # rot_angle -= 5
                            # showAugmentedImage(update=1)
                    if event == cv2.EVENT_RBUTTONDOWN:
                        # print('flags: {}'.format(flags))
                        # print('flags: {0:b}'.format(flags))

                        if flags == rbutton_keys_to_flags['alt']:
                            show_bbox = 1 - show_bbox
                        elif flags == rbutton_keys_to_flags['none']:
                            flip_lr = 1 - flip_lr
                            showAugmentedImage(update=1)
                        elif flags == rbutton_keys_to_flags['shift']:
                            flip_ud = 1 - flip_ud
                            showAugmentedImage(update=1)
                        elif flags == rbutton_keys_to_flags['ctrl']:
                            # rot_cw = 1
                            rot_angle -= rot_diff
                            showAugmentedImage(update=1)
                            # elif flags == keys_to_flags['ctrl+shift']:
                            rot_cw = 1
                            # rot_angle = max(0, rot_angle - 5)
                            # showAugmentedImage(update=1)

                    elif event == cv2.EVENT_MOUSEMOVE:
                        if flags == 1 or flags == 25:
                            # left mouse
                            if prev_mouse_pt:
                                tx, ty = x - prev_mouse_pt[0], y - prev_mouse_pt[1]

                                offset_x += tx
                                offset_y += ty

                                if not updateAugmentedImage():
                                    return
                                showAugmentedImage()
                            prev_mouse_pt = [x, y]

                    elif event == cv2.EVENT_MOUSEWHEEL:
                        # flags -= 7864320
                        print('flags: {}'.format(flags))

                        if flags > 0:
                            if flags == mouse_whl_keys_to_flags['shift'][0]:
                                offset_y -= trans_diff
                                showAugmentedImage(update=1)
                            elif flags == mouse_whl_keys_to_flags['ctrl'][0]:
                                offset_x -= trans_diff
                                showAugmentedImage(update=1)
                            elif flags == mouse_whl_keys_to_flags['ctrl+shift'][0]:
                                offset_y -= trans_diff
                                offset_x -= trans_diff
                                showAugmentedImage(update=1)
                            elif flags == mouse_whl_keys_to_flags['ctrl+alt'][0] or \
                                    flags == mouse_whl_keys_to_flags['ctrl+shift+rbutton'][0]:
                                offset_y -= trans_diff
                                offset_x += trans_diff
                                showAugmentedImage(update=1)
                            elif flags == mouse_whl_keys_to_flags['none'][0]:
                                resize_ratio += resize_ratio_diff
                                showAugmentedImage(update=1)
                        else:
                            if flags == mouse_whl_keys_to_flags['shift'][1]:
                                offset_y += trans_diff
                                showAugmentedImage(update=1)
                            elif flags == mouse_whl_keys_to_flags['ctrl'][1]:
                                offset_x += trans_diff
                                showAugmentedImage(update=1)
                            elif flags == mouse_whl_keys_to_flags['ctrl+alt'][1] or \
                                    flags == mouse_whl_keys_to_flags['ctrl+shift+rbutton'][1]:
                                offset_y += trans_diff
                                offset_x += trans_diff
                                showAugmentedImage(update=1)
                            elif flags == mouse_whl_keys_to_flags['ctrl+shift'][1]:
                                offset_y += trans_diff
                                offset_x -= trans_diff
                                showAugmentedImage(update=1)
                            elif flags == mouse_whl_keys_to_flags['none'][1]:
                                resize_ratio = max(0.1, resize_ratio - resize_ratio_diff)
                                showAugmentedImage(update=1)

                def showAugmentedImage(update=0):
                    if update:
                        updateAugmentedPatchAndMask()
                        updateAugmentedImage()
                    dst_img_disp = np.copy(dst_img)
                    if show_bbox:
                        cv2.rectangle(dst_img_disp, (dst_xmin, dst_ymin), (dst_xmax, dst_ymax), (0, 255, 0), 2)
                    # dst_img_disp = resizeAR(dst_img_disp, vis_width, vis_height)
                    cv2.imshow('augmented_image', dst_img_disp)

                if visualize:
                    # bkg_img_disp = np.copy(bkg_img)
                    # _bbox = bkg_bbox
                    # _bkg_xmin, _bkg_ymin, _bkg_xmax, _bkg_ymax = _bbox
                    # _bkg_xmin, _bkg_ymin, _bkg_xmax, _bkg_ymax = int(_bkg_xmin), int(_bkg_ymin), int(_bkg_xmax), int(
                    #     _bkg_ymax)
                    # col = (0, 255, 0)
                    # cv2.rectangle(bkg_img_disp, (_bkg_xmin, _bkg_ymin), (_bkg_xmax, _bkg_ymax), col, 2)
                    # bkg_img_disp = resizeAR(bkg_img_disp, vis_width, vis_height)
                    # cv2.imshow('bkg_img', bkg_img_disp)
                    cv2.imshow('dst_patch_mask', dst_patch_mask)
                    cv2.imshow('dst_patch', dst_patch)

                    # cv2.imshow('dst_patch_matched', dst_patch_matched)
                    # print('dst_patch_mask.shape', dst_patch_mask.shape)
                    # print('bkg_patch.shape', bkg_patch.shape)
                    # print('dst_patch.shape', dst_patch.shape)

                    showAugmentedImage()

                    if pause_after_frame:
                        cv2.setMouseCallback('augmented_image', mouseHandler)

                    _pause = 1
                    while _pause:
                        _pause = pause_after_frame

                        k = cv2.waitKeyEx(100)
                        if k < 0:
                            continue

                        print('k: {}'.format(k))
                        # src_id -= 1
                        if k == ord('r'):
                            print('resetting')
                            src_id = -1
                            prev_src_bbox = None
                            continue
                        # elif ord('9') >= k >= ord('1'):
                        #     rot_angle = 10 * (k - ord('1') + 1)
                        #     showAugmentedImage(update=1)
                        # elif k == ord('2'):
                        #     rot_angle = 20
                        # elif k == ord('3'):
                        #     rot_angle = 30
                        # elif k == ord('4'):
                        #     rot_angle = 40
                        # elif k == ord('5'):
                        #     rot_angle = 50
                        # elif k == ord('6'):
                        #     rot_angle = 60
                        # elif k == ord('7'):
                        #     rot_angle = 70
                        # elif k == ord('8'):
                        #     rot_angle = 80
                        # elif k == ord('9'):
                        #     rot_angle = 90
                        #     showAugmentedImage(update=1)
                        elif k == ord('q'):
                            sys.exit(0)
                        elif k == ord('v'):
                            visualize = 1 - visualize
                        elif k == 27 or k == 'a':
                            break
                        elif k == 32:
                            pause_after_frame = 1 - pause_after_frame
                        elif k == 2490368 or k == ord('8'):
                            # print('up')
                            offset_y -= trans_diff
                            showAugmentedImage(update=1)
                        elif k == 2621440 or k == ord('2'):
                            # print('down')
                            offset_y += trans_diff
                            showAugmentedImage(update=1)
                        elif k == 2555904 or k == ord('6'):
                            # print('right')
                            offset_x += trans_diff
                            showAugmentedImage(update=1)
                            # continue
                        elif k == 2424832 or k == ord('4'):
                            # print('left')
                            offset_x -= trans_diff
                            showAugmentedImage(update=1)
                            # src_id -= 2
                            # continue
                        elif k == ord('9'):
                            # print('up-right')
                            offset_y -= trans_diff
                            offset_x += trans_diff
                            showAugmentedImage(update=1)
                        elif k == ord('7'):
                            # print('up-left')
                            offset_y -= trans_diff
                            offset_x -= trans_diff
                            showAugmentedImage(update=1)
                        elif k == ord('3'):
                            # print('down-right')
                            offset_y += trans_diff
                            offset_x += trans_diff
                            showAugmentedImage(update=1)
                        elif k == ord('1'):
                            # print('down-left')
                            offset_y += trans_diff
                            offset_x -= trans_diff
                            showAugmentedImage(update=1)
                        elif k == ord('h') or k == ord('H'):
                            pprint(ui_help)

                else:
                    sys.stdout.write('\rDone {:d} images for augmented sequence {:d} '
                                     'using frame {:d} in source sequence {:d}'.format(
                        aug_frame_id, aug_seq_id, src_id + 1, seq_id + 1))
                    sys.stdout.flush()

                if save_seq:
                    _, mask_pts = Shape.contourPtsFromMask(dst_patch_mask.astype(np.uint8))
                    mask_pts = [[x + dst_xmin, y + dst_ymin, f] for x, y, f in mask_pts]
                    xml_writer.addBndBox(dst_xmin, dst_ymin, dst_xmax, dst_ymax,
                                         class_name, 0, 'ground_truth', target_id, 1, mask_pts)

            if save_seq:
                time_taken = time.time() - start_t
                video_out.write(dst_img, dst_id + 1, out_prefix)
                filename_no_ext = os.path.splitext(os.path.basename(video_out.filename))[0]
                xml_fname = os.path.join(aug_xml_path, '{}.xml'.format(filename_no_ext))
                print('\nDone frame {}/{}  Time taken: {:.3f} Saving xml annotations to {}'.format(
                    src_id + 1, n_src_files, time_taken, xml_fname))

                xml_writer.save(_filename=video_out.filename, targetFile=xml_fname, _imgSize=dst_img.shape)
                dst_id += 1
                start_t = time.time()

            prev_src_bbox = src_bbox

            if aug_frame_id == aug_seq_size and frame_id > 0:
                if save_seq:
                    video_out.release()
                    video_out = None
                aug_seq_id += 1
    # print('src_id: {}'.format(src_id))
    if video_out is not None:
        video_out.release()

    if visualize:
        for win_name in win_names:
            cv2.destroyWindow(win_name)
    else:
        sys.stdout.write('\n')
        sys.stdout.flush()
    if not save_seq:
        return dst_img_list


def readBackgroundData(bkg_paths, bkg_size='', img_ext='jpg', border=0, fixed_ar=0):
    bkg_files = []
    bkg_data_dict = {}
    bkg_imgs = []

    for bkg_path in bkg_paths:
        if os.path.isdir(bkg_path):
            if bkg_size:
                bkg_img_width, bkg_img_height = [int(x) for x in bkg_size.split('x')]
            #     bkg_pkl_path = os.path.join(bkg_path, 'bkg_imgs_{}.pkl'.format(bkg_size))
            # else:
            #     bkg_pkl_path = os.path.join(bkg_path, 'bkg_imgs.pkl')

            curr_bkg_files = [os.path.join(bkg_path, k) for k in os.listdir(bkg_path)
                              if k.endswith('.{:s}'.format(img_ext))]
            curr_bkg_files.sort(key=sortKey)
        elif os.path.isfile(bkg_path):
            curr_bkg_files = [bkg_path, ]
            bkg_path = os.path.dirname(bkg_path)
        else:
            raise IOError('Invalid bkg_path: {}'.format(bkg_path))

        n_bkgs = len(curr_bkg_files)
        if n_bkgs <= 0:
            raise SystemError('No background frames found')
        print('n_bkgs: {}'.format(n_bkgs))

        # if static_bkg:
        #     curr_bkg_files = [curr_bkg_files[static_bkg - 1], ]
        #     bkg_ids = [0, ]
        #     print('Using static background: {}'.format(os.path.join(bkg_path, curr_bkg_files[0 - 1])))
        # else:
        bkg_ids = list(range(n_bkgs))

        bkg_files += curr_bkg_files

        bkg_det_path = os.path.join(bkg_path, 'annotations.csv')
        df = pd.read_csv(bkg_det_path)

        # if os.path.isfile(bkg_pkl_path):
        #     print('Loading background images from {}'.format(bkg_pkl_path))
        #     with open(bkg_pkl_path, 'rb') as f:
        #         bkg_imgs = pickle.load(f)
        # else:
        print('Reading background image sequence from {}'.format(bkg_path))
        for i in bkg_ids:
            bkg_img_path = curr_bkg_files[i]
            filename = os.path.basename(bkg_img_path)

            bboxes = df.loc[df['filename'] == filename]
            if bboxes.empty:
                print('df[filename]', df['filename'])
                raise IOError('No bboxes found for: {}'.format(filename))

            bkg_img = cv2.imread(bkg_img_path)
            orig_shape = bkg_img.shape
            img_h, img_w = orig_shape[:2]

            n_bboxes = len(bboxes.index)
            df = df.drop(bboxes.index[:n_bboxes])

            bkg_data_dict[bkg_img_path] = []

            for box_id in range(n_bboxes):
                bbox = bboxes.iloc[box_id]
                class_name = bbox.loc['class']
                if class_name == 'background':
                    continue

                xmin = bbox.loc['xmin']
                ymin = bbox.loc['ymin']
                xmax = bbox.loc['xmax']
                ymax = bbox.loc['ymax']

                xmin, ymin, xmax, ymax = fix_bbox((xmin, ymin, xmax, ymax), fixed_ar, border, orig_shape)

                # if border:
                #     xmin = max(xmin - border, 0)
                #     ymin = max(ymin - border, 0)
                #     xmax = min(xmax + border, img_w)
                #     ymax = min(ymax + border, img_h)

                # box_w, box_h = xmax - xmin, ymax - ymin

                bkg_data_dict[bkg_img_path].append([xmin, ymin, xmax, ymax])

            resize_factor = 1.0
            if bkg_size:
                bkg_img, resize_factor, _, _ = resizeAR(bkg_img, bkg_img_width, bkg_img_height,
                                                        return_factors=True, add_border=False)
                # print('bkg_img.shape: ', bkg_img.shape)
            bkg_imgs.append({'name': bkg_img_path, 'image': bkg_img,
                             'resize_factor': resize_factor, 'orig_shape': orig_shape})

            sys.stdout.write('\rDone {:d} frames'.format(i + 1))
            sys.stdout.flush()
        print()
        # print('Saving background images to {}'.format(bkg_pkl_path))
        # with open(bkg_pkl_path, 'wb') as f:
        #     pickle.dump(bkg_imgs, f, pickle.HIGHEST_PROTOCOL)

    return bkg_data_dict, bkg_imgs, bkg_files


def main():
    params = {
        'labels_path': 'data/wildlife_label_map.pbtxt',
        'src_root_dir': '',
        'src_paths': '',
        'src_postfix': '',
        'src_id': 0,

        'img_dir': '',
        'mask_root_dir': '',
        'mask_paths': '',
        'mask_postfix': '',
        'mask_dir': '',
        'mask_ext': 'png',

        'bkg_root_dir': '',
        'bkg_paths': '',
        'save_path': '',
        'load_path': '',
        'aug_seq_prefix': '',
        'mask_prefix': '',
        'n_aug': 19,
        'n_classes': 7,
        'img_ext': 'jpg',
        'batch_size': 1,
        'show_img': 1,
        'start_frame_id': 0,
        'end_frame_id': 0,
        'vis_size': '1280x720',
        'border_ratio': 0.0,
        'make_square': 1,
        'mask_type': 0,
        'hist_match_type': 0,
        'random_bkgs': 0,
        'static_bkg': 1,
        'aug_seq_size': 1000,
        'visualize': 0,
        'bkg_size': '',
        'bkg_iou_thresh': 0.1,
        'only_one_src_obj': 1,
        'raw_mask': 0,
        'boxes_per_bkg': 1,
        'random_bkg_box': 0,
        'save_seq': 1,
        'fixed_ar': 0,
        'border': 0,
        'map_to_bbox': 0,
        'show_bbox': 0,
        'show_blended': 1,
        'apply_mask_contour': 0,
        'resize_ratio_diff': 0.01,
        'trans_diff': 2,
    }

    processArguments(sys.argv[1:], params)

    src_root_dir = params['src_root_dir']
    src_paths = params['src_paths']
    src_postfix = params['src_postfix']
    src_id = params['src_id']

    mask_root_dir = params['mask_root_dir']
    mask_paths = params['mask_paths']
    mask_postfix = params['mask_postfix']

    bkg_root_dir = params['bkg_root_dir']
    bkg_paths = params['bkg_paths']

    n_aug = params['n_aug']
    save_path = params['save_path']
    # aug_seq_prefix = params['aug_seq_prefix']
    img_ext = params['img_ext']
    start_frame_id = params['start_frame_id']
    end_frame_id = params['end_frame_id']
    # vis_size = params['vis_size']
    # borderq_ratio = params['border_ratio']
    # make_square = params['make_square']
    # mask_type = params['mask_type']
    # hist_match_type = params['hist_match_type']
    # random_bkgs = params['random_bkgs']
    # aug_seq_size = params['aug_seq_size']
    # visualize = params['visualize']
    bkg_size = params['bkg_size']
    # bkg_iou_thresh = params['bkg_iou_thresh']
    # only_one_src_obj = params['only_one_src_obj']
    # static_bkg = params['static_bkg']
    border = params['border']
    fixed_ar = params['fixed_ar']
    # mask_root_path = params['mask_root_path']
    # mask_seq_name = params['mask_seq_name']
    # mask_postfix = params['mask_postfix']
    img_dir = params['img_dir']
    mask_dir = params['mask_dir']
    mask_ext = params['mask_ext']
    # raw_mask = params['raw_mask']
    # boxes_per_bkg = params['boxes_per_bkg']
    # random_bkg_box = params['random_bkg_box']
    # save_seq = params['save_seq']

    if src_paths:
        if os.path.isfile(src_paths):
            print('Reading source sequence names from {}'.format(src_paths))
            src_paths = [x.strip() for x in open(src_paths).readlines() if x.strip()]
        else:
            src_paths = src_paths.split(',')
        if src_root_dir:
            src_paths = [os.path.join(src_root_dir, name) for name in src_paths]
    else:
        src_paths = [os.path.join(src_root_dir, name) for name in os.listdir(src_root_dir) if
                     os.path.isdir(os.path.join(src_root_dir, name))]
    if src_postfix:
        src_paths = ['{}_{}'.format(name, src_postfix) for name in src_paths]

    if img_dir:
        src_paths = [os.path.join(name, img_dir) for name in src_paths]

    src_paths.sort(key=sortKey)

    if mask_paths:
        if os.path.isfile(mask_paths):
            mask_paths = [x.strip() for x in open(mask_paths).readlines() if x.strip()]
        else:
            mask_paths = mask_paths.split(',')
        if mask_root_dir:
            mask_paths = [os.path.join(mask_root_dir, name) for name in mask_paths]
    elif mask_root_dir:
        mask_paths = [os.path.join(mask_root_dir, name) for name in os.listdir(mask_root_dir) if
                      os.path.isdir(os.path.join(mask_root_dir, name))]
    else:
        mask_paths = [os.path.dirname(src_path) if img_dir else src_path for src_path in src_paths]
        if mask_postfix:
            mask_paths = ['{}_{}'.format(name, mask_postfix) for name in mask_paths]

    mask_paths.sort(key=sortKey)
    if src_id > 0:
        src_paths = src_paths[src_id:]
        mask_paths = mask_paths[src_id:]

    n_seq = len(src_paths)
    n_mask_seq = len(mask_paths)

    if n_seq != n_mask_seq:
        raise IOError('Mismatch between no. of source sequences: {} and mask sequences: {}'.format(
            n_seq, n_mask_seq))

    if mask_dir:
        mask_paths = [os.path.join(name, mask_dir) for name in mask_paths]

    if bkg_paths:
        if os.path.isfile(bkg_paths):
            print('Reading background sequence names from {}'.format(bkg_paths))
            bkg_paths = [x.strip() for x in open(bkg_paths).readlines() if x.strip()]
        else:
            bkg_paths = bkg_paths.split(',')
        if bkg_root_dir:
            bkg_paths = [os.path.join(bkg_root_dir, name) for name in bkg_paths]
    else:
        bkg_paths = [os.path.join(bkg_root_dir, name) for name in os.listdir(bkg_root_dir) if
                     os.path.isdir(os.path.join(bkg_root_dir, name))]
    bkg_paths.sort(key=sortKey)

    bkg_data_dict, bkg_imgs, bkg_files = readBackgroundData(bkg_paths, bkg_size, img_ext)

    print('bkg_files:')
    bkg_files_prefix = os.path.commonprefix(bkg_files)
    bkg_files_rel = {i + 1: k.replace(bkg_files_prefix, '') for i, k in enumerate(bkg_files)}
    pprint(bkg_files_rel)
    # sys.exit()

    if not save_path:
        save_path = os.path.dirname(src_paths[0])

    src_data_dict = {}
    src_files = []

    for seq_id in range(n_seq):
        src_path = src_paths[seq_id]
        seq_name = os.path.splitext(os.path.basename(src_path))[0]
        seq_dir = os.path.dirname(src_path)

        det_path = os.path.join(src_path, 'annotations.csv')
        print('\nsequence {}/{}: {}'.format(seq_id + 1, n_seq, seq_name))

        print('Reading source images from: {}'.format(src_path))

        curr_src_files = [os.path.join(src_path, k) for k in os.listdir(src_path) if
                          k.endswith('.{:s}'.format(img_ext))]
        n_src_files = len(curr_src_files)
        if n_src_files <= 0:
            raise IOError('No source frames found for {}'.format(src_path))
        curr_src_files.sort(key=sortKey)

        # print('start_frame_id: {}'.format(start_frame_id))
        # print('end_frame_id: {}'.format(end_frame_id))

        if end_frame_id <= 0:
            _end_frame_id = n_src_files
        else:
            _end_frame_id = end_frame_id

        if start_frame_id > 0 or _end_frame_id < n_src_files:
            print('Including only frames from {} to {} out of {} files'.format(
                start_frame_id, _end_frame_id, n_src_files))

            curr_src_files = curr_src_files[start_frame_id:_end_frame_id]
            n_src_files = len(curr_src_files)

        if n_src_files == 0:
            raise IOError('No src_files found')

        print('n_src_files: {}'.format(n_src_files))

        df = pd.read_csv(det_path)

        mask_files = []
        mask_path = mask_paths[seq_id]

        # if mask_paths:
        #     # if not mask_root_path:
        #     #     mask_root_path = seq_dir
        #     # if not mask_seq_name:
        #     #     mask_seq_name = '{}_{}'.format(seq_name, mask_postfix)
        #     # mask_path = os.path.join(mask_root_path, mask_seq_name)
        #     mask_path = mask_paths[seq_id]
        # else:
        #     print('seq_dir: {}'.format(seq_dir))
        #     print('mask_dir: {}'.format(mask_dir))
        #
        #     mask_path = os.path.join(seq_dir, mask_dir)

        print('Loading masks from {}'.format(mask_path))
        mask_files = [os.path.join(mask_path, k) for k in os.listdir(mask_path) if
                      k.endswith('.{:s}'.format(mask_ext))]
        mask_files.sort(key=sortKey)

        # n_masks = len(mask_files)
        # if n_masks != n_src_files:
        #     raise SystemError('Mismatch between no. of masks: {} and frames: {}'.format(n_masks, n_src_files))

        valid_src_files = 0

        for file_id in range(n_src_files):
            src_file = curr_src_files[file_id]
            src_file = src_file.replace('\\', '/')

            filename = os.path.basename(src_file)
            filename_no_ext = os.path.splitext(filename)[0]

            curr_mask_files = [k for k in mask_files if os.path.basename(k).startswith(filename_no_ext)]

            if not curr_mask_files:
                # print('No matching mask file found for {}'.format(src_file))
                continue
            n_masks = len(curr_mask_files)

            bboxes = df.loc[df['filename'] == filename]
            n_bboxes = len(bboxes.index)

            if n_masks != n_bboxes:
                raise IOError('Mismatch between n_bboxes: {} and n_masks: {} for {}'.format(
                    n_bboxes, n_masks, src_file))

            df = df.drop(bboxes.index[:n_bboxes])
            for box_id in range(n_bboxes):

                mask_file = curr_mask_files[box_id].replace('\\', '/')

                # mask_h, mask_w = curr_mask.shape[:2]

                bbox = bboxes.iloc[box_id]
                xmin = bbox.loc['xmin']
                ymin = bbox.loc['ymin']
                xmax = bbox.loc['xmax']
                ymax = bbox.loc['ymax']
                img_w = bbox.loc['width']
                img_h = bbox.loc['height']
                target_id = bbox.loc['target_id']
                class_name = bbox.loc['class']

                xmin, ymin, xmax, ymax = fix_bbox((xmin, ymin, xmax, ymax), fixed_ar, border, (img_h, img_w, 3))

                box_w, box_h = xmax - xmin, ymax - ymin
                bbox = [xmin, ymin, xmax, ymax]

                # if not mask_files:
                #     mask_str = ''
                #     try:
                #         mask_str = row['mask']
                #     except KeyError:
                #         pass
                #     if mask_str is None or not mask_str:
                #         raise IOError('No mask found for {}'.format(filename))
                #     mask = [pt.split(',') for pt in mask_str.split(';') if pt]
                #     # print('mask_str: ', mask_str)
                #     # print('mask: ', mask)
                #     mask = [(float(x), float(y)) for x, y in mask]
                #     mask_loaded = 0
                # else:
                #     mask_id = curr_src_files.index(filename)
                #     mask = cv2.imread(mask_files[mask_id])
                #     mask[mask > 0] = 1
                #     mask_loaded = 1

                if filename not in src_data_dict:
                    src_data_dict[src_file] = {
                        'img': src_file,
                        'data': [],
                    }
                src_data_dict[src_file]['data'].append(
                    {"class_name": class_name, "bbox": bbox, 'mask': mask_file, 'target_id': target_id,
                     'seq_id': seq_id})

            src_files.append(src_file)
            valid_src_files += 1
            sys.stdout.write('\rDone {:d}/{:d} files with {:d} boxes'.format(
                file_id + 1, n_src_files, n_bboxes))
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        if valid_src_files == 0:
            raise IOError('No valid_src_files found')
        print('valid_src_files: {}'.format(valid_src_files))

    params['save_path'] = save_path

    args, varargs, varkw, defaults = inspect.getargspec(generateAugmentedSequence)
    params = {k: params[k] for k in params if k in args}
    pprint(params)

    generateAugmentedSequence(src_files, src_data_dict, bkg_files, bkg_data_dict, bkg_imgs,
                              **params,
                              # static_bkg, boxes_per_bkg, random_bkg_box,
                              # only_one_src_obj, aug_seq_size,
                              # aug_seq_prefix, save_path, hist_match_type,
                              # save_seq, visualize, vis_size, raw_mask
                              )


if __name__ == '__main__':
    main()
