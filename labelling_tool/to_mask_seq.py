import os, sys, glob, re
import numpy as np
import pandas as pd
import cv2
import shutil
from PIL import Image
from tqdm import tqdm

from libs.pascal_voc_io import PascalVocReader, PascalVocWriter
from libs.frames_readers import get_frames_reader
from libs.shape import Shape

sys.path.append('..')
from tf_api.utilities import sortKey, isEmpty, map_mask_to_bbox, resizeAR, clamp
from tracking.Utilities import col_bgr

_pause = 1
_exit = 0


class Params:
    def __init__(self):
        self.cfg = ()
        self.allow_skipping_images = 0
        self.batch_size = 1
        self.class_names_path = '../labelling_tool/data//predefined_classes_orig.txt'
        self.read_colors = 1
        self.codec = 'H264'
        self.combine_seq = 0
        self.csv_file_name = ''
        self.enable_out_suffix = 1
        self.enable_xml_annotations = 1
        self.fixed_ar = 0.0
        self.fps = 20
        self.img_ext = 'png'
        self.load_path = ''
        self.map_folder = ''
        self.map_to_bbox = 0
        self.masks_per_seq = 0
        self.n_classes = 4
        self.n_frames = 0
        self.out_border = 0
        # self.out_img_dir = 'images'
        self.out_mask_size = '0x0'
        self.out_root_path = ''
        self.root_dir = ''
        self.save_combined = 2
        self.single_id = -1
        self.save_file_name = ''
        self.save_raw_mask = 0
        self.save_test = 0
        self.save_train = 1
        self.save_video = 0
        self.seq_paths = ''
        self.show_img = 0
        self.crop = 0
        self.blended_vis = 0
        self.save_patches = 0
        self.extract_features = 0
        self.n_proc = 1


def save_masks(seq_path, xml_path, out_mask_size, out_border, fixed_ar,
               save_raw_mask, show_img, classes, out_root_path='', save_test=0,
               save_train=1, frames_reader=None, masks_per_seq=0, enable_out_suffix=1,
               train_fnames=None, test_fnames=None, map_to_bbox=0,
               enable_xml_annotations=0, allow_skipping_images=0, save_combined=0,
               class_to_color=None, blended_vis=0, save_patches=0):
    global _pause, _exit

    if xml_path is None:
        xml_path = os.path.join(seq_path, 'annotations')

    if not xml_path or not os.path.isdir(xml_path):
        raise IOError('Folder containing the loaded boxes does not exist: {}'.format(xml_path))

    files = glob.glob(os.path.join(xml_path, '*.xml'))
    n_files = len(files)
    if n_files == 0:
        raise IOError('No loaded boxes found')

    seq_root_dir = os.path.dirname(seq_path)
    seq_name = os.path.basename(seq_path)
    print('seq_name: {}'.format(seq_name))

    if frames_reader is None:
        frames_reader = get_frames_reader(seq_path, save_as_bin=False)

    min_dim = max_dim = 0
    out_w, out_h = out_mask_size
    # print('out_mask_size: {}'.format(out_mask_size))

    if out_w == -1 and out_h == -1:
        out_w = out_h = 0

    if out_w == -1:
        max_dim = out_h
    elif out_h == -1:
        min_dim = out_w

    if fixed_ar:
        print('Using fixed aspect ratio: {}'.format(fixed_ar))

    # print('out_border: {}'.format(out_border))

    def getint(fn):
        basename = os.path.basename(fn)
        num = re.sub("\D", "", basename)
        try:
            return int(num)
        except:
            return 0

    if len(files) > 0:
        files = sorted(files, key=getint)

    print('Loading annotations from {:d} files'.format(n_files))
    file_id = 0
    n_boxes = 0

    out_db_name = 'masks'
    cmb_train_root_path = os.path.join(seq_root_dir, out_db_name)

    if enable_out_suffix:
        if map_to_bbox:
            out_db_name = '{}_mapped'.format(out_db_name)
        else:
            out_db_name = '{}_{}x{}'.format(out_db_name, out_w, out_h)
            if fixed_ar:
                out_db_name = '{}_ar_{}'.format(out_db_name, fixed_ar)
            else:
                out_db_name = '{}_{}'.format(out_db_name, out_border)

        out_db_name = out_db_name.replace('.', 'p')

    train_root_path = os.path.join(seq_root_dir, out_db_name)
    test_root_path = os.path.join(seq_root_dir, out_db_name, 'test')

    assert save_test or save_train, 'Either save_test or save_train must be enabled'

    # print('Saving output sequences to  {}'.format(out_root_path))

    if save_train:
        out_img_root_path = os.path.join(train_root_path, 'images', seq_name)
        out_mask_root_path = os.path.join(train_root_path, 'labels', seq_name)
        out_vis_root_path = os.path.join(train_root_path, 'vis', seq_name)
        out_xml_path = os.path.join(out_img_root_path, 'annotations')

        if save_combined != 2:
            # out_overlap_vis_root_path = os.path.join(train_root_path, 'overlap_vis', seq_name)

            print('Saving training mask sequence to {}'.format(train_root_path))
            os.makedirs(out_img_root_path, exist_ok=1)
            os.makedirs(out_mask_root_path, exist_ok=1)
            os.makedirs(out_vis_root_path, exist_ok=1)
            # os.makedirs(out_overlap_vis_root_path, exist_ok=1)

            if enable_xml_annotations:
                out_xml_path = os.path.join(out_img_root_path, 'annotations')
                print('Saving xml_annotations to {}'.format(out_xml_path))
                if not os.path.isdir(out_xml_path):
                    os.makedirs(out_xml_path)

        if save_combined:
            full_out_img_root_path = os.path.join(cmb_train_root_path, 'images', seq_name)
            full_out_mask_root_path = os.path.join(cmb_train_root_path, 'labels', seq_name)
            full_out_vis_root_path = os.path.join(cmb_train_root_path, 'vis', seq_name)
            # full_out_overlap_vis_root_path = os.path.join(cmb_train_root_path, 'overlap_vis', seq_name)

            print('Saving full image training mask sequence to {}'.format(full_out_mask_root_path))

            os.makedirs(full_out_img_root_path, exist_ok=1)
            os.makedirs(full_out_mask_root_path, exist_ok=1)
            os.makedirs(full_out_vis_root_path, exist_ok=1)
            # os.makedirs(full_out_overlap_vis_root_path, exist_ok=1)

    if save_combined != 2 and save_test:
        out_test_seq_name = seq_name
        test_img_root_path = os.path.join(test_root_path, out_test_seq_name)

        print('Saving unlabeled testing mask sequence to {}'.format(test_img_root_path))
        os.makedirs(test_img_root_path, exist_ok=1)

    win_name = 'patch and mask'

    disable_resizing = 0
    scale_x = scale_y = 1.0
    if out_w == 0 and out_h == 0:
        print('Resizing disabled')
        disable_resizing = 1

    csv_raw = []
    test_csv_raw = []

    n_files = len(files)

    if save_raw_mask:
        print('Saving raw labels')
        if class_to_color is None:
            class_to_color = {
                _class: (_id + 1, _id + 1, _id + 1) for _id, _class in enumerate(classes)
            }
    else:
        if class_to_color is None:
            n_classes = len(classes)
            col_diff = 255.0 / n_classes
            class_id_to_col_gs = {
                _id: int(col_diff * (_id + 1)) for _id in range(n_classes)
            }
            class_to_color = {
                classes[_id]: (col, col, col) for _id, col in class_id_to_col_gs.items()
            }
            print('using default class colors:\n{}'.format(class_to_color))

    n_masks = 0

    _train_fnames = []
    _test_fnames = []

    _exit_seq = 0

    disp_img = None

    class_label_to_count = {}
    class_ids = []

    out_img_paths = []
    out_mask_paths = []

    cmb_out_img_paths = []
    cmb_out_mask_paths = []

    for file_id, file in enumerate(tqdm(files)):
        xml_reader = PascalVocReader(file)
        filename = os.path.basename(xml_reader.filename)
        filename_no_ext = os.path.splitext(filename)[0]
        # file_id = int(re.sub("\D", "", filename))

        # print('filename: {}'.format(filename))
        # print('file_id: {}'.format(file_id))

        img = frames_reader.get_frame_by_name(filename, convert_to_rgb=0)
        if img is None:
            msg = 'image {} could not be read'.format(filename)
            # if allow_skipping_images:
            #     print('\n' + msg + '\n')
            #     continue
            # else:
            #     raise AssertionError(msg)
            print('\n' + msg + '\n')
            continue

        img_h, img_w = img.shape[:2]

        mask_img = None

        shapes = xml_reader.getShapes()
        n_shapes = len(shapes)

        # if n_shapes > 1:
        #     print('{} boxes found for {} in {}'.format(n_shapes, filename, file))

        obj_id = 0

        img_written = 0

        full_mask = np.zeros_like(img)

        img_vis = img.copy()

        for shape_id, shape in enumerate(shapes):
            label, points, _, _, difficult, bbox_source, id_number, score, mask, mask_img = shape
            class_col = class_to_color[label]

            if not mask:
                if not save_test:
                    continue

                xmin, ymin = points[0]
                xmax, ymax = points[2]
                img_root_path = test_img_root_path
            else:

                if not save_train:
                    continue

                mask_pts_list = Shape.getContourPts(mask, verbose=0)

                mask_pts = np.asarray(mask_pts_list)
                xmin, ymin = np.min(mask_pts, axis=0).astype(np.int32)
                xmax, ymax = np.max(mask_pts, axis=0).astype(np.int32)

                img_root_path = out_img_root_path

            xmin, xmax = clamp((xmin, xmax), 0, img_w - 1)
            ymin, ymax = clamp((ymin, ymax), 0, img_h - 1)

            w, h = xmax - xmin, ymax - ymin

            if w <= 0 or h <= 0:
                print('\nskipping object {} with invalid box: {}\n'.format(shape_id + 1, [xmin, ymin, xmax, ymax]))
                continue

            if fixed_ar:

                src_ar = float(w) / float(h)
                if fixed_ar > src_ar:
                    border_x = int((h * fixed_ar - w) / 2.0)
                    border_y = 0
                else:
                    border_y = int((w / fixed_ar - h) / 2.0)
                    border_x = 0
            else:
                border_x = border_y = out_border

            # start_row, start_col = max(0, ymin - border_y), max(0, xmin - border_x)
            # end_row, end_col = min(img_h - 1, ymax + border_y), min(img_w - 1, xmax + border_x)

            start_row, start_col = ymin - border_y, xmin - border_x
            end_row, end_col = ymax + border_y, xmax + border_x

            if start_row < 0 or start_col < 0 or end_row >= img_h or end_col >= img_w:
                msg = 'Invalid border {} for box {} in image {} of size {}'.format(
                    [border_x, border_y], [xmin, ymin, xmax, ymax], filename, [img_w, img_h]
                )
                if allow_skipping_images:
                    print('\n' + msg + '\n')
                    continue
                else:
                    raise AssertionError(msg)

            if mask:
                n_masks += 1

            w, h = end_col - start_col, end_row - start_row
            patch_img = img[start_row:end_row, start_col:end_col, :]

            if not disable_resizing:
                if max_dim > 0:
                    if w > h:
                        out_w = max_dim
                        out_h = 0
                    else:
                        out_h = max_dim
                        out_w = 0
                elif min_dim > 0:
                    if w < h:
                        out_w = min_dim
                        out_h = 0
                    else:
                        out_h = min_dim
                        out_w = 0
                else:
                    out_w, out_h = out_mask_size

                scale_x = float(out_w) / float(w)
                scale_y = float(out_h) / float(h)
                if scale_x == 0:
                    scale_x = scale_y
                    out_w = int(w * scale_x)
                elif scale_y == 0:
                    scale_y = scale_x
                    out_h = int(h * scale_y)
                try:
                    patch_img = cv2.resize(patch_img, (out_w, out_h))
                    # print('patch_img: {}'.format(patch_img.shape))
                except cv2.error as e:
                    print('patch_img: {}'.format(patch_img.shape))
                    print('out_size: {}, {}'.format(start_row, start_col))
                    print('out_size: {}, {}'.format(end_row, end_col))
                    print('out_size: {}, {}'.format(out_w, out_h))
                    raise cv2.error(e)
            else:
                out_w, out_h = w, h

            _label = label
            if id_number is None:
                id_number = -1
            if id_number > 0:
                _label = '{}_{}'.format(_label, id_number)

            if enable_out_suffix:
                out_fname = '{}_{}_{}'.format(filename_no_ext, obj_id, label)
            else:
                out_fname = filename_no_ext

            _xmin, _ymin = int((xmin - start_col) * scale_x), int((ymin - start_row) * scale_y)
            _xmax, _ymax = int((xmax - start_col) * scale_x), int((ymax - start_row) * scale_y)

            if blended_vis:
                cv2.rectangle(img_vis, (xmin, ymin), (xmax, ymax), class_col, 2)

            if map_to_bbox:
                if not img_written:
                    img_written = 1

                    if save_combined != 2:
                        out_img_path = os.path.join(img_root_path, filename)
                        cv2.imwrite(out_img_path, img)

                        out_img_paths.append((out_img_path, label))

                    if label not in class_label_to_count:
                        class_label_to_count[label] = 0

                    class_label_to_count[label] += 1

                    label_id = classes.index(label)
                    class_ids.append(label_id)

                    if enable_xml_annotations:
                        imageShape = [xml_reader.height, xml_reader.width, 3]
                        xml_writer = PascalVocWriter(out_xml_path, filename, imageShape)

                if mask:
                    if enable_xml_annotations:
                        bndbox = [xmin, ymin, xmax, ymax]
                        xml_writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3],
                                             label, difficult, bbox_source, id_number, score, mask, mask_img)

                raw_data = {
                    'target_id': int(id_number),
                    'filename': filename,
                    'width': img_w,
                    'height': img_h,
                    'class': label,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                }

                if show_img:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), class_col, 2)
                    disp_img = img
            else:
                img_out_fname = out_fname + '.jpg'
                if mask:
                    if save_combined != 2:
                        out_img_path = os.path.join(img_root_path, img_out_fname)
                        cv2.imwrite(out_img_path, patch_img)
                        out_img_paths.append((out_img_path, label))

                        if enable_xml_annotations:
                            n_mask = len(mask)

                            _mask = []
                            for i in range(n_mask):
                                _mask.append([
                                    (mask[i][0] - start_col) * scale_x,
                                    (mask[i][1] - start_row) * scale_y,
                                    mask[i][2]
                                ])

                            imageShape = [xml_reader.height, xml_reader.width, 3]
                            xml_writer = PascalVocWriter(out_xml_path, xml_reader.filename, imageShape)
                            xml_writer.addBndBox(_xmin, _ymin, _xmax, _ymax,
                                                 label, difficult, bbox_source, id_number, score, _mask)

                    if label not in class_label_to_count:
                        class_label_to_count[label] = 0

                    class_label_to_count[label] += 1
                    label_id = classes.index(label)
                    class_ids.append(label_id)

                raw_data = {
                    'target_id': int(id_number),
                    'filename': img_out_fname,
                    'width': out_w,
                    'height': out_h,
                    'class': label,
                    'xmin': _xmin,
                    'ymin': _ymin,
                    'xmax': _xmax,
                    'ymax': _ymax
                }

                if show_img:
                    cv2.rectangle(patch_img, (_xmin, _ymin), (_xmax, _ymax), class_col, 2)
                    disp_img = patch_img

            if mask:
                if mask_img is None:
                    mask_img = np.zeros_like(img)
                # print('border_x: {}'.format(border_x))
                # print('border_y: {}'.format(border_y))
                # print('scale_x: {}'.format(scale_x))
                # print('scale_y: {}'.format(scale_y))
                #
                # print('xmin: {}'.format(xmin))
                # print('ymin: {}'.format(ymin))

                mask_pts = [[(x - xmin + border_x) * scale_x, (y - ymin + border_y) * scale_y] for x, y in
                            mask_pts]
                curr_mask = np.zeros_like(patch_img, dtype=np.uint8)
                # print('mask_img: {}'.format(mask_img.shape))
                mask_out_fname = out_fname + '.png'

                # np.savetxt('mask_seq_mask_pts.txt', mask_pts, fmt='%.6f')

                curr_mask = cv2.fillPoly(curr_mask, np.array([mask_pts, ], dtype=np.int32), class_col)

                # print('min: {} max: {}'.format(
                #     np.min(mask_img.flatten()),
                #     np.max(mask_img.flatten()))
                # )

                mapped_mask_img = map_mask_to_bbox((xmin, ymin, xmax, ymax), curr_mask, fixed_ar, out_border,
                                                   mask_img.shape, mask_img, full_mask)

                if map_to_bbox:
                    mask_img = mapped_mask_img
                else:
                    mask_img = curr_mask

                    if save_patches and save_combined != 2:
                        out_mask_path = os.path.join(out_mask_root_path, mask_out_fname)
                        out_vis_path = os.path.join(out_vis_root_path, mask_out_fname)

                        cv2.imwrite(out_mask_path, mask_img)

                        if blended_vis:
                            mask_img_gs = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                            vis_img = patch_img.copy().astype(np.float32)
                            _mask_img = mask_img.astype(np.float32)
                            mask_binary = mask_img_gs > 0
                            vis_img[mask_binary] = (0.5 * patch_img[mask_binary] + 0.5 * _mask_img[mask_binary]).astype(
                                np.uint8)
                        else:
                            vis_img = np.concatenate((patch_img, mask_img), axis=1)
                        cv2.imwrite(out_vis_path, vis_img)

                        out_mask_paths.append((out_mask_path, label))

                        _train_fnames.append((out_img_path, out_mask_path))

                    if show_img:
                        disp_mask_img = mask_img.copy()
                        if save_raw_mask:
                            disp_mask_img[disp_mask_img > 0] = 255
                        blended_img = np.asarray(
                            Image.blend(Image.fromarray(patch_img), Image.fromarray(disp_mask_img), 0.5))
                        disp_img = np.concatenate((disp_img, disp_mask_img, blended_img), axis=1)
                csv_raw.append(raw_data)
            else:
                test_csv_raw.append(raw_data)
                if not map_to_bbox:
                    _test_fnames.append(out_img_path)

            if show_img:
                full_mask_disp = np.concatenate([img, full_mask], axis=1)
                full_mask_disp = resizeAR(full_mask_disp, width=1280)
                cv2.imshow('full_mask', full_mask_disp)

                if not map_to_bbox:
                    cv2.imshow(win_name, disp_img)

                k = cv2.waitKey(1 - _pause)
                if k == ord('q'):
                    _exit = 1
                    break
                elif k == 27:
                    _exit_seq = 1
                    break
                elif k == 32:
                    _pause = 1 - _pause
            obj_id += 1

        if map_to_bbox and img is not None:
            out_img_path = os.path.join(out_img_root_path, filename)
            if save_train and mask_img is not None:
                if save_combined != 2:
                    mask_out_fname = filename_no_ext + '.png'
                    out_mask_path = os.path.join(out_mask_root_path, mask_out_fname)
                    out_mask_paths.append((out_mask_path, label))
                    out_vis_path = os.path.join(out_vis_root_path, img_out_fname)

                    cv2.imwrite(out_mask_path, mask_img)
                    if blended_vis:
                        mask_img_gs = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                        vis_img = img.copy()
                        mask_binary = mask_img_gs > 0
                        vis_img[mask_binary] = (0.5 * img[mask_binary] + 0.5 * mask_img[mask_binary]).astype(
                            np.uint8)
                    else:
                        vis_img = np.concatenate((img, mask_img), axis=1)

                    cv2.imwrite(out_vis_path, vis_img)

                    if enable_xml_annotations:
                        out_xml_file = os.path.join(out_xml_path, os.path.basename(file))
                        xml_writer.save(targetFile=out_xml_file)

                    _train_fnames.append((out_img_path, out_mask_path))

                if show_img:
                    disp_mask_img = mask_img
                    if save_raw_mask:
                        disp_mask_img[disp_mask_img > 0] = 255
                    blended_img = np.asarray(
                        Image.blend(Image.fromarray(img), Image.fromarray(disp_mask_img), 0.5))
                    disp_img = np.concatenate((disp_img, disp_mask_img, blended_img), axis=1)

            elif save_test:
                out_img_path = os.path.join(test_img_root_path, filename)
                if out_img_path in _test_fnames:
                    raise IOError('Duplicate out_img_path: {}'.format(out_img_path))
                _test_fnames.append(out_img_path)

            if show_img and disp_img is not None:
                cv2.imshow(win_name, disp_img)
                k = cv2.waitKey(1 - _pause)
                if k == ord('q'):
                    _exit = 1
                    break
                elif k == 27:
                    break
                elif k == 32:
                    _pause = 1 - _pause

        if save_combined:
            img_out_fname = filename_no_ext + '.jpg'
            cmb_out_img_path = os.path.join(full_out_img_root_path, img_out_fname)
            cv2.imwrite(cmb_out_img_path, img)

            mask_out_fname = filename_no_ext + '.png'
            cmb_out_mask_path = os.path.join(full_out_mask_root_path, mask_out_fname)
            cv2.imwrite(cmb_out_mask_path, full_mask)

            cmb_out_img_paths.append(cmb_out_img_path)
            cmb_out_mask_paths.append(cmb_out_mask_path)

            if blended_vis:

                # full_vis_img = img.copy()
                # _full_mask = full_mask.astype(np.float32)

                full_mask_gs = cv2.cvtColor(full_mask, cv2.COLOR_BGR2GRAY)
                mask_binary = full_mask_gs == 0
                # mask_binary = mask_binary[..., 0]
                # full_vis_img[mask_binary] = (0.5 * full_vis_img[mask_binary] + 0.5 * _full_mask[mask_binary])
                # full_vis_img = full_vis_img.astype(np.uint8)

                full_vis_img = (0.5 * img_vis + 0.5 * full_mask).astype(np.uint8)
                # cv2.imshow('temp1', full_vis_img)

                full_vis_img[mask_binary] = img_vis[mask_binary]

                # full_vis_img[mask_binary] = temp[mask_binary]

                # cv2.imshow('full_vis_img', full_vis_img)
                # cv2.imshow('img_vis', img_vis)
                # cv2.imshow('full_mask_gs', full_mask_gs)
                # cv2.imshow('full_mask', full_mask)
                #
                # cv2.waitKey(0)

                # _full_vis_img = np.concatenate((img, full_vis_img), axis=1)
            else:
                full_vis_img = np.concatenate((img, full_mask), axis=1)
                # _full_vis_img = full_vis_img

            out_mask_path = os.path.join(full_out_vis_root_path, img_out_fname)
            cv2.imwrite(out_mask_path, full_vis_img)

            # if show_img:
            #     _full_vis_img = resizeAR(_full_vis_img, width=1280)
            #     cv2.imshow('full_vis_img concat', _full_vis_img)

        if _exit:
            break

        # sys.stdout.write('\rDone {:d}/{:d} files {:s} ({:d} masks found)'.format(
        #     file_id + 1, n_files, filename, n_masks))
        # sys.stdout.flush()

        if n_masks >= masks_per_seq > 0:
            break

    # sys.stdout.write('\n')
    # sys.stdout.flush()

    if not _exit_seq and save_train and n_masks == 0:
        raise IOError('\nNo masks found for {}\n'.format(seq_path))

    train_csv_path = test_csv_path = ''
    if save_combined != 2:
        if csv_raw:
            print('Saved {} labeled files in training sequence'.format(len(csv_raw)))
            train_csv_path = os.path.join(out_img_root_path, 'annotations.csv')
            pd.DataFrame(csv_raw).to_csv(train_csv_path)

        if test_csv_raw:
            print('Saved {} unlabeled files in test sequence'.format(len(test_csv_raw)))
            test_csv_path = os.path.join(test_img_root_path, 'annotations.csv')
            pd.DataFrame(test_csv_raw).to_csv(test_csv_path)

    if show_img:
        cv2.destroyWindow(win_name)

    if save_train and train_fnames is not None:
        train_fnames[seq_name] = _train_fnames, train_root_path, csv_raw, train_csv_path

    if save_test and test_fnames is not None:
        test_fnames[out_test_seq_name] = _test_fnames, test_img_root_path, test_csv_raw, test_csv_path

    out_paths = {
        'img': out_img_paths,
        'mask': out_mask_paths,
        'cmb_img': cmb_out_img_paths,
        'cmb_mask': cmb_out_mask_paths,
    }
    return n_masks, out_paths, class_label_to_count, class_ids


def main():
    params = Params()

    import paramparse
    paramparse.process(params)

    seq_paths = params.seq_paths
    root_dir = params.root_dir
    out_border = params.out_border
    fixed_ar = params.fixed_ar
    out_mask_size = params.out_mask_size
    show_img = params.show_img
    save_raw_mask = params.save_raw_mask
    save_test = params.save_test
    masks_per_seq = params.masks_per_seq
    out_root_path = params.out_root_path
    enable_out_suffix = params.enable_out_suffix
    enable_xml_annotations = params.enable_xml_annotations
    blended_vis = params.blended_vis
    save_patches = params.save_patches
    allow_skipping_images = params.allow_skipping_images
    save_train = params.save_train
    combine_seq = params.combine_seq
    map_to_bbox = params.map_to_bbox
    save_combined = params.save_combined
    single_id = params.single_id
    crop = params.crop

    if combine_seq and not out_root_path:
        raise IOError('out_root_path must be provided to combine sequences')

    features_save_path = None

    if seq_paths:
        if os.path.isfile(seq_paths):
            features_save_path = os.path.splitext(os.path.basename(seq_paths))[0] + '.npz'
            seq_paths = [x.strip() for x in open(seq_paths).readlines() if x.strip()]
        else:
            seq_paths = seq_paths.split(',')
            features_save_path = os.path.splitext(os.path.basename(seq_paths[0]))[0] + '.npz'
        if root_dir:
            seq_paths = [os.path.join(root_dir, name) for name in seq_paths]
            features_save_path = os.path.join(root_dir, features_save_path)

    elif root_dir:
        seq_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                     os.path.isdir(os.path.join(root_dir, name))]
        seq_paths.sort(key=sortKey)
    else:
        raise IOError('Either seq_paths or root_dir must be provided')

    out_mask_size = [int(x) for x in out_mask_size.split('x')]
    train_fnames = {}
    test_fnames = {}
    total_n_masks = 0

    all_out_paths = {}
    all_class_label_to_count = {}
    class_ids = []

    n_seq = len(seq_paths)

    img_exts = ('.jpg',)
    classes = [k.strip() for k in open(params.class_names_path, 'r').readlines() if k.strip()]
    if params.read_colors:
        classes, class_cols = zip(*[k.split('\t') for k in classes])
        class_to_color = {
            _class: col_bgr[class_cols[_class_id]]
            for _class_id, _class in enumerate(classes)
        }
    else:
        class_cols = class_to_color = None

    if single_id >= 0:
        seq_paths = [seq_paths[single_id], ]

    if params.n_proc > 1:
        import multiprocessing
        import functools

        print('running in parallel over {} processes'.format(params.n_proc))
        pool = multiprocessing.Pool(params.n_proc)
        func = functools.partial(save_masks,
                                 xml_path=None,
                                 out_mask_size=out_mask_size,
                                 out_border=out_border,
                                 fixed_ar=fixed_ar,
                                 save_raw_mask=save_raw_mask,
                                 show_img=show_img,
                                 classes=classes,
                                 save_test=save_test,
                                 masks_per_seq=masks_per_seq,
                                 out_root_path=out_root_path,
                                 save_train=save_train,
                                 train_fnames=train_fnames,
                                 test_fnames=test_fnames,
                                 map_to_bbox=map_to_bbox,
                                 enable_out_suffix=enable_out_suffix,
                                 allow_skipping_images=allow_skipping_images,
                                 enable_xml_annotations=enable_xml_annotations,
                                 save_combined=save_combined,
                                 class_to_color=class_to_color,
                                 blended_vis=blended_vis,
                                 save_patches=save_patches,)

        pool.map(func, seq_paths)
        return

    for seq_id, seq_path in enumerate(seq_paths):
        print('seq {} / {}: {}'.format(seq_id + 1, n_seq, seq_path))
        if params.extract_features == 2:
            """extract features directly from the source images which would be previously created patch images"""

            _src_files = [os.path.join(seq_path, k) for k in os.listdir(seq_path) if
                          os.path.splitext(k.lower())[1] in img_exts]
            out_paths = {
                'img': []
            }
            n_masks = 0
            class_label_to_count = {k: 0 for k in classes}

            for _src_file in _src_files:
                _src_fname = os.path.splitext(os.path.basename(_src_file))[0]

                _src_class = [(i, k) for i, k in enumerate(classes) if _src_fname.endswith('_' + k)]
                assert len(_src_class) == 1, "invalid _src_fname: {}".format(_src_fname)

                _src_class_id, _src_class = _src_class[0]
                class_ids.append(_src_class_id)

                class_label_to_count[_src_class] += 1

                out_paths['img'].append((_src_file, _src_class))

                n_masks += 1

        else:
            voc_path = os.path.join(seq_path, 'annotations')
            n_masks, out_paths, class_label_to_count, class_ids = save_masks(
                seq_path, voc_path, out_mask_size,
                out_border,
                fixed_ar,
                save_raw_mask,
                show_img=show_img,
                classes=classes,
                save_test=save_test,
                masks_per_seq=masks_per_seq,
                out_root_path=out_root_path,
                save_train=save_train,
                train_fnames=train_fnames,
                test_fnames=test_fnames,
                map_to_bbox=map_to_bbox,
                enable_out_suffix=enable_out_suffix,
                allow_skipping_images=allow_skipping_images,
                enable_xml_annotations=enable_xml_annotations,
                save_combined=save_combined,
                class_to_color=class_to_color,
                blended_vis=blended_vis,
                save_patches=save_patches,
            )
        total_n_masks += n_masks

        print('n_masks: {}'.format(n_masks))
        print('class_label_to_count:\n{}'.format(class_label_to_count))

        print()

        for _type in out_paths:
            try:
                all_out_paths[_type] += out_paths[_type]
            except KeyError:
                all_out_paths[_type] = out_paths[_type]

        for _type in class_label_to_count:
            try:
                all_class_label_to_count[_type] += class_label_to_count[_type]
            except KeyError:
                all_class_label_to_count[_type] = class_label_to_count[_type]

        if _exit:
            break

    print('total_n_masks: {}'.format(total_n_masks))
    print('all_class_label_to_count:\n{}'.format(all_class_label_to_count))
    print('crop: {}'.format(crop))

    if params.extract_features:
        load_features = 0
        if features_save_path is not None and os.path.exists(features_save_path):
            load_features = 1

        all_img_paths = all_out_paths['img']

        assert len(all_img_paths) == total_n_masks, "mismatch in output paths counts"

        import feature_extractor

        features, feature_colors = feature_extractor.run(
            all_img_paths, total_n_masks, crop,
            save_path=features_save_path, load=load_features)

        import manifold_embedder

        manifold_embedder.run(features, class_ids, feature_colors)

    if combine_seq:
        if save_train:

            out_train_img_path = os.path.join(out_root_path, 'images')
            out_train_mask_path = os.path.join(out_root_path, 'labels')
            # out_train_vis_path = os.path.join(out_root_path, 'vis')

            print('Writing combined training sequence to: {}, {}'.format(
                out_train_img_path, out_train_mask_path))

            os.makedirs(out_train_img_path, exist_ok=1)
            os.makedirs(out_train_mask_path, exist_ok=1)
            # os.makedirs(out_train_vis_path, exist_ok=1)

            combined_csv = []
            for out_seq_name in train_fnames:
                seq_fnames, seq_root, seq_csv, seq_csv_path = train_fnames[out_seq_name]
                for src_img_fname, src_mask_fname in seq_fnames:
                    dst_img_fname = os.path.join(out_train_img_path, '{}_{}'.format(
                        out_seq_name, os.path.basename(src_img_fname)))
                    dst_mask_fname = os.path.join(out_train_mask_path, '{}_{}'.format(
                        out_seq_name, os.path.basename(src_mask_fname)))
                    shutil.move(src_img_fname, dst_img_fname)
                    shutil.move(src_mask_fname, dst_mask_fname)
                if seq_csv_path:
                    for raw_data in seq_csv:
                        raw_data['filename'] = '{}_{}'.format(out_seq_name, raw_data['filename'])
                    combined_csv += seq_csv
                    os.remove(seq_csv_path)
                if isEmpty(seq_root):
                    shutil.rmtree(seq_root)
                else:
                    print('Not deleting non-empty {}'.format(seq_root))
            combined_csv_path = os.path.join(out_train_img_path, 'annotations.csv')
            pd.DataFrame(combined_csv).to_csv(combined_csv_path)

        if save_test:
            out_test_img_path = os.path.join(out_root_path, 'test') if save_train else out_root_path
            print('Wrtiting combined test sequence to: {}'.format(out_test_img_path))
            if not os.path.isdir(out_test_img_path):
                os.makedirs(out_test_img_path)
            combined_csv = []
            for out_seq_name in test_fnames:
                seq_fnames, seq_root, seq_csv, seq_csv_path = test_fnames[out_seq_name]
                for src_img_fname in seq_fnames:
                    dst_img_fname = os.path.join(out_test_img_path, '{}_{}'.format(
                        out_seq_name, os.path.basename(src_img_fname)))
                    shutil.move(src_img_fname, dst_img_fname)
                if seq_csv_path:
                    for raw_data in seq_csv:
                        raw_data['filename'] = '{}_{}'.format(out_seq_name, raw_data['filename'])
                    combined_csv += seq_csv
                    os.remove(seq_csv_path)
                if isEmpty(seq_root):
                    shutil.rmtree(seq_root)
                else:
                    print('Not deleting non-empty {}'.format(seq_root))
            combined_csv_path = os.path.join(out_test_img_path, 'annotations.csv')
            pd.DataFrame(combined_csv).to_csv(combined_csv_path)


if __name__ == '__main__':
    main()
