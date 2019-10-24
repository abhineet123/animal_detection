import os, sys, glob, re
import numpy as np
import pandas as pd
import cv2
import shutil
from PIL import Image

from libs.pascal_voc_io import PascalVocReader, PascalVocWriter
from libs.frames_readers import get_frames_reader
from libs.shape import Shape

sys.path.append('..')
from tf_api.utilities import processArguments, sortKey, isEmpty, map_mask_to_bbox

_pause = 1
_exit = 0


def saveMasks(seq_path, xml_path, out_mask_size, out_border, fixed_ar,
              save_raw_mask, show_img, out_root_path='', save_test=1,
              save_train=1, frames_reader=None, masks_per_seq=0, enable_out_suffix=1,
              train_fnames=None, test_fnames=None, map_to_bbox=0, out_img_dir='',
              enable_xml_annotations=0, allow_skipping_images=0):
    global _pause, _exit

    if not xml_path or not os.path.isdir(xml_path):
        raise IOError('Folder containing the loaded boxes does not exist: {}'.format(xml_path))

    files = glob.glob(os.path.join(xml_path, '*.xml'))
    n_files = len(files)
    if n_files == 0:
        raise IOError('No loaded boxes found')

    if frames_reader is None:
        frames_reader = get_frames_reader(seq_path, save_as_bin=False)

    min_dim = max_dim = 0
    out_w, out_h = out_mask_size
    print('out_mask_size: {}'.format(out_mask_size))

    if out_w == -1 and out_h == -1:
        out_w = out_h = 0

    if out_w == -1:
        max_dim = out_h
    elif out_h == -1:
        min_dim = out_w

    if fixed_ar:
        print('Using fixed aspect ratio: {}'.format(fixed_ar))

    print('out_border: {}'.format(out_border))

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

    seq_root_dir = os.path.dirname(seq_path)
    seq_name = os.path.basename(seq_path)
    if not out_root_path:
        out_root_path = os.path.join(seq_root_dir, 'masks')

    if not enable_out_suffix:
        out_seq_name = seq_name
    else:
        if map_to_bbox:
            out_seq_name = '{}_mapped'.format(seq_name)
        else:
            out_seq_name = '{}_{}x{}'.format(seq_name, out_w, out_h)
            if fixed_ar:
                out_seq_name = '{}_ar_{}'.format(out_seq_name, fixed_ar)
            else:
                out_seq_name = '{}_{}'.format(out_seq_name, out_border)

        out_seq_name = out_seq_name.replace('.', 'p')

    train_root_path = os.path.join(out_root_path, out_seq_name)

    if not save_test and not save_train:
        raise AssertionError('Either save_test or save_train must be on')

    # print('Saving output sequences to  {}'.format(out_root_path))

    if save_train:
        out_img_root_path = train_root_path
        if out_img_dir:
            out_img_root_path = os.path.join(out_img_root_path, out_img_dir)

        out_mask_root_path = os.path.join(train_root_path, 'labels')
        print('Saving training mask sequence to {}'.format(train_root_path))

        if not os.path.isdir(out_img_root_path):
            os.makedirs(out_img_root_path)

        if not os.path.isdir(out_mask_root_path):
            os.makedirs(out_mask_root_path)

        if enable_xml_annotations:
            out_xml_path = os.path.join(out_img_root_path, 'annotations')
            print('Saving xml_annotations to {}'.format(out_xml_path))
            if not os.path.isdir(out_xml_path):
                os.makedirs(out_xml_path)

    if save_test:
        out_test_seq_name = out_seq_name + '_test'
        test_img_root_path = os.path.join(out_root_path, out_test_seq_name)

        print('Saving unlabeled testing mask sequence to {}'.format(test_img_root_path))
        if not os.path.isdir(test_img_root_path):
            os.makedirs(test_img_root_path)

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
        mask_pix_val = (1, 1, 1)
    else:
        mask_pix_val = (255, 255, 255)

    n_masks = 0

    _train_fnames = []
    _test_fnames = []

    _exit_seq = 0

    disp_img = None

    for file_id, file in enumerate(files):
        xml_reader = PascalVocReader(file)
        filename = os.path.basename(xml_reader.filename)
        filename_no_ext = os.path.splitext(filename)[0]
        # file_id = int(re.sub("\D", "", filename))

        # print('filename: {}'.format(filename))
        # print('file_id: {}'.format(file_id))

        img = frames_reader.get_frame_by_name(filename, convert_to_rgb=0)
        if img is None:
            print('image {} could not be read'.format(filename))
            continue

        img_h, img_w = img.shape[:2]

        mask_img = None

        shapes = xml_reader.getShapes()
        n_shapes = len(shapes)
        if n_shapes > 1:
            print('{} boxes found for {} in {}'.format(n_shapes, filename, file))

        obj_id = 0

        img_written = 0
        for shape in shapes:
            label, points, _, _, difficult, bbox_source, id_number, score, mask, mask_img = shape
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

            if fixed_ar:
                w, h = xmax - xmin, ymax - ymin
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

            if map_to_bbox:
                if not img_written:
                    img_written = 1
                    out_img_path = os.path.join(img_root_path, filename)
                    cv2.imwrite(out_img_path, img)
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
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    disp_img = img
            else:
                img_out_fname = out_fname + '.jpg'
                if mask:
                    out_img_path = os.path.join(img_root_path, img_out_fname)
                    cv2.imwrite(out_img_path, patch_img)

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
                        bndbox = [_xmin, _ymin, _xmax, _ymax]
                        xml_writer.addBndBox(_xmin, _ymin, _xmax, _ymax,
                                             label, difficult, bbox_source, id_number, score, _mask)
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
                    cv2.rectangle(patch_img, (_xmin, _ymin), (_xmax, _ymax), (0, 255, 0), 2)
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

                curr_mask = cv2.fillPoly(curr_mask, np.array([mask_pts, ], dtype=np.int32), mask_pix_val)

                # print('min: {} max: {}'.format(
                #     np.min(mask_img.flatten()),
                #     np.max(mask_img.flatten()))
                # )

                if map_to_bbox:
                    mask_img = map_mask_to_bbox((xmin, ymin, xmax, ymax), curr_mask,
                                                fixed_ar, out_border, mask_img.shape, mask_img)
                else:
                    mask_img = curr_mask
                    out_mask_path = os.path.join(out_mask_root_path, mask_out_fname)
                    cv2.imwrite(out_mask_path, mask_img)

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

            if show_img and not map_to_bbox:
                # if _pause:
                #     print('frame {} :: {}'.format(file_id, filename))
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
                mask_out_fname = filename_no_ext + '.png'
                out_mask_path = os.path.join(out_mask_root_path, mask_out_fname)
                cv2.imwrite(out_mask_path, mask_img)

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
        if _exit:
            break

        sys.stdout.write('\rDone {:d}/{:d} files {:s} ({:d} masks found)'.format(
            file_id + 1, n_files, filename, n_masks))
        sys.stdout.flush()

        if masks_per_seq > 0 and n_masks >= masks_per_seq:
            break

    sys.stdout.write('\n')
    sys.stdout.flush()

    if not _exit_seq and save_train and n_masks == 0:
        raise IOError('\nNo masks found for {}\n'.format(seq_path))

    train_csv_path = test_csv_path = ''
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
        train_fnames[out_seq_name] = _train_fnames, train_root_path, csv_raw, train_csv_path

    if save_test and test_fnames is not None:
        test_fnames[out_test_seq_name] = _test_fnames, test_img_root_path, test_csv_raw, test_csv_path

    return n_masks


def main():
    params = {
        'class_names_path': '../labelling_tool/data//predefined_classes_orig.txt',
        'seq_paths': '',
        'root_dir': '',
        'save_file_name': '',
        'csv_file_name': '',
        'out_root_path': '',
        'map_folder': '',
        'load_path': '',
        'n_classes': 4,
        'out_border': 0,
        'fixed_ar': 0.0,
        'img_ext': 'png',
        'batch_size': 1,
        'show_img': 0,
        'save_video': 0,
        'out_mask_size': '0x0',
        'n_frames': 0,
        'codec': 'H264',
        'fps': 20,
        'save_raw_mask': 0,
        'save_test': 1,
        'save_train': 1,
        'masks_per_seq': 0,
        'combine_seq': 0,
        'map_to_bbox': 0,
        'out_img_dir': 'images',
        'enable_out_suffix': 1,
        'enable_xml_annotations': 1,
        'allow_skipping_images': 0,
    }
    processArguments(sys.argv[1:], params)
    seq_paths = params['seq_paths']
    root_dir = params['root_dir']
    out_border = params['out_border']
    fixed_ar = params['fixed_ar']
    out_mask_size = params['out_mask_size']
    show_img = params['show_img']
    save_raw_mask = params['save_raw_mask']
    save_test = params['save_test']
    masks_per_seq = params['masks_per_seq']
    out_root_path = params['out_root_path']
    enable_out_suffix = params['enable_out_suffix']
    enable_xml_annotations = params['enable_xml_annotations']
    allow_skipping_images = params['allow_skipping_images']
    save_train = params['save_train']
    combine_seq = params['combine_seq']
    map_to_bbox = params['map_to_bbox']
    out_img_dir = params['out_img_dir']

    if combine_seq and not out_root_path:
        raise IOError('out_root_path must be provided to combine sequences')

    if seq_paths:
        if os.path.isfile(seq_paths):
            seq_paths = [x.strip() for x in open(seq_paths).readlines() if x.strip()]
        else:
            seq_paths = seq_paths.split(',')
        if root_dir:
            seq_paths = [os.path.join(root_dir, name) for name in seq_paths]

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
    for seq_path in seq_paths:
        voc_path = os.path.join(seq_path, 'annotations')
        n_masks = saveMasks(seq_path, voc_path, out_mask_size, out_border, fixed_ar, save_raw_mask,
                            show_img=show_img, save_test=save_test, masks_per_seq=masks_per_seq,
                            out_root_path=out_root_path, save_train=save_train, out_img_dir=out_img_dir,
                            train_fnames=train_fnames, test_fnames=test_fnames, map_to_bbox=map_to_bbox,
                            enable_out_suffix=enable_out_suffix, allow_skipping_images=allow_skipping_images,
                            enable_xml_annotations=enable_xml_annotations)
        total_n_masks += n_masks

        if _exit:
            break
    print('total_n_masks: {}'.format(total_n_masks))

    if combine_seq:
        if save_train:

            out_train_img_path = os.path.join(out_root_path, 'images')
            out_train_mask_path = os.path.join(out_root_path, 'labels')

            print('Wrtiting combined training sequence to: {}, {}'.format(
                out_train_img_path, out_train_mask_path))

            if not os.path.isdir(out_train_img_path):
                os.makedirs(out_train_img_path)
            if not os.path.isdir(out_train_mask_path):
                os.makedirs(out_train_mask_path)
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
