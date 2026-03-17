import cv2
import numpy as np
import sys
import pandas as pd
import pickle

sys.path.append("..")

from tf_api.utilities import processArguments, sortKey, resizeAR
from tf_api.utilities import get2DGaussianErrorFunctionArray, get2DGaussianArray, get2DGaussianArray2, hist_match, \
    hist_match2, compareHist, addBorder, getIOU
import os
import random
from tracking.Visualizer import ImageWriter

params = {
    'labels_path': 'data/wildlife_label_map.pbtxt',
    'list_file_name': '',
    'src_path': '',
    'bkg_path': '',
    'save_path': '',
    'load_path': '',
    'aug_seq_prefix': '',
    'n_aug': 19,
    'n_classes': 7,
    'img_ext': 'jpg',
    'batch_size': 1,
    'show_img': 1,
    'n_frames': 0,
    'vis_size': '1280x720',
    'border_ratio': 0.0,
    'make_square': 1,
    'mask_type': 0,
    'hist_match_type': 0,
    'random_bkgs': 0,
    'aug_seq_size': 1000,
    'visualize': 0,
    'bkg_size': '1280x720',
    'bkg_iou_thresh': 0.1,
}

processArguments(sys.argv[1:], params)
list_file_name = params['list_file_name']
_src_path = params['src_path']
labels_path = params['labels_path']
bkg_path = params['bkg_path']
n_aug = params['n_aug']
n_classes = params['n_classes']
save_path = params['save_path']
load_path = params['load_path']
aug_seq_prefix = params['aug_seq_prefix']
img_ext = params['img_ext']
batch_size = params['batch_size']
show_img = params['show_img']
n_frames = params['n_frames']
vis_size = params['vis_size']
border_ratio = params['border_ratio']
make_square = params['make_square']
mask_type = params['mask_type']
hist_match_type = params['hist_match_type']
random_bkgs = params['random_bkgs']
aug_seq_size = params['aug_seq_size']
visualize = params['visualize']
bkg_size = params['bkg_size']
bkg_iou_thresh = params['bkg_iou_thresh']

if hist_match_type == 1:
    _hist_match = hist_match
elif hist_match_type == 2:
    _hist_match = hist_match2

if vis_size:
    vis_width, vis_height = [int(x) for x in vis_size.split('x')]
else:
    vis_width, vis_height = 800, 600

bkg_file_list = [k for k in os.listdir(bkg_path) if k.endswith('.{:s}'.format(img_ext))]
n_bkgs = len(bkg_file_list)
if n_bkgs <= 0:
    raise SystemError('No background frames found')

print('n_bkgs: {}'.format(n_bkgs))
print('aug_seq_size: {}'.format(aug_seq_size))

bkg_file_list.sort(key=sortKey)
bkg_ids = [int(i) for i in range(n_bkgs)]

bkg_det_path = os.path.join(bkg_path, 'annotations.csv')
df = pd.read_csv(bkg_det_path)
bkg_data_dict = {}
for _, row in df.iterrows():
    filename = row['filename']
    xmin = float(row['xmin'])
    ymin = float(row['ymin'])
    xmax = float(row['xmax'])
    ymax = float(row['ymax'])
    class_name = row['class']
    bbox = [xmin, ymin, xmax, ymax]
    if not filename in bkg_data_dict:
        bkg_data_dict[filename] = {}
    if not class_name in bkg_data_dict[filename]:
        bkg_data_dict[filename][class_name] = []
    bkg_data_dict[filename][class_name].append(bbox)

if list_file_name:
    print('Reading sequence names from {}'.format(list_file_name))
    if not os.path.exists(list_file_name):
        raise IOError('List file: {} does not exist'.format(list_file_name))
    src_paths = [os.path.join(_src_path, x.strip()) for x in open(list_file_name).readlines() if x.strip()]
else:
    src_paths = [_src_path]

if not save_path:
    save_path = os.path.dirname(src_paths[0])
n_seq = len(src_paths)
pause_after_frame = 1

if bkg_size:
    bkg_width, bkg_height = [int(x) for x in bkg_size.split('x')]
if bkg_size:
    bkg_pkl_path = os.path.join(bkg_path, 'bkg_imgs_{}.pkl'.format(bkg_size))
else:
    bkg_pkl_path = os.path.join(bkg_path, 'bkg_imgs.pkl')

if os.path.isfile(bkg_pkl_path):
    print('Loading background images from {}'.format(bkg_pkl_path))
    with open(bkg_pkl_path, 'rb') as f:
        bkg_imgs = pickle.load(f)
else:
    print('Reading background image sequence from {}'.format(bkg_path))
    bkg_imgs = []
    for i in range(n_bkgs):
        _bkg_fname = bkg_file_list[i]
        bkg_img_path = os.path.join(bkg_path, _bkg_fname)
        bkg_img = cv2.imread(bkg_img_path)
        orig_shape = bkg_img.shape
        if bkg_size:
            bkg_img, resize_factor, _, _ = resizeAR(bkg_img, bkg_width, bkg_height,
                                                    return_factors=True, add_border=False)
            print('bkg_img.shape: ', bkg_img.shape)
        bkg_imgs.append({'name': _bkg_fname, 'image': bkg_img,
                         'resize_factor': resize_factor, 'orig_shape': orig_shape})
        sys.stdout.write('\rDone {:d} frames'.format(i + 1))
        sys.stdout.flush()
    print()
    print('Saving background images to {}'.format(bkg_pkl_path))
    with open(bkg_pkl_path, 'wb') as f:
        pickle.dump(bkg_imgs, f, pickle.HIGHEST_PROTOCOL)

frame_id = seq_frame_id = 0
aug_seq_id = 1
if not aug_seq_prefix:
    aug_seq_prefix = 'augmented'
aug_seq_name = '{:s}_{:d}'.format(aug_seq_prefix, aug_seq_id)
aug_seq_path = os.path.join(save_path, aug_seq_name)
if not os.path.isdir(aug_seq_path):
    os.makedirs(aug_seq_path)
video_out = ImageWriter(aug_seq_path)
print('Saving augmented sequence {} to {}'.format(aug_seq_id, aug_seq_path))
csv_raw = []

for seq_id in range(n_seq):
    src_path = src_paths[seq_id]
    seq_name = os.path.splitext(os.path.basename(src_path))[0]

    # dst_path = os.path.join(os.path.dirname(src_path), '{}_aug_{}'.format(seq_name, n_aug))
    # if not os.path.isdir(dst_path):
    #     os.makedirs(dst_path)
    # print('Saving output images to {}'.format(dst_path))

    det_path = os.path.join(src_path, 'annotations.csv')
    print('\nsequence {}/{}: {}'.format(seq_id + 1, n_seq, seq_name))

    print('Reading source images from: {}'.format(src_path))

    df = pd.read_csv(det_path)
    src_data_dict = {}
    for _, row in df.iterrows():
        filename = row['filename']
        xmin = float(row['xmin'])
        ymin = float(row['ymin'])
        xmax = float(row['xmax'])
        ymax = float(row['ymax'])
        class_name = row['class']
        target_id = row['target_id']
        bbox = [xmin, ymin, xmax, ymax]
        if not filename in src_data_dict:
            src_data_dict[filename] = []
        src_data_dict[filename].append({"class_name": class_name, "bbox": bbox, 'target_id': target_id})

    src_files = [k for k in os.listdir(src_path) if k.endswith('.{:s}'.format(img_ext))]
    total_frames = len(src_files)
    if total_frames <= 0:
        raise SystemError('No input frames found')
    print('total_frames: {}'.format(total_frames))
    src_files.sort(key=sortKey)

    for src_id, src_fname in enumerate(src_files):
        src_img_path = os.path.join(src_path, src_fname)
        src_img = cv2.imread(src_img_path)
        img_h, img_w = src_img.shape[:2]

        src_objs = src_data_dict[src_fname]
        n_objs = len(src_objs)

        video_out.write(src_img)

        src_bboxes = []
        class_names = []
        target_ids = []

        for obj_id in range(n_objs):
            src_obj = src_objs[obj_id]
            src_bbox = src_obj['bbox']
            class_name = src_obj['class_name']
            target_id = src_obj['target_id']

            _xmin, _ymin, _xmax, _ymax = src_bbox
            raw_data = {
                'target_id': target_id,
                'filename': video_out.filename,
                'width': img_w,
                'height': img_h,
                'class': class_name,
                'xmin': int(_xmin),
                'ymin': int(_ymin),
                'xmax': int(_xmax),
                'ymax': int(_ymax)
            }
            csv_raw.append(raw_data)
            src_bboxes.append(src_bbox)
            class_names.append(class_name)
            target_ids.append(target_id)

            if visualize:
                cv2.rectangle(src_img, (xmin, ymin), (xmax, ymax), (0, 255, 0))

        if visualize:
            _src_img = resizeAR(src_img, vis_width, vis_height)
            cv2.imshow('src_img', _src_img)

        if random_bkgs:
            random.shuffle(bkg_ids)
        else:
            bkg_ids.sort(key=lambda x: compareHist(src_img, bkg_imgs[x]['image'], method=0))

        frame_id += 1
        seq_frame_id += 1

        aug_id = 0
        while aug_id < n_aug:
            bkg_img = bkg_imgs[bkg_ids[aug_id]]['image']
            bkg_fname = bkg_imgs[bkg_ids[aug_id]]['name']
            bkg_resize_factor = bkg_imgs[bkg_ids[aug_id]]['resize_factor']
            bkg_orig_shape = bkg_imgs[bkg_ids[aug_id]]['orig_shape']
            bkg_img_h, bkg_img_w = bkg_img.shape[:2]

            bkg_boxes = bkg_data_dict[bkg_fname]['deer']
            n_bkg_boxes = len(bkg_boxes)
            _bkg_iou_thresh = bkg_iou_thresh
            bkg_iter = 0
            while True:
                found_bkg_bbox_ids = True
                bkg_bbox_ids = random.sample(range(n_bkg_boxes), n_objs)
                for i in range(n_objs):
                    bkg_box_1 = bkg_boxes[bkg_bbox_ids[i]]
                    for j in range(i + 1, n_objs):
                        bkg_box_2 = bkg_boxes[bkg_bbox_ids[j]]
                        bkg_iou = getIOU(bkg_box_1, bkg_box_2)
                        if bkg_iou > _bkg_iou_thresh:
                            print('bkg_iou: ', bkg_iou)
                            found_bkg_bbox_ids = False
                            break
                    if not found_bkg_bbox_ids:
                        break
                if found_bkg_bbox_ids:
                    break
                bkg_iter += 1
                if bkg_iter % 100 == 0:
                    _bkg_iou_thresh += 0.01

            bkg_bbox_size = {i: (bkg_boxes[i][2] - bkg_boxes[i][0]) * (bkg_boxes[i][3] - bkg_boxes[i][1])
                             for i in bkg_bbox_ids}
            bkg_bbox_ids.sort(key=lambda x: bkg_bbox_size[x])

            dst_img = np.copy(bkg_img)

            if hist_match_type:
                # bkg_img_matched = np.zeros_like(bkg_img, dtype=np.uint8)
                # for ch_id in range(3):
                #     bkg_img_matched[:, :, ch_id] = _hist_match(bkg_img[:, :, ch_id].squeeze(),
                #                                               src_img[:, :, ch_id].squeeze())
                # bkg_img = bkg_img_matched

                src_img_matched = np.zeros_like(src_img, dtype=np.uint8)
                for ch_id in range(3):
                    src_img_matched[:, :, ch_id] = _hist_match(src_img[:, :, ch_id].squeeze(),
                                                               bkg_img[:, :, ch_id].squeeze())
                src_patch = np.copy(src_img_matched[int(ymin):int(ymax), int(xmin):int(xmax), :])

            if visualize:
                dst_img_disp = np.copy(dst_img)
                bkg_img_disp = np.copy(bkg_img)


            video_out.write(dst_img)

            aug_id += 1
            frame_id += 1
            seq_frame_id += 1

            if visualize:
                dst_img_disp = resizeAR(dst_img_disp, vis_width, vis_height)
                cv2.imshow('dst_img', dst_img_disp)

                for _id in range(n_bkg_boxes):
                    _bbox = bkg_boxes[_id]
                    _xmin, _ymin, _xmax, _ymax = _bbox
                    _xmin, _ymin, _xmax, _ymax = int(_xmin), int(_ymin), int(_xmax), int(_ymax)
                    col = (0, 255, 0) if bkg_bbox_id == _id else (0, 0, 255)
                    cv2.rectangle(bkg_img_disp, (_xmin, _ymin), (_xmax, _ymax), col, 2)

                bkg_img_disp = resizeAR(bkg_img_disp, vis_width, vis_height)
                cv2.imshow('bkg_img', bkg_img_disp)
                k = cv2.waitKey(1 - pause_after_frame) & 0xFF
                if k == ord('q'):
                    sys.exit(0)
                elif k == 27:
                    break
                elif k == 32:
                    pause_after_frame = 1 - pause_after_frame
            else:
                sys.stdout.write('\rDone {:d} images for augmented sequence {:d} '
                                 'using frame {:d} in source sequence {:d}'.format(
                    seq_frame_id, aug_seq_id, src_id + 1, seq_id + 1))
                sys.stdout.flush()

            if seq_frame_id == aug_seq_size and frame_id > 0:
                aug_csv_path = os.path.join(aug_seq_path, 'annotations.csv')
                df = pd.DataFrame(csv_raw)
                df.to_csv(aug_csv_path)
                video_out.release()

                aug_seq_id += 1
                seq_frame_id = 0
                aug_seq_name = '{:s}_{:d}'.format(aug_seq_prefix, aug_seq_id)
                aug_seq_path = os.path.join(save_path, aug_seq_name)
                print('\nSaving augmented sequence {} to {}'.format(aug_seq_id, aug_seq_path))
                if not os.path.isdir(aug_seq_path):
                    os.makedirs(aug_seq_path)
                video_out = ImageWriter(aug_seq_path)
                csv_raw = []

    if visualize:
        cv2.destroyAllWindows()
    else:
        sys.stdout.write('\n')
        sys.stdout.flush()

def getAugmentedImage(dst_images, n_objs, src_bboxes, target_ids, class_names, bkg_bbox_ids):
    for obj_id in range(n_objs):
        src_bbox = src_bboxes[obj_id]
        target_id = target_ids[obj_id]
        class_name = class_names[obj_id]
        bkg_bbox_id = bkg_bbox_ids[obj_id]

        _xmin, _ymin, _xmax, _ymax = src_bbox
        _src_width, _src_height = _xmax - _xmin, _ymax - _ymin

        src_bbox = addBorder(src_bbox, src_img, border_ratio, make_square)
        xmin, ymin, xmax, ymax = src_bbox
        src_width, src_height = xmax - xmin, ymax - ymin
        src_ar = float(src_width) / float(src_height)

        offset_x, offset_y = _xmin - xmin, _ymin - ymin
        src_patch = np.copy(src_img[int(ymin):int(ymax), int(xmin):int(xmax), :])

        if hist_match_type:
            src_patch = np.copy(src_img_matched[int(ymin):int(ymax), int(xmin):int(xmax), :])

        bkg_bbox_orig = [int(k / bkg_resize_factor) for k in bkg_boxes[bkg_bbox_id]]
        bkg_bbox = addBorder(bkg_bbox_orig, bkg_img, border_ratio, make_square)

        _xmin, _ymin, _xmax, _ymax = bkg_bbox
        _xmin, _ymin, _xmax, _ymax = int(_xmin), int(_ymin), int(_xmax), int(_ymax)
        _width, _height = _xmax - _xmin, _ymax - _ymin
        bkg_ar = float(_width) / float(_height)

        if (src_ar > 1) != (bkg_ar > 1):
            _width, _height = _height, _width
            bkg_ar = float(_width) / float(_height)
            _xmax, _ymax = _xmin + _width, _ymin + _height
            if _xmax > bkg_img_w:
                diff = _xmax - bkg_img_w
                _xmin -= diff
                _xmax -= diff

            if _ymax > bkg_img_h:
                diff = _ymax - bkg_img_h
                _ymin -= diff
                _ymax -= diff

        if src_ar < bkg_ar:
            dst_width = _width
            dst_height = int(dst_width / src_ar)
        else:
            dst_height = _height
            dst_width = int(dst_height * src_ar)

        try:
            dst_patch = cv2.resize(src_patch, (dst_width, dst_height))
        except cv2.error as e:
            print()
            print('bkg_orig_shape', bkg_orig_shape)
            print('bkg_img.shape', bkg_img.shape)
            print('bkg_bbox_orig', bkg_bbox_orig)
            print('bkg_bbox', bkg_bbox)
            print('bkg_resize_factor', bkg_resize_factor)
            print('dst_width', dst_width)
            print('dst_height', dst_height)
            print('cv2.error: {}'.format(e))
            sys.exit()

        if mask_type == 0:
            dst_patch_mask = np.ones(dst_patch.shape[:2], dtype=np.float64)
        elif mask_type == 1:
            dst_patch_mask = get2DGaussianErrorFunctionArray(dst_width, dst_height)
        elif mask_type == 2:
            dst_patch_mask = get2DGaussianArray(dst_width, dst_height)
        elif mask_type == 3:
            dst_patch_mask = get2DGaussianArray2(dst_width, dst_height)

        dst_patch_mask_rgb = np.dstack((dst_patch_mask, dst_patch_mask, dst_patch_mask))

        start_row, start_col, end_row, end_col = _ymin, _xmin, _ymin + dst_height, _xmin + dst_width
        if end_col > bkg_img_w:
            diff = end_col - bkg_img_w
            start_col -= diff
            end_col -= diff

        if end_row > bkg_img_h:
            diff = end_row - bkg_img_h
            start_row -= diff
            end_row -= diff

        if start_row < 0 or start_col < 0:
            continue

        # mask_img = np.zeros_like(bkg_img, dtype=np.float64)
        # mask_img[start_row:end_row, start_col:end_col, :] = dst_patch_mask_rgb
        # _mask_img = resizeAR((mask_img * 255.0).astype(np.uint8), vis_width, vis_height)

        bkg_patch = bkg_img[start_row:end_row, start_col:end_col, :]

        dst_patch_matched = dst_patch
        # dst_patch_matched = np.zeros_like(dst_patch, dtype=np.uint8)
        # for ch_id in range(3):
        #     dst_patch_matched[:, :, ch_id] = hist_match(dst_patch[:, :, ch_id].squeeze(),
        #                                                 bkg_patch[:, :, ch_id].squeeze())

        # blended_patch = dst_patch_matched
        blended_patch = cv2.add(
            cv2.multiply(1.0 - dst_patch_mask_rgb, bkg_patch.astype(np.float64)),
            cv2.multiply(dst_patch_mask_rgb, dst_patch_matched.astype(np.float64))).astype(np.uint8)

        dst_img[start_row:end_row, start_col:end_col, :] = blended_patch

        resize_ratio = float(dst_width) / float(src_width)

        dst_xmin = int(start_col + offset_x * resize_ratio)
        dst_ymin = int(start_row + offset_y * resize_ratio)

        dst_xmax = int(dst_xmin + _src_width * resize_ratio)
        dst_ymax = int(dst_ymin + _src_height * resize_ratio)

        raw_data = {
            'target_id': target_id,
            'filename': video_out.filename,
            'width': bkg_img_w,
            'height': bkg_img_h,
            'class': class_name,
            'xmin': int(dst_xmin),
            'ymin': int(dst_ymin),
            'xmax': int(dst_xmax),
            'ymax': int(dst_ymax)
        }
        csv_raw.append(raw_data)

        if visualize:
            cv2.rectangle(dst_img_disp, (dst_xmin, dst_ymin), (dst_xmax, dst_ymax), (0, 255, 0), 2)
            cv2.imshow('dst_patch_mask_rgb', dst_patch_mask_rgb)
            cv2.imshow('dst_patch', dst_patch)
            cv2.imshow('dst_patch_matched', dst_patch_matched)
            print('dst_patch_mask_rgb.shape', dst_patch_mask_rgb.shape)
            print('bkg_patch.shape', bkg_patch.shape)
            print('dst_patch.shape', dst_patch.shape)

        dst_images.append(dst_img)

