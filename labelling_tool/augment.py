import cv2
import numpy as np
import sys
import pandas as pd
import itertools
import pickle
import math
from pprint import pprint

sys.path.append("..")

from tf_api.utilities import processArguments, sortKey, resizeAR
from tf_api.utilities import get2DGaussianErrorFunctionArray, get2DGaussianArray, get2DGaussianArray2, hist_match, \
    hist_match2, compareHist, addBorder, getIOU
import os
import random
from tracking.Visualizer import ImageWriter

params = {
    'labels_path': 'data/wildlife_label_map.pbtxt',

    'src_paths': '',
    'src_root_dir': '',
    'src_postfix': '',
    'img_dir': '',
    'img_ext': 'jpg',

    'mask_paths': '',
    'mask_root_dir': '',
    'mask_postfix': '',
    'mask_dir': '',
    'mask_ext': 'png',
    'mask_border': 0,

    'bkg_path': '',
    'save_path': '',
    'load_path': '',
    'bkg_bboxes_csv': '',
    'aug_seq_prefix': '',
    'n_aug': 19,
    'n_classes': 7,

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
    'only_one_src_obj': 0,
    'inclue_src_in_augmented': 1,
    'load_bkg': 1,
    'src_img_per_seq': 0,
    'flip_lr_prob': 0.0,
    'sample_frg_per_bkg': 0,
    'mask_threshold': 127,
}

processArguments(sys.argv[1:], params)

src_paths = params['src_paths']
src_root_dir = params['src_root_dir']
src_postfix = params['src_postfix']
img_dir = params['img_dir']
img_ext = params['img_ext']

mask_paths = params['mask_paths']
mask_root_dir = params['mask_root_dir']
mask_postfix = params['mask_postfix']
mask_dir = params['mask_dir']
mask_ext = params['mask_ext']
mask_border = params['mask_border']

labels_path = params['labels_path']
bkg_path = params['bkg_path']
n_aug = params['n_aug']
n_classes = params['n_classes']
save_path = params['save_path']
load_path = params['load_path']
aug_seq_prefix = params['aug_seq_prefix']
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
only_one_src_obj = params['only_one_src_obj']
inclue_src_in_augmented = params['inclue_src_in_augmented']
load_bkg = params['load_bkg']
bkg_bboxes_csv = params['bkg_bboxes_csv']
src_img_per_seq = params['src_img_per_seq']
flip_lr_prob = params['flip_lr_prob']
sample_frg_per_bkg = params['sample_frg_per_bkg']
mask_threshold = params['mask_threshold']

if flip_lr_prob > 0:
    print('Random horizontal flipping enabled with probability {}'.format(flip_lr_prob))

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

if src_paths:
    if os.path.isfile(src_paths):
        print('Reading source sequence names from {}'.format(src_paths))
        src_paths = [x.strip() for x in open(src_paths).readlines() if x.strip()]
    else:
        src_paths = src_paths.split(',')
    if src_root_dir:
        src_paths = [os.path.join(src_root_dir, name) for name in src_paths]
elif src_root_dir:
    src_paths = [os.path.join(src_root_dir, name) for name in os.listdir(src_root_dir) if
                 os.path.isdir(os.path.join(src_root_dir, name))]
else:
    raise IOError('Either src_paths or src_root_dir must be provided')

if src_postfix:
    src_paths = ['{}_{}'.format(name, src_postfix) for name in src_paths]

if img_dir:
    src_paths = [os.path.join(name, img_dir) for name in src_paths]

src_paths.sort(key=sortKey)

if mask_paths or mask_root_dir:
    mask_type = 4

if mask_type == 4:
    if mask_paths:
        if os.path.isfile(mask_paths):
            print('Reading source sequence names from {}'.format(mask_paths))
            mask_paths = [x.strip() for x in open(mask_paths).readlines() if x.strip()]
        else:
            mask_paths = mask_paths.split(',')
        if mask_root_dir:
            mask_paths = [os.path.join(mask_root_dir, name) for name in mask_paths]
    elif mask_root_dir:
        mask_paths = [os.path.join(mask_root_dir, name) for name in os.listdir(mask_root_dir) if
                      os.path.isdir(os.path.join(mask_root_dir, name))]
    else:
        raise IOError('Either mask_paths or mask_root_dir must be provided for loading external masks')

    if len(mask_paths) != len(src_paths):
        raise IOError('Mismatch between no. of mask_paths: {} and src_paths: {}'.format(
            len(mask_paths), len(src_paths)))
    if mask_postfix:
        mask_paths = ['{}_{}'.format(name, mask_postfix) for name in mask_paths]

    if mask_dir:
        mask_paths = [os.path.join(name, mask_dir) for name in mask_paths]
    mask_paths.sort(key=sortKey)

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

if load_bkg and os.path.isfile(bkg_pkl_path):
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
        bkg_imgs.append({
            'name': _bkg_fname,
            'path': bkg_img_path,
            'image': bkg_img,
            'resize_factor': resize_factor,
            'orig_shape': orig_shape}
        )
        sys.stdout.write('\rDone {:d} frames'.format(i + 1))
        sys.stdout.flush()
    print()
    print('Saving background images to {}'.format(bkg_pkl_path))
    with open(bkg_pkl_path, 'wb') as f:
        pickle.dump(bkg_imgs, f, pickle.HIGHEST_PROTOCOL)

_bkg_seq_name = os.path.basename(bkg_path)

external_bkg_bboxes = {}
external_bkg_bboxes_path = ''
if bkg_bboxes_csv:

    print('Reading background bboxes from external csv {}'.format(bkg_bboxes_csv))
    df = pd.read_csv(bkg_bboxes_csv)
    ex = {}
    for _, row in df.iterrows():
        filename = row['filename']
        xmin = float(row['xmin'])
        ymin = float(row['ymin'])
        xmax = float(row['xmax'])
        ymax = float(row['ymax'])

        width = float(row['width'])
        height = float(row['height'])

        bbox = [xmin, ymin, xmax, ymax]
        img_size = [width, height]

        if filename not in external_bkg_bboxes:
            external_bkg_bboxes[filename] = []
        external_bkg_bboxes[filename].append([bbox, img_size])

    print('Found {} bboxes'.format(len(external_bkg_bboxes)))

    external_bkg_bboxes_path = os.path.dirname(bkg_bboxes_csv)
    # external_bkg_bboxes_files = [os.path.join(external_bkg_bboxes_path, k)
    #                              for k in os.listdir(external_bkg_bboxes_path)
    #                              if k.endswith('.{:s}'.format(img_ext))]
    # external_bkg_bboxes_files.sort(key=sortKey)

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
filenames = []

all_foregrounds = []
if sample_frg_per_bkg > 0:
    for seq_id in range(n_seq):
        src_path = src_paths[seq_id]
        src_files = [k for k in os.listdir(src_path) if k.endswith('.{:s}'.format(img_ext))]
        src_files.sort(key=sortKey)
        total_frames = len(src_files)
        if total_frames <= 0:
            raise SystemError('No input frames found')

        for src_id, src_fname in enumerate(src_files):
            src_img_path = os.path.join(src_path, src_fname)
            all_foregrounds.append(src_img_path)

    n_frgs = len(all_foregrounds)

    if sample_frg_per_bkg > n_frgs:
        raise IOError('sample_frg_per_bkg: {} exceeds n_frgs: {}'.format(sample_frg_per_bkg, n_frgs))

    n_samples = n_bkgs * sample_frg_per_bkg
    print('Generating {} samples from {} bkgs and {} frgs with {} sample_frgs_per_bkg'.format(
        n_samples, n_bkgs, n_frgs, sample_frg_per_bkg))

    sample_bkg_ids = list(range(n_bkgs))
    for _ in range(1, sample_frg_per_bkg):
        sample_bkg_ids += list(np.random.permutation(range(n_bkgs)))

    # sample_bkg_ids = list(range(n_bkgs)) * sample_frg_per_bkg

    n_reps = int(math.ceil(n_samples/n_frgs))
    sample_frg_ids = list(range(n_frgs))
    for _ in range(1, n_reps):
        sample_frg_ids += list(np.random.permutation(range(n_frgs)))

    sample_frg_bkg_ids = list(zip(sample_bkg_ids, sample_frg_ids))

    while True:
        sample_frg_bkg_ids = list(dict.fromkeys(sample_frg_bkg_ids))
        if len(sample_frg_bkg_ids)>=n_samples:
            break

        sample_bkg_ids = []
        for _ in range(sample_frg_per_bkg):
            sample_bkg_ids += list(np.random.permutation(range(n_bkgs)))
        sample_frg_ids = []
        for _ in range(n_reps):
            sample_frg_ids += list(np.random.permutation(range(n_frgs)))

        sample_frg_bkg_ids += list(zip(sample_bkg_ids, sample_frg_ids))

    sample_frg_bkg_ids = sample_frg_bkg_ids[:n_samples]
    aug_seq_path = aug_seq_path.replace(os.sep, '/')
    seq_to_samples = {
        aug_seq_path: []
    }
    # foreground_cmbs = list(itertools.combinations(all_foregrounds, frg_per_bkg))

_frg_id = -1

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
    src_files.sort(key=sortKey)
    total_frames = len(src_files)
    if total_frames <= 0:
        raise IOError('No input frames found')

    if src_img_per_seq > 0:
        print('Including only {}/{} frames'.format(src_img_per_seq, total_frames))
        src_files = [src_files[i] for i in range(src_img_per_seq)]
        total_frames = len(src_files)

    print('total_frames: {}'.format(total_frames))

    if mask_type == 4:
        mask_path = mask_paths[seq_id]
        mask_seq_name = os.path.splitext(os.path.basename(mask_path))[0]

        if seq_name not in mask_seq_name:
            raise IOError('mask_seq_name: {} does not match with seq_name: {}'.format(mask_seq_name, seq_name))

        print('Reading mask images from: {}'.format(mask_path))
        mask_files = [k for k in os.listdir(mask_path) if k.endswith('.{:s}'.format(mask_ext))]
        mask_files.sort(key=sortKey)
        total_mask_frames = len(mask_files)
        if total_mask_frames != total_frames:
            raise IOError('Mismatch between total_frames: {} and total_mask_frames: {}'.format(
                total_frames, total_mask_frames
            ))

    for src_id, src_fname in enumerate(src_files):
        src_img_path = os.path.join(src_path, src_fname)
        src_img = cv2.imread(src_img_path)

        _frg_id += 1

        src_seq_path = os.path.dirname(src_img_path)
        src_seq_name = os.path.splitext(os.path.basename(src_seq_path))[0]
        src_fname_no_ext = os.path.splitext(os.path.basename(src_img_path))[0]

        img_h, img_w = src_img.shape[:2]

        src_objs = src_data_dict[src_fname]

        n_objs = len(src_objs)

        src_bboxes = []
        class_names = []
        target_ids = []

        if inclue_src_in_augmented:
            video_out.write(src_img)
            seq_frame_id += 1

        for obj_id in range(n_objs):
            src_obj = src_objs[obj_id]
            src_bbox = src_obj['bbox']
            class_name = src_obj['class_name']
            target_id = src_obj['target_id']

            _xmin, _ymin, _xmax, _ymax = src_bbox

            if inclue_src_in_augmented:
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

            # filenames.append(filename)
            src_bboxes.append(src_bbox)
            class_names.append(class_name)
            target_ids.append(target_id)

            if visualize:
                src_img_disp = np.copy(src_img)
                cv2.rectangle(img=src_img_disp,
                              pt1=(int(_xmin), int(_ymin)),
                              pt2=(int(_xmax), int(_ymax)),
                              color=(0, 255, 0),
                              thickness=2)

        if visualize:
            _src_img = resizeAR(src_img_disp, vis_width, vis_height)
            cv2.imshow('src_img', _src_img)

        if random_bkgs:
            random.shuffle(bkg_ids)
        else:
            bkg_ids.sort(key=lambda x: compareHist(src_img, bkg_imgs[x]['image'], method=0))

        frame_id += 1

        if only_one_src_obj:
            n_objs = 1

        if mask_type == 4:
            if n_objs > 1:
                raise NotImplementedError('External masks are only supported with one source object per image')

            mask_img_path = os.path.join(mask_path, mask_files[src_id])
            mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

            if mask_threshold:
                    _, mask_img = cv2.threshold(mask_img, mask_threshold, 1, cv2.THRESH_BINARY)
            else:
                mask_img = mask_img / 255.0

            if mask_border > 0:
                mask_img = mask_img[mask_border:-mask_border, mask_border:-mask_border, :]

        if n_aug == 0:
            _n_aug = len(bkg_ids)
        else:
            _n_aug = n_aug

        _aug_id = 0

        while _aug_id < _n_aug:
            bkg_img = bkg_imgs[bkg_ids[_aug_id]]['image']
            bkg_fname = bkg_imgs[bkg_ids[_aug_id]]['name']
            bkg_path = bkg_imgs[bkg_ids[_aug_id]]['path']

            bkg_seq_path = os.path.dirname(bkg_path)
            bkg_seq_name = os.path.basename(bkg_seq_path)
            bkg_fname_no_ext = os.path.splitext(os.path.basename(bkg_path))[0]

            bkg_img_h, bkg_img_w = bkg_img.shape[:2]

            bkg_resize_factor = bkg_imgs[bkg_ids[_aug_id]]['resize_factor']
            bkg_orig_shape = bkg_imgs[bkg_ids[_aug_id]]['orig_shape']

            if external_bkg_bboxes:
                out_prefix = '{}_{}_{}_{}'.format(
                    src_seq_name, src_fname_no_ext, bkg_seq_name, bkg_fname_no_ext)

                matching_fnames = [k for k in external_bkg_bboxes.keys() if out_prefix in k]
                if not matching_fnames:
                    raise IOError('No matching filenames found for {}'.format(out_prefix))
                if len(matching_fnames) > 1:
                    raise IOError('Multiple matching filenames found for {}:\n {}'.format(out_prefix, matching_fnames))
                matching_fnames = matching_fnames[0]
                # print('Getting box for {} from {}'.format(out_prefix, matching_fnames))
                _img_h, _img_w = bkg_orig_shape[:2]
                ext_bbox, ext_img_size = external_bkg_bboxes[matching_fnames][0]

                _bkg_bbox = ext_bbox

                # ext_xmin, ext_ymin, ext_xmax, ext_ymax  = ext_bbox
                ext_img_w, ext_img_h = ext_img_size

                if ext_img_w != bkg_img_w or ext_img_h != bkg_img_h:
                    raise IOError('Mismatch between ext_img_size: {}x{} and bkg_img_size: {}x{}'.format(
                        ext_img_w, ext_img_h, bkg_img_w, bkg_img_h,
                    ))

                # resize_x, resize_y = float(_img_h)/float(ext_img_w), float(_img_w)/float(ext_img_h)
                # _bkg_bbox = [ext_xmin*resize_x, ext_ymin*resize_y, ext_xmax*resize_x, ext_ymax*resize_y]

                bkg_boxes = [_bkg_bbox, ]
                n_bkg_boxes = 1
                bkg_bbox_ids = [0, ]
            else:

                bkg_boxes = bkg_data_dict[bkg_fname]['bear']
                n_bkg_boxes = len(bkg_boxes)
                _bkg_iou_thresh = bkg_iou_thresh
                bkg_iter = 0
                while True:
                    found_bkg_bbox_ids = True
                    # bkg_bbox_ids = random.sample(range(n_bkg_boxes), n_objs)
                    bkg_bbox_ids = list(np.random.permutation(range(n_bkg_boxes)))
                    for i in range(n_objs):
                        bkg_box_1 = bkg_boxes[bkg_bbox_ids[i]]
                        for j in range(i + 1, n_objs):
                            bkg_box_2 = bkg_boxes[bkg_bbox_ids[j]]
                            bkg_iou = getIOU(bkg_box_1, bkg_box_2)
                            if bkg_iou > _bkg_iou_thresh:
                                # print('bkg_iou: ', bkg_iou)
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

            enable_flip = 0
            curr_csv_raw = []
            # dst_bboxes = []
            for obj_id in range(n_bkg_boxes):
                src_bbox = src_bboxes[obj_id % n_objs]
                target_id = target_ids[obj_id % n_objs]
                class_name = class_names[obj_id % n_objs]
                bkg_bbox_id = bkg_bbox_ids[obj_id]

                _xmin, _ymin, _xmax, _ymax = src_bbox
                _src_width, _src_height = _xmax - _xmin, _ymax - _ymin

                orig_src_patch = np.copy(src_img[int(_ymin):int(_ymax), int(_xmin):int(_xmax), :])

                src_bbox = addBorder(src_bbox, src_img, border_ratio, make_square)
                xmin, ymin, xmax, ymax = src_bbox
                src_width, src_height = xmax - xmin, ymax - ymin
                src_ar = float(src_width) / float(src_height)

                offset_x, offset_y = _xmin - xmin, _ymin - ymin
                src_patch = np.copy(src_img[int(ymin):int(ymax), int(xmin):int(xmax), :])

                if hist_match_type:
                    src_patch = np.copy(src_img_matched[int(ymin):int(ymax), int(xmin):int(xmax), :])

                if external_bkg_bboxes:
                    bkg_bbox_orig = bkg_boxes[bkg_bbox_id]
                else:
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

                start_row, start_col, end_row, end_col = _ymin, _xmin, _ymin + dst_height, _xmin + dst_width
                if end_col > bkg_img_w:
                    diff = end_col - bkg_img_w
                    dst_width -= diff
                    end_col -= diff

                if end_row > bkg_img_h:
                    diff = end_row - bkg_img_h
                    dst_height -= diff
                    end_row -= diff

                if start_row < 0 or start_col < 0:
                    print('Skipping bkg_box {} as having invalid normalized box: {}'.format(
                        obj_id, (start_row, end_row, start_col, end_col)))
                    continue

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
                    raise cv2.error(e)

                if mask_type == 0:
                    dst_patch_mask = np.ones(dst_patch.shape[:2], dtype=np.float64)
                elif mask_type == 1:
                    dst_patch_mask = get2DGaussianErrorFunctionArray(dst_width, dst_height)
                elif mask_type == 2:
                    dst_patch_mask = get2DGaussianArray(dst_width, dst_height)
                elif mask_type == 3:
                    dst_patch_mask = get2DGaussianArray2(dst_width, dst_height)
                elif mask_type == 4:
                    dst_patch_mask = cv2.resize(mask_img, (dst_width, dst_height))

                dst_patch_mask_rgb = np.dstack((dst_patch_mask, dst_patch_mask, dst_patch_mask))

                # mask_img = np.zeros_like(bkg_img, dtype=np.float64)
                # mask_img[start_row:end_row, start_col:end_col, :] = dst_patch_mask_rgb
                # _mask_img = resizeAR((mask_img * 255.0).astype(np.uint8), vis_width, vis_height)

                bkg_patch = bkg_img[start_row:end_row, start_col:end_col, :]

                dst_patch_matched = dst_patch

                if flip_lr_prob == -1:
                    # if not external_bkg_bboxes_files:
                    #     raise IOError('external_bkg_bboxes_files is empty')

                    external_bkg_bboxes_file = os.path.join(external_bkg_bboxes_path, matching_fnames)
                    external_bkg_bboxes_img = cv2.imread(external_bkg_bboxes_file)

                    if external_bkg_bboxes_img is None:
                        raise IOError('external_bkg_bboxes_file could not be read: {}'.format(external_bkg_bboxes_file))

                    ext_xmin, ext_ymin, ext_xmax, ext_ymax = ext_bbox

                    external_bkg_bboxes_patch = np.copy(external_bkg_bboxes_img[int(ext_ymin):int(ext_ymax),
                                                        int(ext_xmin):int(ext_xmax), :])

                    external_bkg_bboxes_patch = cv2.resize(external_bkg_bboxes_patch, orig_src_patch.shape[:2][::-1])

                    patch_1 = orig_src_patch.astype(np.float32)
                    patch_2 = external_bkg_bboxes_patch.astype(np.float32)
                    patch_2_flipped = np.fliplr(patch_2)

                    ncc = cv2.matchTemplate(
                        patch_1, patch_2, method=cv2.TM_CCORR_NORMED)

                    ncc_flipped = cv2.matchTemplate(
                        patch_1, patch_2_flipped, method=cv2.TM_CCORR_NORMED)

                    if ncc_flipped > ncc:
                        enable_flip = 1

                    if visualize:
                        cv2.imshow('orig_src_patch', orig_src_patch)
                        cv2.imshow('external_bkg_bboxes_patch', external_bkg_bboxes_patch)

                        print('matching_fnames: {}'.format(matching_fnames))
                        print('external_bkg_bboxes_file: {}'.format(external_bkg_bboxes_file))
                        print('external_bkg_bboxes_img: {}'.format(external_bkg_bboxes_img.shape))
                        print('ext_bbox: {}'.format(ext_bbox))
                        print('bkg_bbox: {}'.format(bkg_bbox))

                        print('src_patch: {} external_bkg_bboxes_patch: {}'.format(
                            src_patch.shape, external_bkg_bboxes_patch.shape))
                        print('ncc: {} ncc_flipped: {}'.format(ncc, ncc_flipped))
                        print('enable_flip: {}'.format(enable_flip))

                        # cv2.waitKey(0)
                elif flip_lr_prob > 0 and random.random() < flip_lr_prob:
                    enable_flip = 1

                if enable_flip:
                    # print('Flipping horizontally')
                    dst_patch_matched = np.fliplr(dst_patch_matched)
                    dst_patch_mask_rgb = np.fliplr(dst_patch_mask_rgb)

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
                    'filename': None,
                    'width': bkg_img_w,
                    'height': bkg_img_h,
                    'class': class_name,
                    'xmin': int(dst_xmin),
                    'ymin': int(dst_ymin),
                    'xmax': int(dst_xmax),
                    'ymax': int(dst_ymax)
                }
                curr_csv_raw.append(raw_data)

                # dst_bboxes.append([dst_xmin, dst_ymin, dst_xmax, dst_ymax])

                if visualize:
                    dst_img_disp = np.copy(dst_img)
                    bkg_img_disp = np.copy(bkg_img)
                    cv2.rectangle(img=dst_img_disp,
                                  pt1=(dst_xmin, dst_ymin),
                                  pt2=(dst_xmax, dst_ymax),
                                  color=(0, 255, 0),
                                  thickness=2)
                    # cv2.rectangle(dst_img_disp, (dst_xmin, dst_ymin), (dst_xmax, dst_ymax), 255, 2)
                    cv2.imshow('dst_patch_mask_rgb', dst_patch_mask_rgb)
                    cv2.imshow('dst_patch', dst_patch)
                    cv2.imshow('dst_patch_matched', dst_patch_matched)
                    print('dst_patch_mask_rgb.shape', dst_patch_mask_rgb.shape)
                    print('bkg_patch.shape', bkg_patch.shape)
                    print('dst_patch.shape', dst_patch.shape)

                if len(curr_csv_raw) >= n_objs:
                    break

            out_prefix = '{}_{}_{}_{}_{}'.format(
                src_seq_name, src_fname_no_ext, bkg_seq_name, bkg_fname_no_ext, bkg_bbox_id)

            video_out.write(dst_img, prefix=out_prefix)

            if sample_frg_per_bkg > 0:
                _frg_bkg_id = (_aug_id, _frg_id)
                # _frg_bkg_id = '{}_{}'.format(_aug_id, _frg_id)
                pprint(_frg_bkg_id)
                if _frg_bkg_id in sample_frg_bkg_ids:
                    sample_frg_bkg_ids.remove(_frg_bkg_id)
                    seq_to_samples[aug_seq_path].append(video_out.curr_file_path.replace(os.sep, '/'))
            # if video_out.filename in filenames:
            #     raise IOError('Amazingly annoying duplicate filename: {} for src_img_path: {}\n'
            #                   'curr_csv_raw:\n {}\n'
            #                   'n_objs: {}\n'
            #                   'aug_id: {}'.format(
            #         video_out.filename, src_img_path, curr_csv_raw, n_objs, aug_id))

            # filenames.append(video_out.filename)

            if not curr_csv_raw:
                raise IOError('No valid boxes found for {} in {}'.format(src_img_path, bkg_fname))
            elif len(curr_csv_raw) != n_objs:
                raise IOError('Incorrect entry count: {} in curr_csv_raw:\n {}\nExpected {}'.format(
                    len(curr_csv_raw), curr_csv_raw, n_objs))

            for raw_data in curr_csv_raw:
                raw_data['filename'] = video_out.filename
                csv_raw.append(raw_data)

            _aug_id += 1
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
            # else:
                # sys.stdout.write('\rDone {:d} images for augmented sequence {:d} '
                #                  'using frame {:d} in source sequence {:d}'.format(
                #     seq_frame_id, aug_seq_id, src_id + 1, seq_id + 1))
                # sys.stdout.flush()



            if seq_frame_id == aug_seq_size and frame_id > 0:
                aug_csv_path = os.path.join(aug_seq_path, 'annotations.csv')
                # print('csv_raw: ', csv_raw)
                print('\nWriting annotations to: {}'.format(aug_csv_path))
                csv_columns = ['target_id', 'filename', 'width', 'height',
                               'class', 'xmin', 'ymin', 'xmax', 'ymax']
                pd.DataFrame(csv_raw).to_csv(aug_csv_path, columns=csv_columns)
                video_out.release()

                # print('filenames:'.format(filenames))
                # pprint(filenames)
                #
                # print('csv_raw:')
                # pprint(csv_raw)
                #
                #
                # if len(csv_raw) != len(filenames):
                #     raise IOError('Amazingly annoying mismatch between len(csv_raw): {} and len(filenames): {}'.format(
                #         len(csv_raw), len(filenames)
                #     ))
                aug_seq_id += 1
                seq_frame_id = 0
                aug_seq_name = '{:s}_{:d}'.format(aug_seq_prefix, aug_seq_id)
                aug_seq_path = os.path.join(save_path, aug_seq_name)
                print('\nSaving augmented sequence {} to {}'.format(aug_seq_id, aug_seq_path))
                if not os.path.isdir(aug_seq_path):
                    os.makedirs(aug_seq_path)
                video_out = ImageWriter(aug_seq_path)
                csv_raw = []
                filenames = []

if csv_raw:
    aug_csv_path = os.path.join(aug_seq_path, 'annotations.csv')
    print('\nWriting annotations to: {}'.format(aug_csv_path))
    csv_columns = ['target_id', 'filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    pd.DataFrame(csv_raw).to_csv(aug_csv_path, columns=csv_columns)
    video_out.release()

if sample_frg_per_bkg > 0:
    if sample_frg_bkg_ids:
        print('sample_frg_bkg_ids: {}'.format(sample_frg_bkg_ids))
    seq_to_samples_fname = os.path.join(aug_seq_path, 'seq_to_samples.txt')
    print('Writing seq_to_samples to: {}'.format(seq_to_samples_fname))
    with open(seq_to_samples_fname, 'w') as logFile:
        pprint(seq_to_samples, logFile)

if visualize:
    cv2.destroyAllWindows()
else:
    sys.stdout.write('\n')
    sys.stdout.flush()
