import numpy as np
import os
import cv2
import math
import time
import sys
import pandas as pd


def str_to_list(_str, _type, _sep):
    return list(map(_type, _str.split(_sep)))


list_str_x = lambda _str: str_to_list(_str, int, 'x')
list_int_comma = lambda _str: str_to_list(_str, int, ',')
list_float_comma = lambda _str: str_to_list(_str, float, ',')
list_str_comma = lambda _str: str_to_list(_str, str, ',')


def computeThreshRecPrec(thresh_idx, score_thresholds, gt_counter_per_class,
                         conf_class, fp_class, tp_class, gt_class):
    _thresh = score_thresholds[thresh_idx]
    idx_thresh = [i for i, x in enumerate(conf_class) if x >= _thresh]

    # conf_class_thresh = [conf_class[i] for i in idx_thresh]
    fp_thresh = [fp_class[i] for i in idx_thresh]
    tp_thresh = [tp_class[i] for i in idx_thresh]

    tp_sum_thresh = np.sum(tp_thresh)
    fp_sum_thresh = np.sum(fp_thresh)

    if tp_sum_thresh > 0:
        _rec_thresh = float(tp_sum_thresh) / gt_counter_per_class[gt_class]
    else:
        _rec_thresh = 0
    try:
        _prec_thresh = float(tp_sum_thresh) / float(fp_sum_thresh + tp_sum_thresh)
    except ZeroDivisionError:
        _prec_thresh = 0

    # sys.stdout.write('\rDone {}/{}: {}'.format(thresh_idx+1, n_score_thresholds, _thresh))
    # sys.stdout.flush()

    return _rec_thresh, _prec_thresh

def sortKey(fname):
    fname = os.path.splitext(os.path.basename(fname))[0]
    # print('fname: ', fname)
    # split_fname = fname.split('_')
    # print('split_fname: ', split_fname)

    # nums = [int(s) for s in fname.split('_') if s.isdigit()]
    # non_nums = [s for s in fname.split('_') if not s.isdigit()]

    split_list = fname.split('_')
    key = ''

    for s in split_list:
        if s.isdigit():
            if not key:
                key = '{:08d}'.format(int(s))
            else:
                key = '{}_{:08d}'.format(key, int(s))
        else:
            if not key:
                key = s
            else:
                key = '{}_{}'.format(key, s)

    # for non_num in non_nums:
    #     if not key:
    #         key = non_num
    #     else:
    #         key = '{}_{}'.format(key, non_num)
    # for num in nums:
    #     if not key:
    #         key = '{:08d}'.format(num)
    #     else:
    #         key = '{}_{:08d}'.format(key, num)

    # try:
    #     key = nums[-1]
    # except IndexError:
    #     return fname

    # print('fname: {}, key: {}'.format(fname, key))
    return key


def processArguments(args, params):
    # arguments specified as 'arg_name=argv_val'
    no_of_args = len(args)
    for arg_id in range(no_of_args):
        arg_str = args[arg_id]
        if arg_str.startswith('--'):
            arg_str = arg_str[2:]
        arg = arg_str.split('=')
        if len(arg) != 2 or not arg[0] in params.keys():
            raise IOError('Invalid argument provided: {:s}'.format(args[arg_id]))

        if not arg[1] or not arg[0] or arg[1] == '#':
            continue

        if isinstance(params[arg[0]], (list, tuple)):

            if ':' in arg[1]:
                inclusive_start = inclusive_end = 1
                if arg[1].endswith(')'):
                    arg[1] = arg[1][:-1]
                    inclusive_end = 0
                if arg[1].startswith(')'):
                    arg[1] = arg[1][1:]
                    inclusive_start = 0

                _temp = [float(k) for k in arg[1].split(':')]
                if len(_temp) == 3:
                    _step = _temp[2]
                else:
                    _step = 1.0
                if inclusive_end:
                    _temp[1] += _step
                if not inclusive_start:
                    _temp[0] += _step
                arg_vals_parsed = list(np.arange(*_temp))
            else:
                if arg[1] and ',' not in arg[1]:
                    arg[1] = '{},'.format(arg[1])

                arg_vals = [x for x in arg[1].split(',') if x]
                arg_vals_parsed = []
                for _val in arg_vals:
                    try:
                        _val_parsed = int(_val)
                    except ValueError:
                        try:
                            _val_parsed = float(_val)
                        except ValueError:
                            _val_parsed = _val
                    if _val_parsed == '__n__':
                        _val_parsed = ''
                    arg_vals_parsed.append(_val_parsed)
            params[arg[0]] = type(params[arg[0]])(arg_vals_parsed)
        else:
            _val_parsed = arg[1]
            if _val_parsed == '__n__':
                _val_parsed = ''
            params[arg[0]] = type(params[arg[0]])(_val_parsed)


def resizeAR(src_img, width, height, return_factors=False, add_border=True):
    src_height, src_width, n_channels = src_img.shape
    src_aspect_ratio = float(src_width) / float(src_height)

    if width == 0 or height == 0:
        raise SystemError('Neither width nor height can be zero')

    aspect_ratio = float(width) / float(height)

    if add_border:
        if src_aspect_ratio == aspect_ratio:
            dst_width = src_width
            dst_height = src_height
            start_row = start_col = 0
        elif src_aspect_ratio > aspect_ratio:
            dst_width = src_width
            dst_height = int(src_width / aspect_ratio)
            start_row = int((dst_height - src_height) / 2.0)
            start_col = 0
        else:
            dst_height = src_height
            dst_width = int(src_height * aspect_ratio)
            start_col = int((dst_width - src_width) / 2.0)
            start_row = 0

        dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8)
        dst_img[start_row:start_row + src_height, start_col:start_col + src_width, :] = src_img
        dst_img = cv2.resize(dst_img, (width, height))
        if return_factors:
            resize_factor = float(height) / float(dst_height)
            return dst_img, resize_factor, start_row, start_col
        else:
            return dst_img
    else:
        if src_aspect_ratio < aspect_ratio:
            dst_width = width
            dst_height = int(dst_width / src_aspect_ratio)
        else:
            dst_height = height
            dst_width = int(dst_height * src_aspect_ratio)
        dst_img = cv2.resize(src_img, (dst_width, dst_height))
        start_row = start_col = 0
        if return_factors:
            resize_factor = float(src_height) / float(dst_height)
            return dst_img, resize_factor, start_row, start_col
        else:
            return dst_img


def drawBox(image, xmin, ymin, xmax, ymax, box_color=(0, 255, 0), label=None):
    # if cv2.__version__.startswith('3'):
    #     font_line_type = cv2.LINE_AA
    # else:
    #     font_line_type = cv2.CV_AA

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color)

    _bb = [xmin, ymin, xmax, ymax]
    if _bb[1] > 10:
        y_loc = int(_bb[1] - 5)
    else:
        y_loc = int(_bb[3] + 5)
    if label is not None:
        cv2.putText(image, label, (int(_bb[0] - 1), y_loc), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, box_color, 1, cv2.LINE_AA)

def isEmpty(src_dir):
    src_file_gen = [[os.path.join(dirpath, f) for f in filenames]
                    for (dirpath, dirnames, filenames) in os.walk(src_dir, followlinks=True)]
    src_files = [item for sublist in src_file_gen for item in sublist]
    if not src_files:
        return True
    return False

def removeEmptyFolders(path, removeRoot=True):
    'Function to remove empty folders'
    if not os.path.isdir(path):
        return

    # remove empty subfolders
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                removeEmptyFolders(fullpath)
    # if folder empty, delete it
    files = os.listdir(path)
    if len(files) == 0 and removeRoot:
        print("Removing empty folder:", path)
        os.rmdir(path)


def fix_bbox(bbox, fixed_ar, border, img_shape):
    xmin, ymin, xmax, ymax = bbox

    border_x = border_y = 0

    if fixed_ar:
        w, h = xmax - xmin, ymax - ymin
        src_ar = float(w) / float(h)
        if fixed_ar > src_ar:
            border_x = int((h * fixed_ar - w) / 2.0)
            border_y = 0
        else:
            border_y = int((w / fixed_ar - h) / 2.0)
            border_x = 0
    elif border:
        border_x = border_y = border

    img_h, img_w = img_shape[:2]

    xmin = max(xmin - border_x, 0)
    ymin = max(ymin - border_y, 0)
    xmax = min(xmax + border_x, img_w)
    ymax = min(ymax + border_y, img_h)

    return xmin, ymin, xmax, ymax


def map_mask_to_bbox(bbox, mask_patch, fixed_ar=0, border=0,
                     img_shape=None, mask_img=None):
    if mask_img is None:
        mask_img = np.zeros(img_shape, dtype=np.uint8)

    xmin, ymin, xmax, ymax = bbox
    box_w, box_h = xmax - xmin, ymax - ymin

    mask_h, mask_w = mask_patch.shape[:2]

    # print('dimensions of orig box: {} and mask: {}'.format(
    #     [box_h, box_w], [mask_h, mask_w]
    # ))

    # if fixed_ar or border:
    bbox = fix_bbox(bbox, fixed_ar, border, img_shape)

    xmin, ymin, xmax, ymax = bbox
    box_w, box_h = xmax - xmin, ymax - ymin

    # scale_factor_x, scale_factor_y = float(mask_w) / float(box_w), float(mask_h) / float(box_h)

    mask_patch = cv2.resize(mask_patch, (box_w, box_h))
    # print('curr_mask.shape[:2]: ', curr_mask.shape[:2])

    mask_h, mask_w = mask_patch.shape[:2]

    if mask_h != box_h or mask_w != box_w:
        raise SystemError('Weird mismatch between dimensions of box: {} and mask: {}'.format(
            [box_h, box_w], [mask_h, mask_w]
        ))

    # print('img_shape: {}'.format(img_shape))
    # print('bbox: {}'.format(bbox))
    # print('dimensions of box: {} and mask: {}'.format(
    #     [box_h, box_w], [mask_h, mask_w]
    # ))

    try:
        mask_img[ymin:ymax, xmin:xmax, :] = mask_patch
    except ValueError as e:
        print('bbox: {}'.format(bbox))
        print('Weird mismatch between dimensions of box: {} and mask: {}'.format(
            [box_h, box_w], [mask_h, mask_w]
        ))
        raise ValueError(e)

    return mask_img


def hist_match(bkg_img, src_img):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    src_img_matched = np.zeros_like(src_img, dtype=np.uint8)
    for ch_id in range(3):
        source = src_img[:, :, ch_id].squeeze()
        template = bkg_img[:, :, ch_id].squeeze()

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        src_img_matched[:, :, ch_id] = interp_t_values[bin_idx].reshape(oldshape)

    return src_img_matched


def hist_match2(bkg_img, src_img, nbr_bins=255):
    src_img_matched = np.zeros_like(src_img, dtype=np.uint8)
    for ch_id in range(3):
        imtint = bkg_img[:, :, ch_id].squeeze()
        imsrc = src_img[:, :, ch_id].squeeze()

        imhist, bins = np.histogram(imsrc.flatten(), nbr_bins, normed=True)
        tinthist, bins = np.histogram(imtint.flatten(), nbr_bins, normed=True)

        cdfsrc = imhist.cumsum()  # cumulative distribution function
        cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8)  # normalize

        cdftint = tinthist.cumsum()  # cumulative distribution function
        cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8)  # normalize

        im2 = np.interp(imsrc.flatten(), bins[:-1], cdfsrc)

        im3 = np.interp(im2, cdftint, bins[:-1])

        imres = im3.reshape((imsrc.shape[0], imsrc.shape[1]))
        src_img_matched[:, :, ch_id] = imres

    return src_img_matched


def get2DGaussianErrorFunctionArray(dst_width, dst_height, offset=4.0):
    dst_patch_mask = np.zeros((dst_height, dst_width), dtype=np.float64)

    c_x, c_y = dst_width / 2.0, dst_height / 2.0
    max_dist = c_x * c_x + c_y * c_y
    for x in range(dst_width):
        for y in range(dst_height):
            dist_x = c_x - x
            dist_y = c_y - y
            dist = dist_x * dist_x + dist_y * dist_y
            norm_dist = ((max_dist - dist) / max_dist) * offset - offset / 2.0
            dst_patch_mask[y, x] = (math.erf(norm_dist) + 1.0) / 2.0
    return dst_patch_mask


def get2DGaussianArray(dst_width, dst_height):
    x, y = np.meshgrid(np.linspace(-1, 1, dst_width), np.linspace(-1, 1, dst_height))
    x_diff = x
    y_diff = y
    d = np.sqrt(x_diff * x_diff + y_diff * y_diff)
    sigma, mu = 1.0, 0.0
    dst_patch_mask = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    dst_patch_mask /= np.max(dst_patch_mask)
    return dst_patch_mask


def get2DGaussianArray2(dst_width, dst_height):
    X = np.linspace(-1, 1, dst_width)
    Y = np.linspace(-1, 1, dst_height)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([0., 1.])
    Sigma = np.array([[1., 0.], [0., 1.]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2 * np.pi) ** n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

        return np.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    dst_patch_mask = multivariate_gaussian(pos, mu, Sigma)
    dst_patch_mask /= np.max(dst_patch_mask)

    return dst_patch_mask


def compareHist(img_1, img_2, method=0):
    hist_1 = cv2.calcHist([img_1], [0, 1, 2], None, [8, 8, 8],
                          [0, 256, 0, 256, 0, 256])
    hist_1 = cv2.normalize(hist_1, None).flatten()

    hist_2 = cv2.calcHist([img_2], [0, 1, 2], None, [8, 8, 8],
                          [0, 256, 0, 256, 0, 256])
    hist_2 = cv2.normalize(hist_2, None).flatten()

    OPENCV_METHODS = (
        ("Correlation", cv2.HISTCMP_CORREL),
        ("Chi-Squared", cv2.HISTCMP_CHISQR),
        ("Intersection", cv2.HISTCMP_INTERSECT),
        ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

    methodName, _method = OPENCV_METHODS[method]

    d = cv2.compareHist(hist_1, hist_2, _method)

    if methodName in ("Correlation", "Intersection"):
        d = 1.0 / (1.0 + d * d)
    return d


def addBorder(bbox, img, border_ratio, make_square, ):
    xmin, ymin, xmax, ymax = bbox
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

    img_h, img_w = img.shape[:2]

    width, height = xmax - xmin, ymax - ymin

    border_x = border_y = 0

    if border_ratio != 0:
        border_x, border_y = int(width * border_ratio), int(height * border_ratio)

    if make_square:
        if width < height:
            border_x += int((height - width) / 2.0 * (1. + border_ratio))
        elif width > height:
            border_y += int((width - height) / 2 * (1. + border_ratio))

    xmin -= border_x
    xmax += border_x
    ymin -= border_y
    ymax += border_y

    if ymax > img_h:
        ymax = img_h

    if xmax > img_w:
        xmax = img_w

    if ymin < 0:
        ymin = 0

    if xmin < 0:
        xmin = 0

    return xmin, ymin, xmax, ymax


def getIOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def readGT(gt_paths, seq_paths, seq_to_samples, class_names, combine_sequences, class_agnostic):
    print('Reading GT...')
    gt_data_dict = {}
    gt_counter_per_class = {}
    n_seq = len(gt_paths)
    all_gt_obj = []

    if combine_sequences:
        gt_data_dict['combined_seq'] = {}

    for seq_idx, gt_path in enumerate(gt_paths):
        seq_path = seq_paths[seq_idx]

        src_file_list = seq_to_samples[seq_idx]

        # valid_filenames = [os.path.basename(k) for k in src_file_list]

        # valid_filenames_set = set(valid_filenames)
        # duplicate_filenames = [k for k in valid_filenames if k not in valid_filenames_set]
        # if duplicate_filenames:
        #     raise IOError('{} :: {} amazingly annoying duplicate files: {}'.format(
        #         seq_name, len(duplicate_filenames), duplicate_filenames
        #     ))

        if not os.path.isfile(gt_path):
            raise IOError('GT file: {} does not exist'.format(gt_path))

        seq_gt_data_dict = {}

        sys.stdout.write('\rProcessing sequence {:d}/{:d}: {:s} '.format(
            seq_idx + 1, n_seq, gt_path))
        sys.stdout.flush()

        df_gt = pd.read_csv(gt_path)
        seq_gt_data_dict['csv'] = gt_path

        # if class_agnostic:
        #     _class_names = ['generic', ]
        # else:
        #     _class_names = class_names

        for _, row in df_gt.iterrows():
            filename = row['filename']

            file_path = os.path.join(seq_path, filename).replace('\\', '/')

            if file_path not in src_file_list:
                # print('Skipping {} :: {}'.format(gt_path, file_path))
                continue

            if file_path not in seq_gt_data_dict:
                seq_gt_data_dict[file_path] = []

            xmin = float(row['xmin'])
            ymin = float(row['ymin'])
            xmax = float(row['xmax'])
            ymax = float(row['ymax'])
            class_name = row['class']

            if class_name not in class_names:
                # print('{} : {} :: Skipping unknown class: {}'.format(gt_path, filename, class_name))
                continue

            if class_agnostic:
                _class_name = 'generic'
            else:
                _class_name = class_name

            # bbox = str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax)
            bbox = [xmin, ymin, xmax, ymax]

            if _class_name in gt_counter_per_class:
                gt_counter_per_class[_class_name] += 1
            else:
                gt_counter_per_class[_class_name] = 1

            _gt_obj = {
                "file_path": file_path,
                "class_name": _class_name,
                "bbox": bbox,
                "used": False,
                "matched": False
            }

            if combine_sequences:
                _gt_obj["seq_path"] = 'combined_seq'
            else:
                _gt_obj["seq_path"] = seq_path

            # if _gt_obj in seq_gt_data_dict[filename]:
            #     raise IOError('{} :: Extraordinarily amazingly annoying duplicate ground truth object: {}'.format(
            #         gt_path, _gt_obj))
            #
            # if _gt_obj in all_gt_obj:
            #     raise IOError('{} :: Extraordinarily amazingly annoying duplicate ground truth object: {}'.format(
            #         gt_path, _gt_obj))

            seq_gt_data_dict[file_path].append(
                _gt_obj
            )
            all_gt_obj += _gt_obj

        if combine_sequences:
            gt_data_dict['combined_seq'].update(seq_gt_data_dict)
        else:
            gt_data_dict[seq_path] = seq_gt_data_dict

    gt_data_dict['counter_per_class'] = gt_counter_per_class

    # print('gt_data_dict: ', gt_data_dict)
    print('gt_counter_per_class: ', gt_counter_per_class)

    return gt_data_dict


def readDetections(seq_path, csv_path, src_file_list, allow_seq_skipping, class_agnostic):
    # print('Reading Detections...')

    # if threshold > 0:
    #     print('Discarding detections with score<{}'.format(threshold))

    bounding_boxes = []
    # valid_filenames = [os.path.basename(k) for k in src_file_list]
    det_file_paths = []

    if not os.path.isfile(csv_path):
        msg = '\nDetection csv file: {} does not exist\n'.format(csv_path)
        if allow_seq_skipping:
            print(msg)
            # return bounding_boxes
        else:
            raise IOError(msg)
    else:
        # n_frames = len(src_file_list)

        # _start_t = time.time()
        df_det = pd.read_csv(csv_path)
        # _end_t = time.time()
        # print('read_csv time: {:.4f} '.format(float(_end_t - _start_t)))

        # _start_t = time.time()
        for _, row in df_det.iterrows():
            filename = row['filename']

            file_path = os.path.join(seq_path, filename).replace(os.sep, '/')
            #
            # if file_path not in src_file_list:
            #     print('src_file_list: {}'.format(pformat(src_file_list)))
            #     print('{} not in src_file_list'.format(file_path))
            #     sys.exit()
            #     # continue

            confidence = row['confidence']
            # if confidence < threshold:
            #     continue

            det_file_paths.append(file_path)

            xmin = float(row['xmin'])
            ymin = float(row['ymin'])
            xmax = float(row['xmax'])
            ymax = float(row['ymax'])
            class_name = row['class']

            if class_agnostic:
                class_name = 'generic'

            bounding_boxes.append(
                {"class": class_name,
                 "confidence": confidence,
                 "file_path": file_path,
                 "bbox": [xmin, ymin, xmax, ymax]}
            )

        # _end_t = time.time()
        # print('bounding_boxes time: {:.4f} '.format(float(_end_t - _start_t)))

    # _start_t = time.time()
    skipped_file_paths = [k for k in src_file_list if k not in det_file_paths]
    for file_path in skipped_file_paths:
        bounding_boxes.append(
            {"class": None,
             "confidence": None,
             "file_path": file_path,
             "bbox": None}
        )
    # _end_t = time.time()
    # print('skipped_file_paths time: {:.4f} '.format(float(_end_t - _start_t)))

    return bounding_boxes


def loadDetections(seq_idx, seq_to_samples, seq_paths, save_dir,
                   combine_sequences, csv_file_name, n_seq,
                   allow_seq_skipping, class_agnostic):
    src_file_list = seq_to_samples[seq_idx]

    if not src_file_list:
        return [], 0

    seq_path = seq_paths[seq_idx]
    seq_name = os.path.splitext(os.path.basename(seq_path))[0]
    n_frames = len(src_file_list)

    print('sequence {}/{}: {}: '.format(seq_idx + 1, n_seq, seq_name))

    if not csv_file_name:
        csv_file_name = os.path.join(save_dir, '{}.csv'.format(seq_name))

    if combine_sequences and not os.path.isfile(csv_file_name):
        # _start_t = time.time()
        _seq_paths = list(set([os.path.dirname(k) for k in src_file_list]))
        raw_det_data_dict = []
        for _seq_path in _seq_paths:
            _seq_name = os.path.splitext(os.path.basename(_seq_path))[0]
            _src_file_list = [k for k in src_file_list if os.path.dirname(k) == _seq_path]
            _csv_file_name = os.path.join(save_dir, '{}.csv'.format(_seq_name))
            print('Loading csv detections from {}'.format(_csv_file_name))
            _start_t = time.time()
            raw_det_data_dict += readDetections(_seq_path, _csv_file_name, _src_file_list,
                                                allow_seq_skipping, class_agnostic)
        # _end_t = time.time()
    else:
        print('Loading csv detections from {}'.format(csv_file_name))
        # _start_t = time.time()
        raw_det_data_dict = readDetections(seq_path, csv_file_name, src_file_list,
                                           allow_seq_skipping, class_agnostic)
        # _end_t = time.time()
    # try:
    #     fps = float(n_frames) / float(_end_t - _start_t)
    #     print('fps: {:.4f} '.format(fps))
    # except ZeroDivisionError:
    #     print('fps: too high')

    return raw_det_data_dict, n_frames
