# Sample script
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
try:
    import tensorflow as tf
except ImportError as e:
    print('tensorflow unavailable: {}'.format(e))
import pandas as pd
import os
import copy
import sys
import glob
import time
import random
# import glob
from random import shuffle
from collections import OrderedDict

from PIL import Image
from pprint import pprint, pformat
from utilities import sortKey
import math
import paramparse
import ast
import numpy as np

from subprocess import Popen, PIPE

try:
    from utils import dataset_util
except ImportError as e:
    print('dataset_util unavailable: {}'.format(e))


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def get_random_samples_with_min_diff(min_tx, max_tx, n_samples, min_diff, max_trials_per_sample=100):
    samples = []
    max_trials = max_trials_per_sample * n_samples
    n_trials = 0
    while len(samples) < n_samples:
        new_sample = random.randint(min_tx, max_tx - 1)
        if all(abs(new_sample - sample) >= min_diff for sample in samples):
            samples.append(new_sample)

        n_trials += 1
        assert n_trials <= max_trials, f"n_trials exceeds max_trials {max_trials} with min_diff {min_diff}"

    return samples


class Params(paramparse.CFG):
    """
    CSV to TFRecord Converter',
    :ivar allow_missing_annotations: 'allow_missing_annotations',
    :ivar allow_seq_skipping: 'allow_seq_skipping',
    :ivar annotations_list_path: 'annotations_list_path',
    :ivar check_images: 'check_images',
    :ivar class_names_path: 'Path to file containing class names',
    :ivar csv_paths: 'List of paths to csv annotations',
    :ivar enable_mask: 'enable_mask',
    :ivar even_sampling: 'use evenly spaced sampling (< 1 would draw samples from only a fraction of the sequence; <
    0 would invert the sampling)',
    :ivar exclude_loaded_samples: 'exclude_loaded_samples',
    :ivar fixed_ar: 'pad images to have fixed aspect ratio',
    :ivar inverted_sampling: 'invert samples defined by the remaining sampling parameters',
    :ivar load_samples: 'text files specifying the mapping from sequence paths to sampled files',
    :ivar load_samples_root: 'folder containing the sample files lists',
    :ivar min_size: 'min_size',
    :ivar n_frames: 'n_frames',
    :ivar only_sampling: 'only_sampling',
    :ivar output_path: 'Path to output TFRecord',
    :ivar random_sampling: 'enable random sampling',
    :ivar root_dir: 'Path to input files',
    :ivar samples_per_class: 'no. of samples to include per class; < 0 would sample from the end',
    :ivar samples_per_seq: 'no. of samples to include per sequence; < 0 would sample from the end; overrides
    samples_per_class',
    :ivar sampling_ratio: 'proportion of images to include in the tfrecord file',
    :ivar seq_paths: 'List of paths to image sequences',
    :ivar shuffle_files: 'shuffle files',
    :ivar write_annotations_list: 'write_annotations_list:1: yolov3_tf style 2: yolov3 (pt) style',
    :ivar write_tfrecord: 'write_tfrecord',

    """

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='csv_to_record')

        self.allow_missing_annotations = 0
        self.allow_seq_skipping = 1
        self.min_samples_per_seq = 0
        self.annotations_list_path = ''
        self.check_images = 0
        self.class_names_path = ''
        self.csv_paths = ''
        self.enable_mask = 0
        self.even_sampling = 0.0
        self.exclude_loaded_samples = 0
        self.fixed_ar = 0
        self.inverted_sampling = 0
        self.load_samples = []
        self.load_samples_root = ''
        self.min_size = 1
        self.n_frames = 0
        self.only_sampling = 0
        self.output_path = ''
        self.output_name = ''
        self.random_sampling = 0
        self.sampling_seq_len = 0
        self.sampling_min_diff = 0
        self.img_ext = 'jpg'

        self.root_dir = ''
        self.samples_per_class = 0
        self.samples_per_seq = 0
        self.sampling_ratio = 1.0
        self.auto_cap_ratio = 1.0
        self.sample_entire_seq = 0
        self.n_sample_permutations = 100

        self.seq_paths = ''
        self.shuffle_files = 0
        self.write_annotations_list = 2
        self.annotations_list_sep = ' '
        self.write_tfrecord = 1

        self.start_id = -1
        self.end_id = -1

        self.class_from_seq = 1
        self.class_from_file = 0


# image_names = []
def checkImage(fn):
    proc = Popen(['identify', '-verbose', fn], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err


# Function to convert csv to TFRecord
def csv_to_record(df_bboxes, file_path, class_dict, min_size,
                  enable_tfrecord=1, check_images=1):
    # Get metadata
    try:
        filename = df_bboxes.iloc[0].loc['filename']
    except IndexError:
        raise IOError('No annotations found for {}'.format(file_path))

    # image_name = linux_path(seq_path, filename)
    image_name = file_path

    if check_images:
        code, output, error = checkImage(image_name)
        if str(code) != "0" or str(error) != "":
            raise IOError("Damaged image found: {} :: {}".format(image_name, error))

    # if image_name in image_names:
    #     raise SystemError('Image {} already exists'.format(image_name))

    if filename != os.path.basename(file_path):
        raise IOError('DF filename: {} does not match the file_path: {}'.format(filename, file_path))

    # print('Adding {}'.format(image_name))
    # image_names.append(image_name)

    # print(image_name)

    width = float(df_bboxes.iloc[0].loc['width'])
    height = float(df_bboxes.iloc[0].loc['height'])

    xmins = []
    ymins = []

    xmaxs = []
    ymaxs = []

    classes_text = []
    class_ids = []

    # # Mapping name to id
    # class_dict = {
    #     'bear': 1,
    #     'moose': 2,
    #     'coyote': 3,
    #     'deer': 4,
    #     'person': 5
    # }
    n_df_bboxes = len(df_bboxes.index)

    bboxes = []
    tf_example = None

    for i in range(n_df_bboxes):
        df_bbox = df_bboxes.iloc[i]
        xmin = df_bbox.loc['xmin']
        ymin = df_bbox.loc['ymin']
        xmax = df_bbox.loc['xmax']
        ymax = df_bbox.loc['ymax']
        class_name = df_bbox.loc['class']

        try:
            class_id = class_dict[class_name]
        except KeyError:
            # print('\nIgnoring box with unknown class {} in image {} '.format(class_name, image_name))
            continue

        w = xmax - xmin
        h = ymax - ymin

        if w < 0 or h < 0:
            print('\nInvalid box in image {} with dimensions {} x {}\n'.format(image_name, w, h))
            xmin, xmax = xmax, xmin
            ymin, ymax = ymax, ymin
            w = xmax - xmin
            h = ymax - ymin
            if w < 0 or h < 0:
                raise IOError('\nSwapping corners still giving invalid dimensions {} x {}\n'.format(w, h))

        if w < min_size or h < min_size:
            print('\nIgnoring image {} with too small {} x {} box '.format(image_name, w, h))
            return None, None, None
            # continue

        def clamp(x, min_value=0.0, max_value=1.0):
            return max(min(x, max_value), min_value)

        bboxes.append((class_id, xmin, ymin, xmax, ymax))

        if not enable_tfrecord:
            continue

        xmin = clamp(xmin / width)
        xmax = clamp(xmax / width)
        ymin = clamp(ymin / height)
        ymax = clamp(ymax / height)

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)

        classes_text.append(class_name)
        class_ids.append(class_id)

    if enable_tfrecord:
        image_format = b'jpg'
        try:
            image_bytes = open(image_name).read()
        except UnicodeDecodeError as e:
            print('Error in reading {} :: {}'.format(image_name, e))
            sys.exit()

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(int(height)),
            'image/width': dataset_util.int64_feature(int(width)),
            'image/filename': dataset_util.bytes_feature(image_name),
            'image/source_id': dataset_util.bytes_feature(image_name),
            'image/encoded': dataset_util.bytes_feature(image_bytes),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            # 'image/object/class/label': dataset_util.int64_list_feature(class_ids)
        }))

    return tf_example, bboxes, (height, width)


def str_to_list(_str, _type=str, _sep=','):
    if _sep not in _str:
        _str += _sep
    k = list(map(_type, _str.split(_sep)))
    k = [_k for _k in k if _k]
    return k


def main():
    params = Params()
    paramparse.process(params)

    output_path = params.output_path
    output_name = params.output_name
    seq_paths = params.seq_paths
    csv_paths = params.csv_paths
    shuffle_files = params.shuffle_files
    n_frames = params.n_frames
    class_names_path = params.class_names_path
    min_size = params.min_size
    root_dir = params.root_dir
    allow_missing_annotations = params.allow_missing_annotations
    inverted_sampling = params.inverted_sampling
    load_samples = params.load_samples
    load_samples_root = params.load_samples_root
    exclude_loaded_samples = params.exclude_loaded_samples
    check_images = params.check_images

    auto_cap_ratio = params.auto_cap_ratio
    sampling_ratio = params.sampling_ratio
    random_sampling = params.random_sampling
    sampling_seq_len = params.sampling_seq_len
    sampling_min_diff = params.sampling_min_diff
    even_sampling = params.even_sampling
    samples_per_class = params.samples_per_class
    samples_per_seq = params.samples_per_seq
    sample_entire_seq = params.sample_entire_seq
    n_sample_permutations = params.n_sample_permutations
    min_samples_per_seq = params.min_samples_per_seq

    enable_mask = params.enable_mask
    only_sampling = params.only_sampling
    write_annotations_list = params.write_annotations_list
    annotations_list_sep = params.annotations_list_sep
    annotations_list_path = params.annotations_list_path
    write_tfrecord = params.write_tfrecord
    allow_seq_skipping = params.allow_seq_skipping

    start_id = params.start_id
    end_id = params.end_id

    class_from_seq = params.class_from_seq
    class_from_file = params.class_from_file

    sample_from_end = 0
    variable_sampling_ratio = 0

    if annotations_list_sep == '0':
        annotations_list_sep = ' '
    elif annotations_list_sep == '1':
        annotations_list_sep = '\t'

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
        if sample_entire_seq:
            variable_sampling_ratio = 3
        else:
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

    class_names = open(class_names_path, 'r').readlines()

    class_names = [x.strip() for x in class_names]
    class_dict = {x: i for (i, x) in enumerate(class_names)}

    if output_name:
        output_path = linux_path(output_path, output_name)

    # Writer object for TFRecord creation
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)
    output_name_no_ext = os.path.splitext(output_name)[0]
    print('output_path: ', output_path)
    print('output_dir: ', output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not annotations_list_path:
        annotations_list_path = linux_path(output_dir, output_name_no_ext + '.txt')

    if seq_paths:
        if seq_paths.endswith('.txt'):
            assert os.path.isfile(seq_paths), f"invalid seq_paths list: {seq_paths}"

        if os.path.isfile(seq_paths):
            seq_paths = [x.strip() for x in open(seq_paths).readlines() if x.strip()]
        else:
            seq_paths = seq_paths.split(',')
        if root_dir:
            seq_paths = [linux_path(root_dir, k) for k in seq_paths]

    elif root_dir:
        seq_paths = [linux_path(root_dir, name) for name in os.listdir(root_dir) if
                     os.path.isdir(linux_path(root_dir, name))]
        seq_paths.sort(key=sortKey)
    else:
        raise IOError('Either seq_paths or root_dir must be provided')

    seq_paths = [k.replace(os.sep, '/') for k in seq_paths]

    if csv_paths:
        with open(csv_paths) as f:
            csv_paths = f.readlines()
        csv_paths = [x.strip() for x in csv_paths if x.strip()]
    else:
        csv_paths = [linux_path(img_path, 'annotations.csv') for img_path in seq_paths]
    n_seq = len(seq_paths)
    if len(csv_paths) != n_seq:
        raise IOError('Mismatch between image and csv paths counts')

    if start_id < 0:
        start_id = 0
    if end_id < start_id:
        end_id = n_seq - 1

    assert end_id >= start_id, "end_id: {} is < start_id: {}".format(end_id, start_id)

    if start_id > 0 or end_id < n_seq - 1:
        print('curtailing sequences to between IDs {} and {}'.format(start_id, end_id))
        seq_paths = seq_paths[start_id:end_id + 1]
        csv_paths = csv_paths[start_id:end_id + 1]
        n_seq = len(seq_paths)

    if not load_samples:
        exclude_loaded_samples = 0

    seq_to_samples = {}

    if len(load_samples) == 1:
        if load_samples[0] == '1' or load_samples[0] == 1:
            load_samples = ['seq_to_samples.txt', ]
        elif load_samples[0] == '0' or load_samples[0] == 0:
            load_samples = []

    if load_samples:
        print('load_samples: {}'.format(pformat(load_samples)))
        if load_samples_root:
            load_samples = [linux_path(load_samples_root, k) for k in load_samples if k]
        print('Loading samples from : {}'.format(pformat(load_samples)))
        for _f in load_samples:
            if os.path.isdir(_f):
                _f = linux_path(_f, 'seq_to_samples.txt')
            with open(_f, 'r') as fid:
                curr_seq_to_samples = ast.literal_eval(fid.read())
                for _seq in curr_seq_to_samples:
                    if _seq in seq_to_samples:
                        seq_to_samples[_seq] += curr_seq_to_samples[_seq]
                    else:
                        seq_to_samples[_seq] = curr_seq_to_samples[_seq]

        if exclude_loaded_samples:
            print('Excluding loaded samples from source files')

    tfrecord_data = []
    total_samples = 0
    total_files = 0

    seq_to_src_files = OrderedDict()
    class_to_n_files = {_class: 0 for _class in class_names}
    class_to_n_seq = {_class: 0 for _class in class_names}
    class_to_n_files_per_seq = {_class: {} for _class in class_names}

    seq_to_n_files = {}

    def get_class_from_file(seq_path):
        with open(linux_path(seq_path, 'class_to_n_files.txt'), 'r') as fid:
            class_to_n_files = fid.readline()
        class_, _ = class_to_n_files.split('\t')
        return class_

    def get_class(seq_path):
        for class_ in class_names:
            if class_ in os.path.basename(seq_path):
                return class_
        raise IOError('No class found for {}'.format(seq_path))

    seq_to_sampling_ratio = {k: sampling_ratio for k in seq_paths}

    if class_from_file:
        seq_to_class = {k: get_class_from_file(k) for k in seq_paths}
    elif class_from_seq:
        seq_to_class = {k: get_class(k) for k in seq_paths}
    else:
        """assume that there is only one class"""
        print('assigning class {} to all sequences'.format(class_names[0]))
        seq_to_class = {k: class_names[0] for k in seq_paths}

    empty_seqs = []
    non_empty_seq_ids = []
    for idx, seq_path in enumerate(seq_paths):

        src_files = glob.glob(linux_path(seq_path, f'*.{params.img_ext}'), recursive=False)

        if exclude_loaded_samples:
            loaded_samples = seq_to_samples[seq_path]
            src_files = [k for k in src_files if k not in loaded_samples]

        if not src_files:
            if allow_seq_skipping:
                print('Skipping empty sequence: {}'.format(seq_path))
                empty_seqs.append(seq_path)
                continue
            else:
                raise IOError('Empty sequence found: {}'.format(seq_path))

        non_empty_seq_ids.append(idx)

        src_files.sort(key=sortKey)
        n_files = len(src_files)

        seq_to_n_files[seq_path] = n_files
        seq_to_src_files[seq_path] = src_files

        _class = seq_to_class[seq_path]
        class_to_n_files_per_seq[_class][seq_path] = n_files
        class_to_n_seq[_class] += 1
        class_to_n_files[_class] += n_files

        total_files += n_files

    print('class_to_n_files:')
    pprint(class_to_n_files)

    n_empty_seqs = len(empty_seqs)
    if n_empty_seqs:
        print('Found {} empty sequences:\n{}'.format(n_empty_seqs, pformat(empty_seqs)))
        seq_paths = [k for k in seq_paths if k not in empty_seqs]
        csv_paths = [csv_paths[i] for i in non_empty_seq_ids]

        n_seq = len(seq_paths)
        if len(csv_paths) != n_seq:
            raise IOError('Mismatch between image and csv paths counts')

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
    elif variable_sampling_ratio == 3:
        seq_to_sampling_ratio = {}
        n_sampled_seq = n_unsampled_seq = 0
        for _class in class_to_n_files:
            n_files_per_seq = class_to_n_files_per_seq[_class]
            class_seqs = list(n_files_per_seq.keys())
            _n_seq = class_to_n_seq[_class]

            min_diff = None
            min_diff_id = None
            for permute_id in range(n_sample_permutations):
                shuffle(class_seqs)
                _n_files_per_seq = [n_files_per_seq[_seq] for _seq in class_seqs]
                _n_files_per_seq_cum = np.cumsum(_n_files_per_seq)

                diff = np.abs(_n_files_per_seq_cum - samples_per_class)
                curr_min_diff_id = np.argmin(diff)
                curr_min_diff = diff[curr_min_diff_id]

                if min_diff is None or curr_min_diff < min_diff:
                    min_diff = curr_min_diff
                    min_diff_id = curr_min_diff_id
                    min_diff_class_seqs = copy.deepcopy(class_seqs)
                    min_diff_n_files_per_seq = copy.deepcopy(_n_files_per_seq)
                    min_diff_n_files_per_seq_cum = copy.deepcopy(_n_files_per_seq_cum)

                    if min_diff == 0:
                        break

            min_diff_seqs = ['{:5d} {} {:10d} {:10d}'.format(
                i, min_diff_class_seqs[i], min_diff_n_files_per_seq[i], min_diff_n_files_per_seq_cum[i])
                for i in range(_n_seq)]

            print(pformat(min_diff_seqs))
            print('_class: {}'.format(_class))
            print('min_diff: {}'.format(min_diff))
            print('min_diff_id: {}'.format(min_diff_id))

            n_sampled_seq += min_diff_id + 1
            n_unsampled_seq += _n_seq - min_diff_id - 1

            _seq_to_sampling_ratio = {
                min_diff_class_seqs[i]: 1 if i <= min_diff_id else 0 for i in range(_n_seq)
            }
            seq_to_sampling_ratio.update(_seq_to_sampling_ratio)

            print()
        print('n_sampled_seq: {}'.format(n_sampled_seq))
        print('n_unsampled_seq: {}'.format(n_unsampled_seq))
        print()

    class_to_n_samples = {_class: 0 for _class in class_names}
    all_sampled_files = []

    n_sampled_seq = n_unsampled_seq = 0

    valid_seq_to_samples = OrderedDict()

    for idx, seq_path in enumerate(seq_paths):
        src_files = seq_to_src_files[seq_path]
        n_files = seq_to_n_files[seq_path]
        sampling_ratio = seq_to_sampling_ratio[seq_path]

        if sampling_ratio > 1.0:
            msg = f'\nInvalid sampling_ratio: {sampling_ratio} for sequence: {seq_path} with {n_files} files\n'
            if not auto_cap_ratio:
                raise IOError(msg)
            print(msg)
            sampling_ratio = 1.0

        if load_samples and not exclude_loaded_samples:
            try:
                loaded_samples = seq_to_samples[seq_path]
            except KeyError:
                loaded_samples = []

            if inverted_sampling:
                loaded_samples = [k for k in src_files if k not in loaded_samples]

            n_loaded_samples = len(loaded_samples)

            if sampling_ratio == 1.0 or n_loaded_samples == 0:
                sampled_files = loaded_samples
            else:
                # n_samples = n_loaded_samples * sampling_ratio
                n_samples = round(n_loaded_samples * sampling_ratio)
                if n_samples < min_samples_per_seq:
                    sampling_ratio = float(min_samples_per_seq) / float(n_files)
                    n_samples = min_samples_per_seq
                    print('\nSetting sampling_ratio for {} to {} to be able to have {} samples\n'.format(
                        seq_path, sampling_ratio, min_samples_per_seq))

                if random_sampling:
                    if sampling_seq_len > 1:
                        assert n_samples % sampling_seq_len == 0, \
                            "n_samples must be divisible by sampling sequence length"
                        assert n_samples <= len(loaded_samples), \
                            "n_samples must be less than or equal to n_files"

                        n_sampled_seq = n_samples // sampling_seq_len

                        seq_starts = get_random_samples_with_min_diff(
                            0, len(loaded_samples) - n_sampled_seq, n_sampled_seq, sampling_seq_len)
                        seq_starts.sort()
                        seq_ends = [seq_start + sampling_seq_len for seq_start in seq_starts]
                        sampled_files = []
                        for seq_start, seq_end in zip(seq_starts, seq_ends, strict=True):
                            sampled_files += loaded_samples[seq_start:seq_end]
                    else:
                        sampled_files = random.sample(loaded_samples, n_samples)
                elif even_sampling:
                    if sampling_ratio > even_sampling:
                        raise SystemError('{} :: sampling_ratio: {} is greater than even_sampling: {}'.format(
                            seq_path, sampling_ratio, even_sampling))
                    sample_1_of_n = int(math.ceil(even_sampling / sampling_ratio))
                    end_file = int(n_files * even_sampling)

                    if sample_from_end:
                        sub_loaded_samples = loaded_samples[slice(-1, -end_file)]
                    else:
                        sub_loaded_samples = loaded_samples[slice(0, end_file)]

                    sampled_files = sub_loaded_samples[::sample_1_of_n]

                    more_samples_needed = n_samples - len(sampled_files)
                    if more_samples_needed > 0:
                        unsampled_files = [k for k in sub_loaded_samples if k not in sampled_files]
                        sampled_files += unsampled_files[:more_samples_needed]
                else:
                    if sample_from_end:
                        sampled_files = loaded_samples[-n_samples:]
                    else:
                        sampled_files = loaded_samples[:n_samples]
            n_samples = len(sampled_files)
        else:
            n_samples = int(round(n_files * sampling_ratio))
            if n_samples < min_samples_per_seq:
                sampling_ratio = float(min_samples_per_seq) / float(n_files)
                n_samples = min_samples_per_seq
                print('\nSetting sampling_ratio for {} to {} to be able to have {} samples\n'.format(
                    seq_path, sampling_ratio, min_samples_per_seq))

            if sampling_ratio != 1.0:
                if n_samples == 0:
                    sampled_files = []
                else:
                    if random_sampling:
                        n_src_files = len(src_files)

                        if sampling_seq_len > 1:
                            assert n_samples % sampling_seq_len == 0, (
                                "n_samples must be divisible by sampling sequence "
                                "length")
                            assert n_samples <= n_src_files, "n_samples must be less than or equal to n_src_files"

                            n_sampled_seq = n_samples // sampling_seq_len

                            if sampling_min_diff <= 0:
                                sampling_min_diff = sampling_seq_len

                            seq_starts = get_random_samples_with_min_diff(
                                min_tx=0, max_tx=n_src_files - sampling_seq_len,
                                n_samples=n_sampled_seq, min_diff=sampling_min_diff,
                                max_trials_per_sample=100, )

                            seq_starts.sort()
                            seq_ends = [seq_start + sampling_seq_len for seq_start in seq_starts]
                            sampled_files = []
                            for seq_start, seq_end in zip(seq_starts, seq_ends, strict=True):
                                assert seq_end <= n_src_files, "seq_end cannot be greater than n_src_files"
                                sampled_files += src_files[seq_start:seq_end]
                        else:
                            sampled_files = random.sample(src_files, n_samples)

                    elif even_sampling:
                        if sampling_ratio > even_sampling:
                            raise SystemError('{} :: sampling_ratio: {} is greater than even_sampling: {}'.format(
                                seq_path, sampling_ratio, even_sampling))
                        sample_1_of_n = int(math.ceil(even_sampling / sampling_ratio))
                        end_file = int(round(n_files * even_sampling))

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

        valid_seq_to_samples[seq_path] = sampled_files
        actual_samples = len(sampled_files)
        class_to_n_samples[seq_to_class[seq_path]] += actual_samples
        total_samples += actual_samples
        csv_path = csv_paths[idx]

        all_sampled_files += sampled_files

        if not sampled_files:
            msg = 'No sampled files found for {} with {} source files'.format(seq_path, n_files)
            if allow_seq_skipping:
                print('\n{}\n'.format(msg))
                n_unsampled_seq += 1
            else:
                raise IOError(msg)
        else:
            print('Processing sequence {}/{}: reading {}({})/{} images from {} and csv from {}'
                  ' (total images: {}/{})'.format(
                idx + 1, n_seq, actual_samples, n_samples, n_files, seq_path, csv_path, total_samples, total_files))
            n_sampled_seq += 1

        if only_sampling:
            continue

        df = pd.read_csv(csv_path)

        for file_path in sampled_files:
            filename = os.path.basename(file_path)

            if not os.path.isfile(file_path):
                raise IOError('Image file not found: {}'.format(file_path))

            try:
                df_multiple_instance = df.loc[df['filename'] == filename]
            except KeyError:
                if allow_missing_annotations:
                    print('\nNo annotations found for {}\n'.format(file_path))
                    continue
                else:
                    raise IOError('No annotations found for {}'.format(file_path))
            try:
                _ = df_multiple_instance.iloc[0].loc['filename']
            except IndexError:
                if allow_missing_annotations:
                    print('\nNo annotations found for {}\n'.format(file_path))
                    continue
                else:
                    raise IOError('No annotations found for {}'.format(file_path))

            num_instances = len(df_multiple_instance.index)

            # df = df.drop(df_multiple_instance.index[:num_instances])
            tfrecord_data.append((df_multiple_instance, file_path))

    seq_to_samples = valid_seq_to_samples

    print('')
    print('Total files: {}'.format(total_files))
    print('Total samples: {}'.format(total_samples))
    print('class_to_n_samples:')

    print('n_sampled_seq: {}'.format(n_sampled_seq))
    print('n_unsampled_seq: {}'.format(n_unsampled_seq))
    print('n_seq: {}'.format(len(seq_paths)))

    pprint(class_to_n_samples)

    out_dir = os.path.dirname(output_path)
    out_name = os.path.splitext(os.path.basename(output_path))[0]
    log_dir = linux_path(out_dir, out_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    print('\nWriting sampling log to: {}\n'.format(log_dir))
    import json
    with open(linux_path(log_dir, 'seq_to_src_files.txt'), 'w') as logFile:
        # pprint(seq_to_src_files, logFile)
        logFile.write(json.dumps(seq_to_src_files))
    with open(linux_path(log_dir, 'seq_to_samples.txt'), 'w') as logFile:
        # pprint(seq_to_samples, logFile)
        logFile.write(json.dumps(seq_to_samples))
    with open(linux_path(log_dir, 'class_to_n_samples.txt'), 'w') as logFile:
        # pprint(class_to_n_samples, logFile)
        logFile.write(json.dumps(class_to_n_samples))
    with open(linux_path(log_dir, 'class_to_n_files.txt'), 'w') as logFile:
        # pprint(class_to_n_files, logFile)
        logFile.write(json.dumps(class_to_n_files))
    with open(linux_path(log_dir, 'class_to_n_samples.txt'), 'w') as logFile:
        # pprint(class_to_n_samples, logFile)
        logFile.write(json.dumps(class_to_n_samples))
    with open(linux_path(log_dir, 'all_sampled_files.txt'), 'w') as logFile:
        logFile.write('\n'.join(all_sampled_files))
        # logFile.write(json.dumps(all_sampled_files))

    if not write_tfrecord and not write_annotations_list:
        only_sampling = 1

    if only_sampling:
        return

    if shuffle_files:
        print('Shuffling data...')
        shuffle(tfrecord_data)

    _n_frames = len(tfrecord_data)
    print('Total frames: {}'.format(_n_frames))

    print('class_names: ', class_names)
    print('class_dict: ', class_dict)

    if n_frames > 0 and n_frames < _n_frames:
        _n_frames = n_frames

    if write_tfrecord:
        print('Writing data for {} frames to file {}...'.format(_n_frames, output_path))
        writer = tf.python_io.TFRecordWriter(output_path)

    if write_annotations_list:
        annotations_list_fid = open(annotations_list_path, 'w')
        print('Writing annotations_list to {} in format {}'.format(annotations_list_path, write_annotations_list))

    if check_images:
        print('Image checking is enabled')

    out_diff = int(_n_frames * 0.05)
    n_ignored_images = 0
    file_path_id = 0
    start_time = time.time()
    for frame_id in range(_n_frames):
        df_multiple_instance, file_path = tfrecord_data[frame_id]
        # Send all object instances of a filename to become TFexample
        tf_example, bboxes, img_shape = csv_to_record(df_multiple_instance, file_path, class_dict, min_size,
                                                      write_tfrecord, check_images=check_images)
        if write_annotations_list and bboxes:
            annotations_list_fid.write('{}{}{}'.format(file_path_id, annotations_list_sep, file_path))
            img_h, img_w = img_shape
            file_path_id += 1
            for bbox in bboxes:
                class_id, xmin, ymin, xmax, ymax = bbox
                if write_annotations_list == 1:
                    annotations_list_fid.write('{}{}{}{}{}{}{}{}{}{}'.format(annotations_list_sep,
                                                                             class_id, annotations_list_sep,
                                                                             xmin, annotations_list_sep,
                                                                             ymin, annotations_list_sep,
                                                                             xmax, annotations_list_sep,
                                                                             ymax))
                else:
                    x_center, y_center, w, h = (xmin + xmax) / (2.0 * img_w), (
                            ymin + ymax) / (2.0 * img_h), (xmax - xmin) / img_w, (ymax - ymin) / img_h

                    x_center = min(max(0, x_center), 1)
                    y_center = min(max(0, y_center), 1)
                    w = min(max(0, w), 1)
                    h = min(max(0, h), 1)

                    annotations_list_fid.write('{}{}{}{}{}{}{}{}{}{}'.format(
                        annotations_list_sep,
                        class_id, annotations_list_sep,
                        x_center, annotations_list_sep,
                        y_center, annotations_list_sep,
                        w, annotations_list_sep,
                        h))
            annotations_list_fid.write('\n')

        if write_tfrecord:
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
            else:
                n_ignored_images += 1

        if frame_id % out_diff == 0:
            end_time = time.time()
            current_fps = 100.0 / (end_time - start_time)
            sys.stdout.write('\rDone {:d}/{:d} frames fps: {}'.format(
                frame_id, _n_frames, current_fps))
            sys.stdout.flush()
            start_time = time.time()

    print()

    if write_tfrecord:
        print('\nSuccessfully wrote {} image data to TFRecord file while ignoring {} images'.format(
            _n_frames - n_ignored_images, n_ignored_images))
        writer.close()
    if write_annotations_list:
        annotations_list_fid.close()


if __name__ == '__main__':
    main()
