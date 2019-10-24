# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_path=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re
import cv2
import sys
import argparse

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

sys.path.append('..')

from dataset_tools import tf_record_creation_util
from utils import dataset_util
from utils import label_map_util
from utilities import sortKey, resizeAR

# mask_pixel: dictionary containing class name and value for pixels belog to mask of each class
# change as per your classes and labeling
mask_pixel = {
    'animal': 255
}


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_score(bbox, img_w, img_h):
    bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    area = bbox_w * bbox_h
    ann_ratio = area / (img_w * img_h)

    ann_center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
    ann_center_bounds = (range(int(img_w / 4), int(img_w - img_w / 4)),
                         range(int(img_h / 4), int(img_h - img_h / 4)))
    ann_centered = ann_center[0] in ann_center_bounds[0] and ann_center[1] in ann_center_bounds[1]

    ann_br = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    ann_fully_contained = bbox[0] > 0 and bbox[1] > 0 and \
                          ann_br[0] < img_w and ann_br[1] < img_h

    return 1 if ann_ratio > 0.05 and ann_centered and ann_fully_contained else -1


def dict_to_tf_example(
        img_path,
        mask_path,
        label_map_dict,
        db_type,
):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      filename: name of the image
      mask_path: String path to PNG encoded mask.
      label_map_dict: A map from string label names to integers ids.
      img_path: String specifying subdirectory within the
        dataset directory holding the actual image data.


    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by filename is not a valid JPEG
    """
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    width = np.asarray(image).shape[1]
    height = np.asarray(image).shape[0]
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    with tf.gfile.GFile(mask_path, 'rb') as fid:
        encoded_mask_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_mask_png)
    mask = PIL.Image.open(encoded_png_io)
    mask_np = np.asarray(mask.convert('L'))
    if mask.format != 'PNG':
        raise ValueError('Mask format not PNG')

    # print('db_type: ', db_type)

    # filename = os.path.splitext(os.path.basename(img_path))

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    masks_remapped = []
    masks = []
    bboxes = []

    for class_name in list(mask_pixel.keys()):
        nonbackground_indices_x = np.any(mask_np == mask_pixel[class_name], axis=0)
        nonbackground_indices_y = np.any(mask_np == mask_pixel[class_name], axis=1)
        nonzero_x_indices = np.where(nonbackground_indices_x)
        nonzero_y_indices = np.where(nonbackground_indices_y)

        if np.asarray(nonzero_x_indices).shape[1] > 0 and np.asarray(nonzero_y_indices).shape[1] > 0:
            xmin = float(np.min(nonzero_x_indices))
            xmax = float(np.max(nonzero_x_indices))
            ymin = float(np.min(nonzero_y_indices))
            ymax = float(np.max(nonzero_y_indices))
            # print(filename, 'bounding box for', class_name, xmin, xmax, ymin, ymax)

            xmins.append(xmin / width)
            ymins.append(ymin / height)
            xmaxs.append(xmax / width)
            ymaxs.append(ymax / height)

            bboxes.append([xmax, ymax, xmin, ymin])

            classes_text.append(class_name.encode('utf8'))
            classes.append(label_map_dict[class_name])

            mask_remapped = (mask_np == mask_pixel[class_name]).astype(np.uint8)
            masks_remapped.append(mask_remapped)

            masks.append(mask_np)

    if db_type == 1:

        n_objs = len(masks)
        # print('n_objs: ', n_objs)

        for i in range(n_objs):
            mask = masks[i]
            mask = cv2.resize(mask, (224, 224))
            # -1 where mask is zero, 1 otherwise
            mask = np.where(mask == 0, -1, 1).astype(np.int8)
            # cv2.imshow('mask', mask)

            bbox = bboxes[i]
            # score = get_score(bbox, width, height)
            score = 1
            print('mask: ', mask.shape)
            print('score: ', score)

            k = cv2.waitKey(0)
            if k == 27:
                sys.exit()

            feature_dict = {
                'score': _int_feature(score),
                'image': _bytes_feature(img_path.encode()),
                'mask': _bytes_feature(mask.tostring())
            }
    else:
        feature_dict = {
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                img_path.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                img_path.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }

        encoded_mask_png_list = []
        for mask in masks_remapped:
            img = PIL.Image.fromarray(mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            encoded_mask_png_list.append(output.getvalue())
        feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png_list))

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     examples,
                     db_type
                     ):
    """Creates a TFRecord file from examples.

    Args:
      output_filename: Path to where output file is saved.
      num_shards: Number of shards for output file.
      label_map_dict: The label map dictionary.
      annotations_dir: Directory where annotation files are stored.
      image_dir: Directory where image files are stored.
      examples: Examples to parse and save to tf record.
    """
    print('Writing tfrecod for {} images to {}'.format(len(examples), output_filename))

    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 10 == 0:
            logging.info('On image %d of %d', idx, len(examples))

        image_path, mask_path = example
        try:
            tf_example = dict_to_tf_example(image_path,
                                            mask_path,
                                            label_map_dict,
                                            db_type)
            if tf_example:
                writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', mask_path)
    writer.close()


def main(_):
    # flags = tf.app.flags
    # flags.DEFINE_string('seq_path', '', 'Path to dataset root directory to .')
    # flags.DEFINE_string('output_path', '', 'Path to directory to output TFRecords.')
    # flags.DEFINE_string('image_dir', 'images', 'Name of the directory contatining images')
    # flags.DEFINE_string('mask_dir', 'labels', 'Name of the directory contatining Annotations')
    # flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
    # flags.DEFINE_integer('num_shards', 1, 'Number of TFRecord shards')
    # flags.DEFINE_integer('db_type', 0, 'db_type')
    # FLAGS = flags.FLAGS

    parser = argparse.ArgumentParser(description="Masks to TFRecord Converter")

    parser.add_argument('--seq_paths', type=str, default='',
                        help='List of paths to image sequences')
    parser.add_argument('--seq_postfix', type=str, default='',
                        help='seq_postfix')
    parser.add_argument('--root_dir', type=str, default='',
                        help='Optional root directory containing all sequences')

    parser.add_argument('--output_path', type=str, default='',
                        help='Path to output TFRecord')
    parser.add_argument('--image_dir', type=str, default='images',
                        help='Name of the subdirectory containing images')
    parser.add_argument('--mask_dir', type=str, default='labels',
                        help='Name of the subdirectory containing mask annotations')
    parser.add_argument('--label_map_path', type=str, default='',
                        help='Path to label map proto')
    parser.add_argument('--img_ext', type=str, default='jpg',
                        help='img_ext')
    parser.add_argument('--mask_ext', type=str, default='png',
                        help='mask_ext')

    parser.add_argument('--num_shards', type=int, default=1,
                        help='Number of TFRecord shards')
    parser.add_argument('--db_type', type=int, default=0,
                        help='db_type')

    parser.add_argument('--shuffle_files', type=int, default=1,
                        help='shuffle files')
    parser.add_argument('--n_frames', type=int, default=0,
                        help='n_frames')
    parser.add_argument('--min_size', type=int, default=1,
                        help='min_size')
    parser.add_argument('--fixed_ar', type=int, default=0,
                        help='pad images to have fixed aspect ratio')

    parser.add_argument('--sampling_ratio', type=float, default=1.0,
                        help='proportion of images to include in the tfrecord file')
    parser.add_argument('--random_sampling', type=int, default=0,
                        help='enable random sampling')
    parser.add_argument('--inverted_sampling', type=int, default=0,
                        help='invert samples defined by the remaining sampling parameters')
    parser.add_argument('--even_sampling', type=float, default=0.0,
                        help='use evenly spaced sampling (< 1 would draw samples from '
                             'only a fraction of the sequence; < 0 would invert the sampling)')
    parser.add_argument('--samples_per_class', type=int, default=0,
                        help='no. of samples to include per class; < 0 would sample from the end')

    parser.add_argument('--allow_missing_annotations', type=int, default=0,
                        help='allow_missing_annotations')

    flags = parser.parse_args()

    seq_paths = flags.seq_paths
    seq_postfix = flags.seq_postfix
    root_dir = flags.root_dir
    output_path = flags.output_path
    db_type = flags.db_type
    image_dir = flags.image_dir
    mask_dir = flags.mask_dir
    label_map_path = flags.label_map_path
    num_shards = flags.num_shards
    img_ext = flags.img_ext
    mask_ext = flags.mask_ext

    if not seq_paths:
        seq_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                     os.path.isdir(os.path.join(root_dir, name))]
        seq_paths.sort(key=sortKey)
    else:
        if os.path.isfile(seq_paths):
            seq_paths = [x.strip() for x in open(seq_paths).readlines()]
        else:
            seq_paths = seq_paths.split(',')

        if seq_postfix:
            seq_paths = ['{}_{}'.format(k, seq_postfix) for k in seq_paths]

        if root_dir:
            seq_paths = [os.path.join(root_dir, k) for k in seq_paths]

    n_seq = len(seq_paths)
    print('Creating tfrecord from {} sequences: {}'.format(n_seq, seq_paths))
    examples = []
    for seq_path in seq_paths:
        image_path = os.path.join(seq_path, image_dir)
        mask_path = os.path.join(seq_path, mask_dir)

        _examples_list = [os.path.splitext(k)[0] for k in os.listdir(image_path)
                          if k.endswith('.{:s}'.format(img_ext))]

        # _examples_list = os.listdir(image_dir)
        # for el in _examples_list:
        #     if el[-3:] != 'jpg':
        #         del _examples_list[_examples_list.index(el)]
        # for el in _examples_list:
        #     _examples_list[_examples_list.index(el)] = el[0:-4]

        examples += [(os.path.join(image_path, '{}.{}'.format(k, img_ext)),
                      os.path.join(mask_path, '{}.{}'.format(k, mask_ext)))
                     for k in _examples_list]

    n_files = len(examples)
    print('Found {} files'.format(n_files))

    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    create_tf_record(output_path,
                     num_shards,
                     label_map_dict,
                     examples,
                     db_type)


if __name__ == '__main__':
    tf.app.run()
