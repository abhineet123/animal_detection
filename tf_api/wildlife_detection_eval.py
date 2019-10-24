# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[ ]:

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

# ## Env setup

# In[ ]:


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.

# In[ ]:


from utils import label_map_util

from utils import visualization_utils as vis_util

from utilities import processArguments, sortKey

params = {
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    'ckpt_path': 'evaluation_frozen_graphs/F-RCNN_Inceptionv2/frozen_inference_graph.pb',
    # List of the strings that is used to add correct label for each box.
    'labels_path': 'data/wildlife_label_map.pbtxt',
    'test_path': 'images/test',
    'save_path': 'images/test_vis',
    'load_path': '',
    'n_classes': 4,
    'img_ext': 'jpg',
    'batch_size': 1,
    'show_img': 0,
    'n_frames': 0,
}

processArguments(sys.argv[1:], params)
ckpt_path = params['ckpt_path']
labels_path = params['labels_path']
n_classes = params['n_classes']
test_path = params['test_path']
save_path = params['save_path']
load_path = params['load_path']
img_ext = params['img_ext']
batch_size = params['batch_size']
show_img = params['show_img']
n_frames = params['n_frames']

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ckpt_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=n_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


print('Reading test images from: {}'.format(test_path))

test_file_list = [k for k in os.listdir(test_path) if k.endswith('.{:s}'.format(img_ext))]
total_frames = len(test_file_list)
# print('file_list: {}'.format(file_list))
if total_frames <= 0:
    raise SystemError('No input frames found')
print('total_frames: {}'.format(total_frames))
test_file_list.sort(key=sortKey)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

if not os.path.isdir(save_path):
    os.makedirs(save_path)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        numbering = 0
        for img_fname in test_file_list:
            img_fname_no_ext = os.path.splitext(img_fname)[0]
            image_path = os.path.join(test_path, img_fname)

            image = Image.open(image_path)

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            new_image = Image.fromarray(image_np, 'RGB')
            new_image.save(os.path.join(save_path, img_fname))
            if show_img:
                # Keep figure shown until next click
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)
                plt.waitforbuttonpress()
                plt.close()
