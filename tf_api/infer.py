import sys

try:
    sys.path.remove('/home/abhineet/labelling_tool/object_detection_module')
    sys.path.remove('/home/abhineet/labelling_tool/object_detection_module/object_detection')
except:
    pass
# sys.path.append('./')
# sys.path.append('./object_detection/')
sys.path.append('./models/research/object_detection/')
sys.path.append('./models/research/')
import cv2
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import utilities

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

# from utils import visualization_utils as vis_util


params = {
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    'ckpt_path': './models/research/inference_graph_faster_rcnn_nas_coco_2018_01_28_64431.pb/frozen_inference_graph.pb',
    # List of the strings that is used to add correct label for each box.
    'labels_path': './mnistdd_label_map.pbtxt',
    'dataset_path': 'dataset',
    'src_path': 'videos/human',
    'save_path': '',
    'load_path': '',
    'x_fname': 'valid_X.npy',
    'y_fname': 'valid_Y.npy',
    'bboxes_fname': 'valid_bboxes.npy',
    'n_classes': 10,
    'img_ext': 'png',
    'batch_size': 128,
    'n_frames': 0,
    'show_img': 0,
    'save_fmt': 1,
    'codec': 'H264',
}

utilities.processArguments(sys.argv[1:], params)
ckpt_path = params['ckpt_path']
labels_path = params['labels_path']
n_classes = params['n_classes']
dataset_path = params['dataset_path']
src_path = params['src_path']
save_path = params['save_path']
load_path = params['load_path']
x_fname = params['x_fname']
y_fname = params['y_fname']
bboxes_fname = params['bboxes_fname']
img_ext = params['img_ext']
batch_size = params['batch_size']
show_img = params['show_img']
save_fmt = params['save_fmt']
codec = params['codec']
n_frames = params['n_frames']

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ckpt_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=n_classes,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.gif')
src_files = [k for k in os.listdir(src_path) if os.path.splitext(k.lower())[1] in img_exts]
src_files.sort()

if n_frames <= 0:
    n_frames = len(src_files)

if n_frames <= 0:
    raise SystemError('No input frames found')
print('n_frames: ', n_frames)
print('batch_size: ', batch_size)

img_id = 0
pause_after_frame = 0

video_out = None
if save_fmt == 1:
    print('Saving output images to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
elif save_fmt == 2:
    print('Saving output video to {}'.format(save_path))
    temp_img = cv2.imread(os.path.join(src_path, src_files[0]))
    height, width, _ = temp_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'{}'.format(codec))
    video_out = cv2.VideoWriter(save_path, fourcc, 20, (width, height))

with detection_graph.as_default():
    with tf.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        while img_id < n_frames:
            src_file = src_files[img_id]

            image_path = os.path.join(src_path, src_file)

            # image = Image.open(image_path)
            # image.convert('RGB')
            # print('image.shape', np.array(image.getdata()).shape)
            # image_np = load_image_into_numpy_array(image)

            # image_np = x_test[i, :].squeeze().reshape((64, 64))
            # image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

            image_np = cv2.imread(image_path)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            img_id += 1

            height, width, _ = image_np.shape

            # Actual detection.

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            detection_classes = output_dict[
                'detection_classes'][0].astype(np.uint8)
            detection_boxes = output_dict['detection_boxes'][0]
            detection_scores = output_dict['detection_scores'][0]
            num_detections = output_dict['num_detections'][0]
            # if num_detections != 2:
            #     raise SystemError('Invalid num_detections: {}'.format(num_detections))

            # print('detection_boxes: ', detection_boxes)
            # print('detection_classes: ', detection_classes)
            # print('detection_scores: ', detection_scores)
            # print('num_detections: ', num_detections)

            detection_boxes_img = []
            for det_id in range(int(num_detections)):
                ymin, xmin, ymax, xmax = detection_boxes[det_id]
                detection_boxes_img.append([ymin * height, xmin * width, ymax * height, xmax * width])

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.asarray(detection_boxes_img),
                np.asarray(detection_classes),
                np.asarray(detection_scores),
                # None,
                category_index,
                use_normalized_coordinates=False,
                line_thickness=4)

            if show_img:
                cv2.imshow('frame', image_np)
                k = cv2.waitKey(1 - pause_after_frame) & 0xFF
                if k == ord('q') or k == 27:
                    break
                elif k == 32:
                    pause_after_frame = 1 - pause_after_frame

            if save_fmt == 1:
                out_file_path = os.path.join(save_path, src_file)
                cv2.imwrite(out_file_path, image_np)
            elif save_fmt == 2:
                video_out.write(image_np)

            sys.stdout.write('Done {:d}/{:d} images\n'.format(img_id, n_frames))
            sys.stdout.flush()

            # Visualization of the results of a detection.
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     image_np,
            #     output_dict['detection_boxes'],
            #     output_dict['detection_classes'],
            #     output_dict['detection_scores'],
            #     category_index,
            #     instance_masks=output_dict.get('detection_masks'),
            #     use_normalized_coordinates=True,
            #     line_thickness=8)

        sys.stdout.write('\n')
        sys.stdout.flush()

if save_fmt == 2:
    video_out.release()

if show_img:
    cv2.destroyAllWindows()
