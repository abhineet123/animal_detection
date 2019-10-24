import cv2
import numpy as np
import time
import sys
import pandas as pd

try:
    sys.path.remove('/home/abhineet/labelling_tool/object_detection_module')
    sys.path.remove('/home/abhineet/labelling_tool/object_detection_module/object_detection')
except:
    pass

try:
    sys.path.remove('/home/abhineet/617_w18/Assignment2/models/research/object_detection')
except:
    pass
    # print('could not remove /home/abhineet/617_w18/Assignment2/models/research/object_detection')

try:
    sys.path.remove('/home/abhineet/617_w18/Assignment2/models/research')
except:
    pass
    # print('could not remove /home/abhineet/617_w18/Assignment2/models/research')
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util
from utilities import processArguments, sortKey
import os

params = {
    'labels_path': 'data/wildlife_label_map.pbtxt',
    # 'src_path': 'images/train',
    # 'det_path': 'data/train.csv',
    # 'save_path': 'images/train_vis',
    'list_file_name': '',
    'src_path': '',
    'det_path': '',
    'save_path': '',
    'load_path': '',
    'n_classes': 7,
    'img_ext': 'jpg',
    'batch_size': 1,
    'show_img': 1,
    'save_ext': 'mkv',
    'n_frames': 0,
}

processArguments(sys.argv[1:], params)
list_file_name = params['list_file_name']
_src_path = params['src_path']
labels_path = params['labels_path']
_det_path = params['det_path']
n_classes = params['n_classes']
save_path = params['save_path']
load_path = params['load_path']
img_ext = params['img_ext']
batch_size = params['batch_size']
show_img = params['show_img']
save_ext = params['save_ext']
n_frames = params['n_frames']

video_exts = ['mkv', 'mp4', 'avi', 'wmv', 'mpeg']

if save_ext in video_exts:
    save_fmt = 2
else:
    save_fmt = 1

if list_file_name:
    print('Reading sequence names from {}'.format(list_file_name))
    if not os.path.exists(list_file_name):
        raise IOError('List file: {} does not exist'.format(list_file_name))
    src_paths = [os.path.join(_src_path, x.strip()) for x in open(list_file_name).readlines() if x.strip()]
    det_paths = [os.path.join(_det_path, os.path.splitext(os.path.basename(src_path))[0] + '.csv')
                 for src_path in src_paths]
else:
    src_paths = [_src_path]
    det_paths = [_det_path]


if not save_path:
    save_path = os.path.join(_det_path, 'videos')

save_path = os.path.abspath(save_path)
if not os.path.isdir(save_path):
    os.makedirs(save_path)

n_seq = len(src_paths)
pause_after_frame = 0
for seq_id in range(n_seq):
    src_path = src_paths[seq_id]
    seq_name = os.path.splitext(os.path.basename(src_path))[0]

    det_path = det_paths[seq_id]
    print('\nsequence {}/{}: {}'.format(seq_id + 1, n_seq, seq_name))

    print('Reading source images from: {}'.format(src_path))

    src_files = [k for k in os.listdir(src_path) if k.endswith('.{:s}'.format(img_ext))]
    total_frames = len(src_files)
    if total_frames <= 0:
        raise SystemError('No input frames found')
    print('total_frames: {}'.format(total_frames))
    src_files.sort(key=sortKey)

    label_map = label_map_util.load_labelmap(labels_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=n_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    class_dict = dict((v['name'], v['id']) for k, v in category_index.items())

    # print('categories: ', categories)
    # print('category_index: ', category_index)

    df = pd.read_csv(det_path)

    video_out = None
    if save_fmt == 1:
        save_path_full = os.path.join(save_path, seq_name)
        print('Saving output images to {}'.format(save_path))
        if not os.path.isdir(save_path_full):
            os.makedirs(save_path_full)
    elif save_fmt == 2:
        save_path_full = os.path.join(save_path, '{}.{}'.format(seq_name, save_ext))
        print('Saving output video to {}'.format(save_path_full))
        temp_img = cv2.imread(os.path.join(src_path, src_files[0]))
        height, width, _ = temp_img.shape
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_out = cv2.VideoWriter(save_path_full, fourcc, 20, (width, height))

    frame_id = 0
    csv_data = {}
    filenames = []
    # Collect instances of objects and remove from df
    print('Reading csv file...')
    while not df.empty:
        filename =  df.iloc[0].loc['filename']

        # Look for objects with similar filenames, group them, send them to csv_to_record function and remove from df
        multiple_instance = df.loc[df['filename'] == filename]
        # Total # of object instances in a file
        no_instances = len(multiple_instance.index)
        # Remove from df (avoids duplication)
        df = df.drop(multiple_instance.index[:no_instances])

        width = float(multiple_instance.iloc[0].loc['width'])
        height = float(multiple_instance.iloc[0].loc['height'])

        classes_text = []
        classes = []
        boxes = []
        scores = []

        for instance in range(0, len(multiple_instance.index)):
            xmin = multiple_instance.iloc[instance].loc['xmin']
            ymin = multiple_instance.iloc[instance].loc['ymin']
            xmax = multiple_instance.iloc[instance].loc['xmax']
            ymax = multiple_instance.iloc[instance].loc['ymax']
            class_name = multiple_instance.iloc[instance].loc['class']
            class_id = class_dict[class_name]

            boxes.append([ymin, xmin, ymax, xmax])
            classes.append(class_id)
            scores.append(1)

        boxes = np.asarray(boxes, dtype=np.float32)
        classes = np.asarray(classes)
        scores = np.asarray(scores)

        csv_data[filename] = [boxes, classes, scores]
        filenames.append(filename)

    print('Done')
    filenames.sort(key=sortKey)
    for filename in filenames:
        boxes, classes, scores = csv_data[filename]
        file_path = os.path.join(src_path, filename)
        if not os.path.exists(file_path):
            raise SystemError('Image file {} does not exist'.format(file_path))

        image = cv2.imread(file_path)
        # print('boxes', boxes)
        # print('scores', scores)
        # print('classes', classes)

        # print('boxes.shape', boxes.shape)
        # print('scores.shape', scores.shape)
        # print('classes.shape', classes.shape)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            boxes,
            classes.astype(np.int32),
            scores,
            # None,
            category_index,
            use_normalized_coordinates=False,
            line_thickness=4)

        if show_img:
            cv2.imshow(seq_name, image)
            k = cv2.waitKey(1 - pause_after_frame) & 0xFF
            if k == ord('q') or k == 27:
                break
            elif k == 32:
                pause_after_frame = 1 - pause_after_frame

        if save_fmt == 1:
            out_file_path = os.path.join(save_path, filename)
            cv2.imwrite(out_file_path, image)
        elif save_fmt == 2:
            video_out.write(image)

        frame_id += 1
        sys.stdout.write('\rDone {:d} frames '.format(frame_id))
        sys.stdout.flush()

        if n_frames > 0 and frame_id >= n_frames:
            break

    sys.stdout.write('\n')
    sys.stdout.flush()

    if save_fmt == 2:
        video_out.release()

    if show_img:
        cv2.destroyWindow(seq_name)
