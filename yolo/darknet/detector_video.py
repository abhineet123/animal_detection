import cv2
import numpy as np
import pandas as pd
import time
import sys, os

try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []

# print('user_paths: ', user_paths)

# sys.path.append(os.path.join(os.getcwd(), 'python/'))
sys.path.append(os.path.join(os.getcwd(), '../../'))
import python.darknet as dn
from tf_api.utilities import processArguments, sortKey
from tf_api.utils import visualization_utils as vis_util

params = {
    'weights_path': 'evaluation_frozen_graphs/F-RCNN_Inceptionv2/frozen_inference_graph.pb',
    'cfg_path': 'data/wildlife_label_map.pbtxt',
    'meta_path': 'data/wildlife_label_map.pbtxt',
    'file_name': '',
    'list_file_name': '',
    'root_dir': '',
    'save_dir': '',
    'save_file_name': '',
    'csv_file_name': '',
    'map_folder': '',
    'load_path': '',
    'n_classes': 4,
    'n_frames': 0,
    'img_ext': 'png',
    'batch_size': 1,
    'show_img': 0,
    'save_video': 0,
    'codec': 'H264',
    'fps': 20,
    'thresh': 0.5,
    'hier_thresh': 0.5,
    'nms': 0.45,
    'gpu': 0,
}

processArguments(sys.argv[1:], params)
weights_path = params['weights_path']
cfg_path = params['cfg_path']
meta_path = params['meta_path']
n_classes = params['n_classes']
file_name = params['file_name']
list_file_name = params['list_file_name']
root_dir = params['root_dir']
save_dir = params['save_dir']
save_file_name = params['save_file_name']
csv_file_name = params['csv_file_name']
map_folder = params['map_folder']
load_path = params['load_path']
img_ext = params['img_ext']
batch_size = params['batch_size']
show_img = params['show_img']
save_video = params['save_video']
n_frames = params['n_frames']
codec = params['codec']
fps = params['fps']
thresh = params['thresh']
hier_thresh = params['hier_thresh']
nms = params['nms']
gpu = params['gpu']

dn.set_gpu(gpu)
cfg_path = cfg_path.encode('utf-8')
weights_path = weights_path.encode('utf-8')
net = dn.load_net(cfg_path, weights_path, 0)

meta_data = open(meta_path, 'r').readlines()
meta_path = meta_path.encode('utf-8')
meta = dn.load_meta(meta_path)

class_names_path = None
for line in meta_data:
    if line.startswith('names'):
        class_names_path = line.split('=')[1].strip()
        break
if class_names_path is None:
    raise SystemError('class_names_path not found in meta file: {}'.format(meta_path))
class_names = open(class_names_path, 'r').readlines()
class_names = {x.strip(): i + 1 for (i, x) in enumerate(class_names)}
category_index = {v: {'name': k, 'id': v} for k, v in class_names.items()}
print('class_names: ', class_names)
print('category_index: ', category_index)

if list_file_name:
    if not os.path.exists(list_file_name):
        raise IOError('List file: {} does not exist'.format(list_file_name))
    file_list = [x.strip() for x in open(list_file_name).readlines()]
    if root_dir:
        file_list = [os.path.join(root_dir, x) for x in file_list]
elif root_dir:
    file_list = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                 os.path.isdir(os.path.join(root_dir, name))]
    file_list.sort(key=sortKey)
else:
    if not file_name:
        raise IOError('Either list file or a single sequence file must be provided')
    file_list = [file_name]

if not save_dir:
    save_dir = 'results'

n_seq = len(file_list)
print('Running over {} sequences'.format(n_seq))
avg_fps_list = np.zeros((n_seq,))

for file_idx, file_name in enumerate(file_list):
    seq_name = os.path.basename(file_name)
    print('sequence {}/{}: {}: '.format(file_idx + 1, n_seq, seq_name))

    if os.path.isdir(file_name):
        cap = cv2.VideoCapture(os.path.join(file_name, 'image%06d.jpg'))
    else:
        seq_name = os.path.splitext(seq_name)[0]
        if not os.path.exists(file_name):
            raise SystemError('Source video file: {} does not exist'.format(file_name))
        cap = cv2.VideoCapture(file_name)

    if not cap:
        raise SystemError('Source video file: {} could not be opened'.format(file_name))

    if not save_file_name:
        save_file_name = os.path.join(save_dir, '{}.mkv'.format(seq_name))
    _save_dir = os.path.dirname(save_file_name)
    if not os.path.isdir(_save_dir):
        os.makedirs(_save_dir)

    if not csv_file_name:
        csv_file_name = os.path.join(save_dir, '{}.csv'.format(seq_name))
    csv_save_dir = os.path.dirname(csv_file_name)
    if not os.path.isdir(csv_save_dir):
        os.makedirs(csv_save_dir)
    print('Saving csv detections to {}'.format(csv_file_name))

    if not map_folder:
        map_folder = os.path.join(save_dir, '{}_mAP'.format(seq_name))

    if not os.path.isdir(map_folder):
        os.makedirs(map_folder)
    print('Saving mAP detections to {}'.format(map_folder))

    width = cap.get(3)
    width = int(width)
    height = cap.get(4)
    height = int(height)

    print('width: ', width)
    print('height: ', height)

    video_out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # save_file_name = 'grizzly_bear_detection.avi'
        video_out = cv2.VideoWriter(save_file_name, fourcc, fps, (width, height))
        if not video_out:
            raise SystemError('Output video file: {} could not be opened'.format(save_file_name))
        print('Saving visualizations to {}'.format(save_file_name))

    if show_img:
        cv2.namedWindow(seq_name)
        cv2.setWindowProperty(seq_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_id = 0
    avg_fps = 0
    csv_raw = []
    while True:
        ret, image_np = cap.read()
        if not ret:
            print('Frame {} could not be read'.format(frame_id + 1))
            break

        # Actual detection
        _start_t = time.time()
        detections = dn.detect(net, meta, image_np)
        _end_t = time.time()

        n_detections = len(detections)

        fps = 1.0 / float(_end_t - _start_t)

        # print('detections: ', detections)
        filename = 'image{:06d}.jpg'.format(frame_id + 1)
        map_out_fname = os.path.join(map_folder, 'image{:06d}.txt'.format(frame_id + 1))
        map_file = open(map_out_fname, 'w')

        classes_text = []
        classes = []
        boxes = []
        scores = []

        for _class, _score, _box in detections:
            x, y, w, h = _box

            xmin = x - w / 2.0
            ymin = y - h / 2.0

            xmax = xmin + w
            ymax = ymin + h

            _class = _class.decode("utf-8")

            boxes.append([ymin, xmin, ymax, xmax])

            if not _class in class_names.keys():
                raise SystemError('Invalid _class: {}'.format(_class))

            classes.append(class_names[_class])
            scores.append(_score)

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.asarray(boxes),
                np.asarray(classes, dtype=np.int32),
                np.asarray(scores),
                # None,
                category_index,
                use_normalized_coordinates=False,
                line_thickness=4)

            raw_data = {
                'filename': filename,
                'width': width,
                'height': height,
                'class': _class,
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax)
            }

            csv_raw.append(raw_data)

            map_file.write('{:s} {:f} {:d} {:d} {:d} {:d}\n'.format(_class, _score, int(xmin), int(ymin),
                                                                    int(xmax), int(ymax)))

        if save_video:
            video_out.write(image_np)

        if show_img:
            cv2.imshow(seq_name, image_np)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break
        map_file.close()

        frame_id += 1
        avg_fps += (fps - avg_fps) / float(frame_id)

        sys.stdout.write('\rDone {:d} frames fps: {:.4f} avg_fps: {:.4f} n_detections: {:d}'.format(
            frame_id, fps, avg_fps, n_detections))
        sys.stdout.flush()

        if n_frames > 0 and frame_id >= n_frames:
            break

    sys.stdout.write('\n')
    sys.stdout.flush()

    cap.release()
    if save_video:
        video_out.release()

    df = pd.DataFrame(csv_raw)
    df.to_csv(csv_file_name)

    if show_img:
        cv2.destroyAllWindows()
    print('avg_fps: ', avg_fps)
    avg_fps_list[file_idx] = avg_fps

    save_file_name = ''
    csv_file_name = ''
    map_folder = ''

overall_avg_fps = np.mean(avg_fps_list)
print('overall_avg_fps: ', overall_avg_fps)
