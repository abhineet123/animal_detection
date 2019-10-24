import glob
import os, sys
import shutil
import pickle
import time

import argparse
import cv2
import map_utils as utils
import matplotlib.pyplot as plt

sys.path.append('..')
from tf_api.utilities import resizeAR, sortKey

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
parser.add_argument('--labels_path', type=str, help="text file containing class labels")
parser.add_argument('--gt_path_list_file', type=str, help="file containing list of GT folders", default='')
parser.add_argument('--det_path_list_file', type=str, help="file containing list of detection folders")
parser.add_argument('--img_path_list_file', type=str, help="file containing list of image folders")
parser.add_argument('--img_ext', type=str, help="image extension", default='jpg')
parser.add_argument('--img_root_dir', type=str, help="folder to save the animation result in", default='')
parser.add_argument('--gt_root_dir', type=str, help="folder to save the animation result in", default='')
parser.add_argument('--save_dir', type=str, help="folder to save the animation result in", default='')
parser.add_argument('--save_file_name', type=str, help="file to save the animation result in", default='mAP.mkv')
parser.add_argument('--save_file_res', type=str, help="resolution of the saved video", default='1280x720')
parser.add_argument('--pkl_files_path', type=str, help="location of temporary json files", default='tmp_files')
parser.add_argument('--out_fname', type=str, help="out_fname", default='')
parser.add_argument('--codec', type=str, help="file to save the animation result in", default='H264')
parser.add_argument('--fps', type=int, help="FPS", default=30)
parser.add_argument('--delete_tmp_files', type=int, default=0)
parser.add_argument('--show_animation', type=int, default=1)
parser.add_argument('--show_only_tp', type=int, default=0)
parser.add_argument('--show_text', type=int, default=1)
parser.add_argument('--show_stats', type=int, default=1)
parser.add_argument('--show_gt', type=int, default=1)
parser.add_argument('--draw_plot', type=int, default=0)
parser.add_argument('--iou_thresh', type=float, help="iou_thresh", default=0.5)

args = parser.parse_args()

print('args.gt_path_list_file', args.gt_path_list_file)
print('args.det_path_list_file', args.det_path_list_file)
print('args.img_path_list_file', args.img_path_list_file)
print('args.labels_path', args.labels_path)

save_dir = args.save_dir
save_file_name = args.save_file_name
codec = args.codec
fps = args.fps
labels_path = args.labels_path
iou_thresh = args.iou_thresh
save_file_res = args.save_file_res
pkl_files_path = args.pkl_files_path
delete_tmp_files = args.delete_tmp_files
show_animation = args.show_animation
show_only_tp = args.show_only_tp
show_text = args.show_text
show_stats = args.show_stats
show_gt = args.show_gt
draw_plot = args.draw_plot
out_fname = args.out_fname

gt_path_list_file = args.gt_path_list_file
img_path_list_file = args.img_path_list_file
det_path_list_file = args.det_path_list_file

img_root_dir = args.img_root_dir
gt_root_dir = args.gt_root_dir

if not gt_root_dir:
    gt_root_dir = os.path.join(img_root_dir, 'test', 'labels')

save_w, save_h = [int(x) for x in save_file_res.split('x')]

if not save_dir:
    save_dir = 'results'

_pause = 0
# if there are no classes to ignore then replace None by empty list
if args.ignore is None:
    args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
    specific_iou_flagged = True

# if there are no images then no animation can be shown
# img_path = args.img_path
img_ext = args.img_ext
# if not os.path.exists(img_path):
#     raise SystemError('image file folder does not exist')

# src_file_list = [k for k in os.listdir(img_path) if k.endswith('.{:s}'.format(img_ext))]

# for dirpath, dirnames, files in os.walk(img_path):
#     if not files:
#         args.no_animation = True
#         raise SystemError('no image files found')


# try to import OpenCV if the user didn't choose the option --no-animation
# if not args.no_animation:
#     try:
#         import cv2
#         show_animation = True
#     except ImportError:
#         print("\"opencv-python\" not found, please install to visualize the results.")
#         args.no_animation = True

# try to import Matplotlib if the user didn't choose the option --no-plot
# draw_plot = False
# if not args.no_plot:
#     try:
#         import matplotlib.pyplot as plt
#
#         draw_plot = True
#     except ImportError:
#         print("\"matplotlib\" not found, please install it to get the resulting plots.")
#         args.no_plot = True

if not os.path.exists(pkl_files_path):  # if it doesn't exist already
    os.makedirs(pkl_files_path)

results_files_path = "results"
# if os.path.exists(results_files_path):  # if it exist already
#     # reset the results directory
#     shutil.rmtree(results_files_path)
# else:
#     os.makedirs(results_files_path)
try:
    if draw_plot:
        os.makedirs(results_files_path + "/classes")
    if show_animation:
        os.makedirs(results_files_path + "/images")
except:
    pass

# img = cv2.imread(os.path.join(img_path, src_file_list[0]))
fourcc = cv2.VideoWriter_fourcc(*codec)

if os.path.isdir(img_path_list_file):
    img_path_list = [os.path.join(img_path_list_file, name) for name in os.listdir(img_path_list_file) if
                     os.path.isdir(os.path.join(img_path_list_file, name))]
    img_path_list.sort(key=sortKey)
else:
    img_path_list = utils.file_lines_to_list(img_path_list_file)
    if not gt_path_list_file:
        gt_path_list_file = img_path_list_file
    if img_root_dir:
        img_path_list = [os.path.join(img_root_dir, name) for name in img_path_list]

if os.path.isdir(gt_path_list_file):
    gt_path_list = [os.path.join(gt_path_list_file, name) for name in os.listdir(gt_path_list_file) if
                    os.path.isdir(os.path.join(gt_path_list_file, name))]
    gt_path_list.sort(key=sortKey)
else:
    gt_path_list = utils.file_lines_to_list(gt_path_list_file)
    if gt_root_dir:
        gt_path_list = [os.path.join(gt_root_dir, name) for name in gt_path_list]

if os.path.isdir(det_path_list_file):
    det_path_list = [os.path.join(det_path_list_file, name) for name in os.listdir(det_path_list_file) if
                     os.path.isdir(os.path.join(det_path_list_file, name))]
    det_path_list.sort(key=sortKey)
else:
    det_path_list = utils.file_lines_to_list(det_path_list_file)

n_seq = len(gt_path_list)
if n_seq != len(det_path_list):
    raise IOError('Mismatch between the no. of GT ({}) and detection ({}) sequences'.format(n_seq, len(det_path_list)))
if n_seq != len(img_path_list):
    raise IOError('Mismatch between the no. of GT ({}) and image ({}) sequences'.format(n_seq, len(img_path_list)))

seq_name_list = [os.path.basename(x) for x in img_path_list]

# print('gt_path_list: ', gt_path_list)
# print('det_path_list: ', det_path_list)

ground_truth_files_list = []
for gt_path in gt_path_list:
    gt_files = glob.glob('{}/*.txt'.format(gt_path))
    ground_truth_files_list.append([os.path.join(gt_path, x) for x in gt_files])

if len(ground_truth_files_list) == 0:
    raise SystemError("Error: No ground-truth files found!")

# ground_truth_files_list.sort()
# dictionary with counter per class
gt_counter_per_class_loaded = 0
gt_counter_per_class = {}
gt_counter_per_class_path = os.path.join(pkl_files_path, 'gt_counter_per_class.pkl')
if os.path.exists(gt_counter_per_class_path):
    with open(gt_counter_per_class_path, 'rb') as f:
        gt_counter_per_class = pickle.load(f)
        gt_counter_per_class_loaded = 1

max_gt_path_len = max([len(x) for x in gt_path_list])
gt_data_dict_loaded = 0
gt_data_dict = {}
gt_data_dict_path = os.path.join(pkl_files_path, 'gt_data_dict.pkl')
if os.path.exists(gt_data_dict_path):
    print('Loading GT data from {}'.format(gt_data_dict_path))
    with open(gt_data_dict_path, 'rb') as f:
        gt_data_dict = pickle.load(f)
        gt_data_dict_loaded = 1
else:
    print('Generating GT data...')

gt_start_t = time.time()
for seq_idx, gt_path in enumerate(gt_path_list):

    if not os.path.isdir(gt_path):
        raise IOError('GT folder: {} does not exist'.format(gt_path))

    if gt_data_dict_loaded and gt_counter_per_class_loaded:
        break

    seq_gt_data_dict = {}

    sys.stdout.write('\rProcessing sequence {:d}/{:d}: {:s} '.format(
        seq_idx + 1, n_seq, gt_path.ljust(max_gt_path_len)))
    sys.stdout.flush()

    det_path = det_path_list[seq_idx]
    seq_name = seq_name_list[seq_idx]

    gt_seq_name = os.path.basename(gt_path)
    det_seq_name = os.path.basename(det_path)

    if not det_seq_name.startswith(gt_seq_name):
        print('gt_det_path_list: ', [(os.path.basename(x), os.path.basename(y))
                                     for x, y in zip(gt_path_list, det_path_list)])

        raise IOError('Mismatch between GT and detection sequences: {}, {}'.format(
            gt_seq_name, det_seq_name))

    gt_files = glob.glob('{}/*.txt'.format(gt_path))

    # json_dir = os.path.join(pkl_files_path, seq_name)
    # if not os.path.isdir(json_dir):
    #     os.makedirs(json_dir)

    for gt_file in gt_files:
        # txt_file = os.path.basename(txt_path)
        # gt_path = os.path.dirname(txt_file)
        file_id = gt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        # json_path = os.path.join(json_dir, '{}_ground_truth.json'.format(file_id))
        # if gt_counter_per_class_loaded and os.path.exists(json_path):
        # continue

        # check if there is a correspondent predicted objects file
        det_file = '{}/'.format(det_path) + file_id + ".txt"
        if not os.path.exists(det_file):
            error_msg = "Error. File not found: {}\n".format(det_file)
            print('\ntxt_file: ', gt_file)
            print('\ndet_file: ', det_file)
            raise SystemError(error_msg)
        lines_list = utils.file_lines_to_list(gt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        for line in lines_list:
            try:
                class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + gt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom>\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"rename_class.py\" in the \"extra/\" folder."
                raise SystemError(error_msg)
            # check if class is in the ignore list, if yes skip
            if class_name in args.ignore:
                continue
            bbox = left + " " + top + " " + right + " " + bottom
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "matched": False})
            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1
        seq_gt_data_dict[file_id] = bounding_boxes

        # dump bounding_boxes into a ".json" file
        # with open(json_path, 'w') as outfile:
        #     json.dump(bounding_boxes, outfile)
    gt_data_dict[seq_name] = seq_gt_data_dict

sys.stdout.write('\n')
sys.stdout.flush()

gt_end_t = time.time()
print('Time taken: {} sec'.format(gt_end_t - gt_start_t))

gt_classes = open(labels_path, 'r').readlines()
gt_classes = [x.strip() for x in gt_classes]

for _class in gt_classes:
    if _class not in gt_counter_per_class.keys():
        gt_counter_per_class[_class] = 0

total_gt = 0
for _class in gt_counter_per_class.keys():
    total_gt += gt_counter_per_class[_class]

gt_fraction_per_class = {_class: float(gt_counter_per_class[_class]) / float(total_gt) for _class in
                         gt_counter_per_class.keys()}

# gt_classes = list(gt_counter_per_class.keys())
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)

print('gt_classes: ', gt_classes)
print('n_classes: ', n_classes)
print('gt_counter_per_class: ', gt_counter_per_class)

# print(gt_classes)
# print(gt_counter_per_class)
if not gt_counter_per_class_loaded:
    print('Saving GT count data to {}'.format(gt_data_dict_path))
    with open(gt_counter_per_class_path, 'wb') as f:
        pickle.dump(gt_counter_per_class, f, pickle.HIGHEST_PROTOCOL)
if not gt_data_dict_loaded:
    print('Saving GT data to {}'.format(gt_data_dict_path))
    with open(gt_data_dict_path, 'wb') as f:
        pickle.dump(gt_data_dict, f, pickle.HIGHEST_PROTOCOL)
"""
 Check format of the flag --set-class-iou (if used)
  e.g. check if class exists
"""
if specific_iou_flagged:
    n_args = len(args.set_class_iou)
    error_msg = \
        '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    if n_args % 2 != 0:
        raise SystemError('Error, missing arguments. Flag usage:' + error_msg)
    # [class_1] [IoU_1] [class_2] [IoU_2]
    # specific_iou_classes = ['class_1', 'class_2']
    specific_iou_classes = args.set_class_iou[::2]  # even
    # iou_list = ['IoU_1', 'IoU_2']
    iou_list = args.set_class_iou[1::2]  # odd
    if len(specific_iou_classes) != len(iou_list):
        raise SystemError('Error, missing arguments. Flag usage:' + error_msg)
    for tmp_class in specific_iou_classes:
        if tmp_class not in gt_classes:
            raise SystemError('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
    for num in iou_list:
        if not utils.is_float_between_0_and_1(num):
            raise SystemError('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

"""
 Predicted
   Load each of the predicted files into a temporary ".json" file.
"""
max_det_path_len = max([len(x) for x in det_path_list])
det_data_dict_loaded = 0
det_data_dict = {}
det_data_dict_path = os.path.join(pkl_files_path, 'det_data_dict.pkl')
if os.path.exists(det_data_dict_path):
    print('Loading detection data from {}'.format(gt_data_dict_path))
    with open(det_data_dict_path, 'rb') as f:
        det_data_dict = pickle.load(f)
        det_data_dict_loaded = 1
else:
    print('Generating detection data...')
for class_index, class_name in enumerate(gt_classes):

    if det_data_dict_loaded:
        break

    class_det_data_dict = {}
    print('Processing class {:d}: {:s}'.format(class_index, class_name))

    det_start_t = time.time()

    for seq_idx, det_path in enumerate(det_path_list):
        sys.stdout.write('\rProcessing sequence {:d}/{:d}: {:s} '.format(
            seq_idx + 1, n_seq, det_path.ljust(max_det_path_len)))
        sys.stdout.flush()

        gt_path = gt_path_list[seq_idx]
        seq_name = seq_name_list[seq_idx]

        # json_path = os.path.join(pkl_files_path, seq_name, '{}_predictions.json'.format(class_name))
        # if os.path.exists(json_path):
        #     continue

        # get a list with the predicted files
        predicted_files_list = glob.glob('{}/*.txt'.format(det_path))
        predicted_files_list.sort()

        # seq_det_data_dict = {}
        bounding_boxes = []
        for det_file in predicted_files_list:
            # print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = det_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            if class_index == 0:
                curr_gt_path = os.path.join(gt_path, file_id + ".txt")
                if not os.path.exists(curr_gt_path):
                    error_msg = "Error. File not found: {}\n".format(curr_gt_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
                    raise SystemError(error_msg)
            lines = utils.file_lines_to_list(det_file)

            det_exists = 0

            # bounding_boxes = []
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    try:
                        tmp_class_name_1, tmp_class_name_2, confidence, left, top, right, bottom = line.split()
                        tmp_class_name = '{}_{}'.format(tmp_class_name_1, tmp_class_name_2)
                    except ValueError:
                        error_msg = "Error: File " + det_file + " in the wrong format.\n"
                        error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                        error_msg += " Received: " + line
                        raise SystemError(error_msg)
                if tmp_class_name == class_name:
                    # print("match")
                    # print('txt_file: ', txt_file)
                    # print('tmp_class_name: ', tmp_class_name)
                    # print('confidence: ', confidence)
                    # print('left: ', left)
                    # print('top: ', top)
                    # print('right: ', right)
                    # print('bottom: ', bottom)
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
                    det_exists = 1
                    # print(bounding_boxes)
            if not det_exists:
                gt_data = gt_data_dict[seq_name][file_id]
                for obj in gt_data:
                    if obj['class_name'] == class_name:
                        bounding_boxes.append({"confidence": None, "file_id": file_id, "bbox": None})
                        break

            # seq_det_data_dict[file_id] = bounding_boxes

        # sort predictions by decreasing confidence
        # bounding_boxes.sort(key=lambda x: x['file_id'], reverse=False)

        class_det_data_dict[seq_name] = bounding_boxes
        # class_det_data_dict[seq_name] = seq_det_data_dict

        # with open(json_path, 'w') as outfile:
        #     json.dump(bounding_boxes, outfile)
    det_end_t = time.time()
    det_data_dict[class_name] = class_det_data_dict
    sys.stdout.write('\n')
    sys.stdout.flush()

    print('Time taken: {} sec'.format(det_end_t - det_start_t))

if not det_data_dict_loaded:
    print('Saving detection data to {}'.format(det_data_dict_path))
    with open(det_data_dict_path, 'wb') as f:
        pickle.dump(det_data_dict, f, pickle.HIGHEST_PROTOCOL)

if not save_file_name:
    seq_name = seq_name_list[0]
    save_file_name = os.path.join(save_dir, '{}.mkv'.format(seq_name))
_save_dir = os.path.dirname(save_file_name)
if _save_dir and not os.path.isdir(_save_dir):
    os.makedirs(_save_dir)

"""
 Calculate the AP for each class
"""
weighted_mAP = sum_AP = 0.0
weighted_prec = weighted_rec = 0.0
sum_prec = sum_rec = 0.0

ap_dictionary = {}
print('Calculating the AP for each class...')
# open file to store the results

# with open(results_files_path + "/results.txt", 'w') as results_file:
#     results_file.write("# AP and precision/recall per class\n")


# colors (OpenCV works with BGR)
white = (255, 255, 255)
blue = (255, 0, 0)
light_blue = (255, 200, 100)
green = (0, 255, 0)
light_red = (30, 30, 255)
magenta = (255, 0, 255)
# 1st line
margin = 10
# Add bottom border to image
bottom_border = 60
BLACK = [0, 0, 0]

if show_animation:
    video_h = save_h
    video_w = save_w
    if show_text:
        video_h += bottom_border
    video_out = cv2.VideoWriter(save_file_name, fourcc, fps, (video_w, video_h))
    if not video_out:
        raise SystemError('Visualizations video file: {} could not be opened'.format(save_file_name))
    print('Saving visualizations to {}\n'.format(save_file_name))
    win_name = "s: next sequence c: next class q/escape: quit"

count_true_positives = {}
tp_sum_overall = 0
fp_sum_overall = 0
fn_sum_overall = 0
gt_overall = 0

if not out_fname:
    out_fname = os.path.join('results', '{:s}.txt'.format(
        os.path.splitext(os.path.basename(save_file_name))[0]))
out_file = open(out_fname, 'w')
print('Writing result summary to {}'.format(out_fname))

text = 'class\tAP(%)\tPrecision(%)\tRecall(%)\tTP\tFN\tFP\tGT'
print(text)
out_file.write(text + '\n')

if cv2.__version__.startswith('2'):
    font_line_type = cv2.CV_AA
else:
    font_line_type = cv2.LINE_AA
for class_index, class_name in enumerate(gt_classes):

    # print('Processing class {:d}: {:s}'.format(class_index + 1, class_name))

    count_true_positives[class_name] = 0
    end_class = 0
    """
     Load predictions of that class
    """
    fp_class = []
    tp_class = []
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for seq_idx, det_path in enumerate(det_path_list):
        img_path = img_path_list[seq_idx]
        seq_name = seq_name_list[seq_idx]

        seq_gt_data_dict = gt_data_dict[seq_name]

        # predictions_file = os.path.join(pkl_files_path, seq_name, '{}_predictions.json'.format(class_name))
        # predictions_data = json.load(open(predictions_file))

        predictions_data = det_data_dict[class_name][seq_name]
        predictions_data.sort(key=lambda x: x['file_id'])

        """
         Assign predictions to ground truth objects
        """
        missing_detections = []
        nd = len(predictions_data)
        tp = [0] * nd
        fp = [0] * nd
        fn = [0] * nd
        idx = 0
        img = None
        while True:

            if show_animation and img is not None and \
                    (status == "MATCH!" or not show_only_tp):
                # _h, _w = img.shape[:2]
                # print('_h: ', _h)
                # print('_w: ', _w)
                cv2.imshow(win_name, img)
                video_out.write(img)
                k = cv2.waitKey(1 - _pause)
                if k == ord('q') or k == 27:
                    video_out.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                elif k == ord('c'):
                    end_class = 1
                    break
                elif k == ord('s'):
                    break
                elif k == 32:
                    _pause = 1 - _pause

            if idx >= nd:
                break

            prediction = predictions_data[idx]
            file_id = prediction["file_id"]
            if show_animation:
                ground_truth_img = glob.glob1(img_path, file_id + ".*")
                if len(ground_truth_img) == 0:
                    raise IOError("Error. Image not found for sequence {} with id: {}".format(seq_name, file_id))
                elif len(ground_truth_img) > 1:
                    raise IOError("Error. Multiple image with id: {}".format(file_id))
                else:
                    img_full_path = os.path.join(img_path, ground_truth_img[0])
                    img = cv2.imread(img_full_path)
                    if img is None:
                        raise IOError('Image {} could not be read'.format(img_full_path))

            # gt_file = os.path.join(pkl_files_path, seq_name, '{}_ground_truth.json'.format(file_id))
            # ground_truth_data = json.load(open(gt_file))
            # if gt_file not in loaded_json:
            #     loaded_json.add(gt_file)
            #     for obj in ground_truth_data:
            #         obj["used"] = False

            if show_animation and missing_detections:
                # print('\nfile_id: ', file_id)
                # print('missing_detections:\n ', missing_detections)
                for bb in missing_detections:
                    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), magenta, 2)

                img = resizeAR(img, save_w, save_h)
                img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                height, _ = img.shape[:2]
                v_pos = int(height - margin - (bottom_border / 2))
                text = "{}: {} ".format(seq_name, ground_truth_img[0])
                img, line_width = utils.draw_text_in_image(img, text, (margin, v_pos), white, 0)
                text = "Class [" + str(class_index + 1) + "/" + str(n_classes) + "]: " + class_name + " "
                img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue,
                                                           line_width)

                color = light_red
                status = "MISSING DETECTION"

                text = "Result: {}".format(status)
                img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), color,
                                                           line_width)

                if show_stats:
                    v_pos += int(bottom_border / 2)
                    try:
                        _recall = float(tp_sum) / float(tp_sum + fn_sum)*100.0
                    except ZeroDivisionError:
                        _recall = 0
                    try:
                        _prec = float(tp_sum) / float(tp_sum + fp_sum) * 100.0
                    except ZeroDivisionError:
                        _prec = 0
                    text = 'tp: {:d} fn: {:d} fp: {:d} recall: {:5.2f}% prec: {:5.2f}%'.format(
                        tp_sum, fn_sum, fp_sum, _recall, _prec)
                    img, line_width = utils.draw_text_in_image(img, text, (margin, v_pos), white, line_width)
                missing_detections = []
                idx += 1
                continue

            ground_truth_data = seq_gt_data_dict[file_id]

            if prediction["bbox"] is None:
                fn[idx] = 1
                missing_detections += [[int(x) for x in obj["bbox"].split()] for obj in ground_truth_data if
                                       obj['class_name'] == class_name]
                fn_sum += len(missing_detections)
                if show_animation:
                    if missing_detections:
                        img = None
                else:
                    missing_detections = []
                    idx += 1
                continue

            ovmax = -1
            gt_match = -1
            # load prediction bounding-box
            bb = [float(x) for x in prediction["bbox"].split()]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # assign prediction as true positive or false positive
            if show_animation:
                status = "NO MATCH FOUND!"  # status is only used in the animation
            # set minimum overlap
            min_overlap = iou_thresh
            if specific_iou_flagged:
                if class_name in specific_iou_classes:
                    index = specific_iou_classes.index(class_name)
                    min_overlap = float(iou_list[index])
            if ovmax >= min_overlap:
                if not bool(gt_match["used"]):
                    # true positive
                    tp[idx] = 1
                    tp_sum += 1
                    gt_match["used"] = True
                    count_true_positives[class_name] += 1

                    # update the ".json" file
                    # with open(gt_file, 'w') as f:
                    #     f.write(json.dumps(ground_truth_data))

                    if show_animation:
                        status = "MATCH!"
                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
                    fp_sum += 1
                    if show_animation:
                        status = "REPEATED MATCH!"
            else:
                # false positive
                fp[idx] = 1
                fp_sum += 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

            """
             Draw image to show animation
            """
            if show_animation:

                if status == "MATCH!":
                    box_color = green
                else:
                    box_color = light_red
                cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), box_color, 2)

                if show_gt and ovmax > 0:  # if there is intersections between the bounding-boxes
                    bbgt = [float(x) for x in gt_match["bbox"].split()]
                    cv2.rectangle(img, (int(bbgt[0]), int(bbgt[1])), (int(bbgt[2]), int(bbgt[3])), light_blue, 2)

                img, resize_factor, start_row, start_col = resizeAR(img, save_w, save_h, return_factors=True)

                if not show_text:
                    _xmin = (bb[0] + start_col) * resize_factor
                    _xmax = (bb[2] + start_col) * resize_factor
                    _ymin = (bb[1] + start_row) * resize_factor
                    _ymax = (bb[3] + start_row) * resize_factor

                    _bb = [_xmin, _ymin, _xmax, _ymax]
                    if _bb[1] > 10:
                        y_loc = int(_bb[1] - 5)
                    else:
                        y_loc = int(_bb[3] + 5)
                    box_label = '{}: {:.2f}%'.format(class_name, float(prediction["confidence"]) * 100)
                    cv2.putText(img, box_label, (int(_bb[0] - 1), y_loc),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, font_line_type)
                else:
                    img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                    height, _ = img.shape[:2]
                    v_pos = int(height - margin - (bottom_border / 2))
                    text = "{}: {} ".format(seq_name, ground_truth_img[0])
                    img, line_width = utils.draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    text = "Class [" + str(class_index + 1) + "/" + str(n_classes) + "]: " + class_name + " "
                    img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue,
                                                               line_width)
                    color = light_red
                    if status == "MATCH!":
                        color = green
                    text = "Result: " + status + " "
                    img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), color,
                                                               line_width)

                    v_pos += int(bottom_border / 2)
                    # rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                    # text = "Prediction #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                    #     float(prediction["confidence"]) * 100)

                    text = ''
                    if show_stats:
                        try:
                            _recall = float(tp_sum) / float(tp_sum + fn_sum) * 100.0
                        except ZeroDivisionError:
                            _recall = 0
                        try:
                            _prec = float(tp_sum) / float(tp_sum + fp_sum) * 100.0
                        except ZeroDivisionError:
                            _prec = 0

                        text += 'tp: {:d} fn: {:d} fp: {:d} recall: {:5.2f}% prec: {:5.2f}% '.format(
                            tp_sum, fn_sum, fp_sum, _recall, _prec)
                        # text += 'tp: {:d} fn: {:d} fp: {:d} '.format(tp_sum, fn_sum, fp_sum)
                    text += "confidence: {0:.2f}% ".format(float(prediction["confidence"]) * 100)
                    img, line_width = utils.draw_text_in_image(img, text, (margin, v_pos), white, 0)

                    if ovmax != -1:
                        color = light_red
                        if status == "INSUFFICIENT OVERLAP":
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                        else:
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                            color = green
                        img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), color,
                                                                   line_width)

            if idx == nd - 1 or predictions_data[idx + 1]['file_id'] != file_id:
                missing_detections += [[int(x) for x in obj["bbox"].split()] for obj in ground_truth_data if
                                       obj["class_name"] == class_name and not obj['used']]
                fn_sum += len(missing_detections)
                if show_animation:
                    if missing_detections:
                        # img = None
                        continue
                else:
                    missing_detections = []
            idx += 1

        if end_class:
            break

        fp_class += [x for i, x in enumerate(fp) if fn[i] == 0]
        tp_class += [x for i, x in enumerate(tp) if fn[i] == 0]

        # sys.stdout.write('\rDone sequence {:d}/{:d}: {:s}'.format(seq_idx + 1, n_seq, img_path))
        # sys.stdout.flush()

    # print('fp_class: ', fp_class)
    # print('tp_class: ', tp_class)

    # compute precision/recall
    cumsum = 0
    for idx, val in enumerate(fp_class):
        fp_class[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp_class):
        tp_class[idx] += cumsum
        cumsum += val
    # print('tp: ', tp)

    # print('fp_class: ', fp_class)
    # print('tp_class: ', tp_class)

    rec = tp_class[:]
    for idx, val in enumerate(tp_class):
        if tp_class[idx] > 0:
            rec[idx] = float(tp_class[idx]) / gt_counter_per_class[class_name]
    # print(rec)
    prec = tp_class[:]
    for idx, val in enumerate(tp_class):
        try:
            prec[idx] = float(tp_class[idx]) / (fp_class[idx] + tp_class[idx])
        except ZeroDivisionError:
            prec[idx] = 0

    # print(prec)

    ap, mrec, mprec = utils.voc_ap(rec, prec)

    if tp_sum > 0:
        _rec = float(tp_sum) / gt_counter_per_class[class_name]
    else:
        _rec = 0
    try:
        _prec = float(tp_sum) / (fp_sum + tp_sum)
    except ZeroDivisionError:
        _prec = 0

    weighted_mAP += ap * gt_fraction_per_class[class_name]
    weighted_prec += _prec * gt_fraction_per_class[class_name]
    weighted_rec += _rec * gt_fraction_per_class[class_name]

    sum_AP += ap
    sum_prec += _prec
    sum_rec += _rec

    # text = "{:s} AP: {:.2f}% prec: {:.2f}% rec: {:.2f}% tp: {:d} fn: {:d} fp: {:d} gt: {:d}\n".format(
    #     class_name, ap * 100, _prec * 100, _rec * 100, tp_sum, fn_sum, fp_sum, gt_counter_per_class[class_name])

    text = "{:s}\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}\t{:d}\t{:d}\t{:d}".format(
        class_name, ap * 100, _prec * 100, _rec * 100, tp_sum, fn_sum, fp_sum, gt_counter_per_class[class_name])

    tp_sum_overall += tp_sum
    fp_sum_overall += fp_sum
    fn_sum_overall += fn_sum
    gt_overall += gt_counter_per_class[class_name]

    """
     Write to results.txt
    """
    rounded_prec = ['%.2f' % elem for elem in prec]
    rounded_rec = ['%.2f' % elem for elem in rec]

    # results_file.write(
    #     text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

    # if not args.quiet:
    print(text)
    out_file.write(text + '\n')

    ap_dictionary[class_name] = ap

    """
     Draw plot
    """
    if draw_plot:
        plt.plot(rec, prec, '-o')
        plt.fill_between(mrec, 0, mprec, alpha=0.2, edgecolor='r')
        # set window title
        fig = plt.gcf()  # gcf - get current figure
        fig.canvas.set_window_title('AP ' + class_name)
        # set plot title
        plt.title('class: ' + text)
        # plt.suptitle('This is a somewhat long figure title', fontsize=16)
        # set axis titles
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # optional - set axes
        axes = plt.gca()  # gca - get current axes
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
        # Alternative option -> wait for button to be pressed
        # while not plt.waitforbuttonpress(): pass # wait for key display
        # Alternative option -> normal display
        # plt.show()
        # save the plot
        fig.savefig(results_files_path + "/classes/" + class_name + ".png")
        plt.cla()  # clear axes for next plot

    # sys.stdout.write('\n')
    # sys.stdout.flush()

if show_animation:
    cv2.destroyAllWindows()

mAP = sum_AP / n_classes
m_prec = sum_prec / n_classes
m_rec = sum_rec / n_classes

text = 'Average\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}\t{:d}\t{:d}\t{:d}'.format(
    mAP * 100, m_prec * 100, m_rec * 100,
    tp_sum_overall, fn_sum_overall, fp_sum_overall, gt_overall)

out_file.write(text + '\n')
print(text)

text = 'Weighted\t{:.2f}\t{:.2f}\t{:.2f}'.format(weighted_mAP * 100, weighted_prec * 100, weighted_rec * 100)

out_file.write(text + '\n')
print(text)

out_file.close()

if show_animation:
    video_out.release()
# remove the tmp_files directory
if delete_tmp_files:
    shutil.rmtree(pkl_files_path)

sys.exit(0)

"""
 Count total of Predictions
"""
# iterate through all the files
pred_counter_per_class = {}
print('Counting total predictions...')
for seq_idx, det_path in enumerate(det_path_list):
    sys.stdout.write('\rProcessing sequence {:d}/{:d}: {:s} '.format(seq_idx + 1, n_seq, det_path))
    sys.stdout.flush()

    # get a list with the predicted files
    predicted_files_list = glob.glob('{}/*.txt'.format(det_path))
    predicted_files_list.sort()
    bounding_boxes = []
    # all_classes_predicted_files = set([])
    for gt_file in predicted_files_list:
        # get lines to list
        lines_list = utils.file_lines_to_list(gt_file)
        for line in lines_list:
            class_name = line.split()[0]
            # check if class is in the ignore list, if yes skip
            if class_name in args.ignore:
                continue
            # count that object
            if class_name in pred_counter_per_class:
                pred_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                pred_counter_per_class[class_name] = 1
# print(pred_counter_per_class)
pred_classes = list(pred_counter_per_class.keys())

"""
 Plot the total number of occurences of each class in the ground-truth
"""
if draw_plot:
    window_title = "Ground-Truth Info"
    plot_title = "Ground-Truth\n"
    plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = results_files_path + "/Ground-Truth Info.png"
    to_show = False
    plot_color = 'forestgreen'
    utils.draw_plot_func(
        gt_counter_per_class,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        '',
    )

"""
 Write number of ground-truth objects per class to results.txt
"""
with open(results_files_path + "/results.txt", 'a') as results_file:
    results_file.write("\n# Number of ground-truth objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

"""
 Finish counting true positives
"""
for class_name in pred_classes:
    # if class exists in predictions but not in ground-truth then there are no true positives in that class
    if class_name not in gt_classes:
        count_true_positives[class_name] = 0
# print(count_true_positives)

"""
 Plot the total number of occurences of each class in the "predicted" folder
"""
if draw_plot:
    window_title = "Predicted Objects Info"
    # Plot title
    plot_title = "Predicted Objects\n"
    # plot_title += "(" + str(len(predicted_files_list)) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(pred_counter_per_class.values()))
    plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = results_files_path + "/Predicted Objects Info.png"
    to_show = False
    plot_color = 'forestgreen'
    true_p_bar = count_true_positives
    utils.draw_plot_func(
        pred_counter_per_class,
        len(pred_counter_per_class),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        true_p_bar
    )

"""
 Write number of predicted objects per class to results.txt
"""
with open(results_files_path + "/results.txt", 'a') as results_file:
    results_file.write("\n# Number of predicted objects per class\n")
    for class_name in sorted(pred_classes):
        n_pred = pred_counter_per_class[class_name]
        text = class_name + ": " + str(n_pred)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"
        results_file.write(text)

"""
 Draw mAP plot (Show AP's of all classes in decreasing order)
"""
if draw_plot:
    window_title = "mAP"
    plot_title = "mAP = {0:.2f}%".format(mAP * 100)
    x_label = "Average Precision"
    output_path = results_files_path + "/mAP.png"
    to_show = True
    plot_color = 'royalblue'
    utils.draw_plot_func(
        ap_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )
