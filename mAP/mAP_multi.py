import glob
import json
import os, sys
import shutil
import pickle

import argparse
import cv2
import map_utils as utils
import matplotlib.pyplot as plt

sys.path.append('..')
from tf_api.utilities import resizeAR

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
parser.add_argument('--labels_path', type=str, help="text file containing class labels")
parser.add_argument('--gt_path_list_file', type=str, help="file containing list of GT folders")
parser.add_argument('--det_path_list_file', type=str, help="file containing list of detection folders")
parser.add_argument('--img_path_list_file', type=str, help="file containing list of image folders")
parser.add_argument('--img_ext', type=str, help="image extension", default='jpg')
parser.add_argument('--save_dir', type=str, help="folder to save the animation result in", default='')
parser.add_argument('--save_file_name', type=str, help="file to save the animation result in", default='mAP.mkv')
parser.add_argument('--save_file_res', type=str, help="resolution of the saved video", default='1280x720')
parser.add_argument('--tmp_files_path', type=str, help="location of temporary json files", default='tmp_files')
parser.add_argument('--codec', type=str, help="file to save the animation result in", default='H264')
parser.add_argument('--fps', type=int, help="FPS", default=20)
parser.add_argument('--delete_tmp_files', type=int, default=0)
parser.add_argument('--show_animation', type=int, default=1)
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
tmp_files_path = args.tmp_files_path
delete_tmp_files = args.delete_tmp_files
show_animation = args.show_animation
draw_plot = args.draw_plot

save_w, save_h = [int(x) for x in save_file_res.split('x')]

if not save_dir:
    save_dir = 'results'

_pause = 1
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


"""
 Create a "tmp_files/" and "results/" directory
"""
if not os.path.exists(tmp_files_path):  # if it doesn't exist already
    os.makedirs(tmp_files_path)

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

"""
 Ground-Truth
   Load each of the ground-truth files into a temporary ".json" file.
   Create a list of all the class names present in the ground-truth (gt_classes).
"""
# get a list with the ground-truth files

gt_path_list = utils.file_lines_to_list(args.gt_path_list_file)
det_path_list = utils.file_lines_to_list(args.det_path_list_file)
img_path_list = utils.file_lines_to_list(args.img_path_list_file)
n_seq = len(gt_path_list)
if n_seq != len(det_path_list):
    raise IOError('Mismatch between the no. of GT and detection sequences')
if n_seq != len(img_path_list):
    raise IOError('Mismatch between the no. of GT and image sequences')

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
gt_counter_per_class_path = os.path.join(tmp_files_path, 'gt_counter_per_class.pkl')
if os.path.exists(gt_counter_per_class_path):
    with open(gt_counter_per_class_path, 'rb') as f:
        gt_counter_per_class = pickle.load(f)
        gt_counter_per_class_loaded = 1

gt_data_dict_loaded = 0
gt_data_dict = {}
gt_data_dict_path = os.path.join(tmp_files_path, 'gt_data_dict.pkl')
if os.path.exists(gt_data_dict_path):
    print('Loading GT data from {}'.format(gt_data_dict_path))
    with open(gt_data_dict_path, 'rb') as f:
        gt_data_dict = pickle.load(f)
        gt_data_dict_loaded = 1
else:
    print('Generating GT data...')
for seq_idx, gt_path in enumerate(gt_path_list):

    if gt_data_dict_loaded and gt_counter_per_class_loaded:
        break

    seq_gt_data_dict = {}

    sys.stdout.write('\rProcessing sequence {:d}/{:d}: {:s} '.format(seq_idx + 1, n_seq, gt_path))
    sys.stdout.flush()

    det_path = det_path_list[seq_idx]
    seq_name = seq_name_list[seq_idx]

    gt_files = glob.glob('{}/*.txt'.format(gt_path))

    # json_dir = os.path.join(tmp_files_path, seq_name)
    # if not os.path.isdir(json_dir):
    #     os.makedirs(json_dir)

    for txt_file in gt_files:
        # txt_file = os.path.basename(txt_path)
        # gt_path = os.path.dirname(txt_file)
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        # json_path = os.path.join(json_dir, '{}_ground_truth.json'.format(file_id))
        # if gt_counter_per_class_loaded and os.path.exists(json_path):
        # continue

        # check if there is a correspondent predicted objects file
        det_file = '{}/'.format(det_path) + file_id + ".txt"
        if not os.path.exists(det_file):
            error_msg = "Error. File not found: {}\n".format(det_file)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
            raise SystemError(error_msg)
        lines_list = utils.file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        for line in lines_list:
            try:
                class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
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
# print(gt_classes)
# print(gt_counter_per_class)
if not gt_counter_per_class_loaded:
    with open(gt_counter_per_class_path, 'wb') as f:
        pickle.dump(gt_counter_per_class, f, pickle.HIGHEST_PROTOCOL)
if not gt_data_dict_loaded:
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

det_data_dict_loaded = 0
det_data_dict = {}
det_data_dict_path = os.path.join(tmp_files_path, 'det_data_dict.pkl')
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

    for seq_idx, det_path in enumerate(det_path_list):
        sys.stdout.write('\rProcessing sequence {:d}/{:d}: {:s} '.format(seq_idx + 1, n_seq, det_path))
        sys.stdout.flush()

        gt_path = gt_path_list[seq_idx]
        seq_name = seq_name_list[seq_idx]

        # json_path = os.path.join(tmp_files_path, seq_name, '{}_predictions.json'.format(class_name))
        # if os.path.exists(json_path):
        #     continue

        # get a list with the predicted files
        predicted_files_list = glob.glob('{}/*.txt'.format(det_path))
        predicted_files_list.sort()

        seq_det_data_dict = {}
        # bounding_boxes = []
        for txt_file in predicted_files_list:
            # print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            if class_index == 0:
                curr_gt_path = os.path.join(gt_path, file_id + ".txt")
                if not os.path.exists(curr_gt_path):
                    error_msg = "Error. File not found: {}\n".format(curr_gt_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
                    raise SystemError(error_msg)
            lines = utils.file_lines_to_list(txt_file)

            det_exists = 0

            bounding_boxes = []
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
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

            seq_det_data_dict[file_id] = bounding_boxes

        # sort predictions by decreasing confidence
        # bounding_boxes.sort(key=lambda x: x['file_id'], reverse=False)

        # class_det_data_dict[seq_name] = bounding_boxes
        class_det_data_dict[seq_name] = seq_det_data_dict

        # with open(json_path, 'w') as outfile:
        #     json.dump(bounding_boxes, outfile)
    det_data_dict[class_name] = class_det_data_dict

sys.stdout.write('\n')
sys.stdout.flush()
if not det_data_dict_loaded:
    print('Saving detection data to {}'.format(gt_data_dict_path))
    with open(det_data_dict_path, 'wb') as f:
        pickle.dump(det_data_dict, f, pickle.HIGHEST_PROTOCOL)

if not save_file_name:
    seq_name = seq_name_list[0]
    save_file_name = os.path.join(save_dir, '{}.mkv'.format(seq_name))
_save_dir = os.path.dirname(save_file_name)
print('Saving visualizations to {}'.format(save_file_name))
if _save_dir and not os.path.isdir(_save_dir):
    os.makedirs(_save_dir)
video_out = cv2.VideoWriter(save_file_name, fourcc, fps, (save_w, save_h))
if not video_out:
    raise SystemError('Output video file: {} could not be opened'.format(save_file_name))

"""
 Calculate the AP for each class
"""
weighted_AP = sum_AP = 0.0
ap_dictionary = {}
print('Calculating the AP for each class...')
# open file to store the results

# with open(results_files_path + "/results.txt", 'w') as results_file:
#     results_file.write("# AP and precision/recall per class\n")

loaded_json = set()

count_true_positives = {}
for class_index, class_name in enumerate(gt_classes):

    print('Processing class {:d}: {:s}'.format(class_index, class_name))

    count_true_positives[class_name] = 0
    end_class = 0
    """
     Load predictions of that class
    """
    fp_class = []
    tp_class = []
    for seq_idx, det_path in enumerate(det_path_list):
        img_path = img_path_list[seq_idx]
        seq_name = seq_name_list[seq_idx]

        seq_gt_data_dict = gt_data_dict[seq_name]

        # predictions_file = os.path.join(tmp_files_path, seq_name, '{}_predictions.json'.format(class_name))
        # predictions_data = json.load(open(predictions_file))

        seq_det_data = det_data_dict[class_name][seq_name]

        # predictions_data = det_data_dict[class_name][seq_name]
        # predictions_data.sort(key=lambda x: x['file_id'])

        for file_id, predictions_data in seq_det_data.items():
		
		  file_id = prediction["file_id"]
		  if show_animation:
			# find ground truth image
			ground_truth_img = glob.glob1(img_path, file_id + ".*")
			#tifCounter = len(glob.glob1(myPath,"*.tif"))
			if len(ground_truth_img) == 0:
			  error("Error. Image not found with id: " + file_id)
			elif len(ground_truth_img) > 1:
			  error("Error. Multiple image with id: " + file_id)
			else: # found image
			  #print(img_path + "/" + ground_truth_img[0])
			  # Load image
			  img = cv2.imread(img_path + "/" + ground_truth_img[0])
			  # Add bottom border to image
			  bottom_border = 60
			  BLACK = [0, 0, 0]
			  img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
            """
             Assign predictions to ground truth objects
            """
            nd = len(predictions_data)
            tp = [0] * nd
            fp = [0] * nd
            fn = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                # gt_file = os.path.join(tmp_files_path, seq_name, '{}_ground_truth.json'.format(file_id))
                # ground_truth_data = json.load(open(gt_file))
                # if gt_file not in loaded_json:
                #     loaded_json.add(gt_file)
                #     for obj in ground_truth_data:
                #         obj["used"] = False

                ground_truth_data = seq_gt_data_dict[file_id]

                ovmax = -1
                gt_match = -1

                if prediction["bbox"] is None:
                    if show_animation:
                        status = 'MISSING DETECTION'
                else:
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
                        if show_animation:
                            status = "REPEATED MATCH!"
                else:
                    # false positive
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

            """
             Draw image to show animation
            """
            if show_animation:
                height, widht = img.shape[:2]
                # colors (OpenCV works with BGR)
                white = (255, 255, 255)
                light_blue = (255, 200, 100)
                green = (0, 255, 0)
                light_red = (30, 30, 255)
                magenta = (255, 0, 255)
                # 1st line
                margin = 10
                v_pos = int(height - margin - (bottom_border / 2))
                text = "{}: {} ".format(seq_name, ground_truth_img[0])
                img, line_width = utils.draw_text_in_image(img, text, (margin, v_pos), white, 0)
                text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue,
                                                           line_width)
                if status == "MISSING DETECTION":
                    for obj in ground_truth_data:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        cv2.rectangle(img, (int(bbgt[0]), int(bbgt[1])), (int(bbgt[2]), int(bbgt[3])), magenta, 2)
                else:
                    if ovmax != -1:
                        color = light_red
                        if status == "INSUFFICIENT OVERLAP":
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                        else:
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                            color = green
                        img, _ = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                    # 2nd line
                    v_pos += int(bottom_border / 2)
                    rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                    text = "Prediction #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                        float(prediction["confidence"]) * 100)
                    img, line_width = utils.draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    if status == "MATCH!":
                        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), green, 2)
                    else:
                        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), light_red, 2)

                color = light_red
                if status == "MATCH!":
                    color = green
                text = "Result: " + status + " "
                img, line_width = utils.draw_text_in_image(img, text, (margin + line_width, v_pos), color,
                                                           line_width)
                if ovmax > 0:  # if there is intersections between the bounding-boxes
                    bbgt = [float(x) for x in gt_match["bbox"].split()]
                    cv2.rectangle(img, (int(bbgt[0]), int(bbgt[1])), (int(bbgt[2]), int(bbgt[3])), light_blue, 2)

                img_res_ar = resizeAR(img, save_w, save_h)
                cv2.imshow("Animation", img_res_ar)
                video_out.write(img_res_ar)
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

                # save image to results
                # output_img_path = results_files_path + "/images/" + class_name + "_prediction" + str(idx) + ".jpg"
                # cv2.imwrite(output_img_path, img)

        if end_class:
            break

        fp_class = fp_class + fp
        tp_class = tp_class + tp

        sys.stdout.write('\rDone sequence {:d}/{:d}: {:s}'.format(seq_idx + 1, n_seq, img_path))
        sys.stdout.flush()

    # print(tp)
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
    rec = tp_class[:]
    for idx, val in enumerate(tp_class):
        if tp_class[idx] > 0:
            rec[idx] = float(tp_class[idx]) / gt_counter_per_class[class_name]
    # print(rec)
    prec = tp_class[:]
    for idx, val in enumerate(tp_class):
        prec[idx] = float(tp_class[idx]) / (fp_class[idx] + tp_class[idx])
    # print(prec)

    ap, mrec, mprec = utils.voc_ap(rec, prec)
    sum_AP += ap
    weighted_AP += ap * gt_fraction_per_class[class_name]
    text = "\n{0:.2f}%".format(
        ap * 100) + " = " + class_name + " AP  "  # class_name + " AP = {0:.2f}%".format(ap*100)
    """
     Write to results.txt
    """
    rounded_prec = ['%.2f' % elem for elem in prec]
    rounded_rec = ['%.2f' % elem for elem in rec]

    # results_file.write(
    #     text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

    # if not args.quiet:
    print(text)

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
    sys.stdout.write('\n')
    sys.stdout.flush()

if show_animation:
    cv2.destroyAllWindows()

# results_file.write("\n# mAP of all classes\n")

mAP = sum_AP / n_classes
text = "mAP = {0:.2f}%\n".format(mAP * 100)
text += "weighted_AP = {0:.2f}%\n".format(weighted_AP * 100)

# results_file.write(text)

print(text)
video_out.release()

# remove the tmp_files directory
if delete_tmp_files:
    shutil.rmtree(tmp_files_path)

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
    for txt_file in predicted_files_list:
        # get lines to list
        lines_list = utils.file_lines_to_list(txt_file)
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
