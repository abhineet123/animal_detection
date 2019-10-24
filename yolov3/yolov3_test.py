import sys

sys.path.append('..')

import torch.multiprocessing as multiprocessing

try:
     multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
from contextlib import closing

import time
import functools
import paramparse

import pandas as pd
from pprint import pformat

from torch.utils.data import DataLoader

from YOLOv3TestParams import YOLOv3TestParams

from yolov3_models import *
from yolov3_utils.datasets import *
from yolov3_utils.utils import *

from labelling_tool.tracking.DaSiamRPN.DaSiamRPN import DaSiamRPN
from labelling_tool.tracking.siamfc.SiamFC import SiamFC
from labelling_tool.tracking.SiamMask.SiamMask import SiamMask

from labelling_tool.tracking.Utilities import drawBox


def runDetector(_idx,
                model, imgs, device, conf_thresh, nms_type, nms_thresh,
                # trackers=None, curr_frame=None
                ):
    # if _idx > 0:
    #     # _start_t = time.time()
    #     tracker = trackers[_idx - 1]
    #     tracker.update(curr_frame)
    #     # _end_t = time.time()
    #     # remove_tracker = 0
    #     # if tracker.confidence < track_thresh:
    #     #     # trackers_to_remove.append(tracker)
    #     #     remove_tracker = 1
    #     #     print('Removing tracker {} with confidence: {}'.format(
    #     #         tracker.target_id, tracker.confidence))
    #     return None, None, None, None

    _start_t = time.time()

    # targets = targets.to(device)
    # Plot images with bounding boxes
    # if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
    #     plot_images(imgs=imgs, targets=targets, fname='test_batch0.jpg')

    # Run model
    inf_out, train_out = model(imgs)  # inference and training outputs

    # Compute loss
    # if hasattr(model, 'hyp'):  # if model has loss hyperparameters
    #     loss_i, _ = compute_loss(train_out, targets, model)
    #     loss += loss_i.item()

    _end_t = time.time()

    # Run NMS
    output = non_max_suppression(inf_out, conf_thres=conf_thresh,
                                 nms_thres=nms_thresh,
                                 nms_style=nms_type)

    nms_end_t = time.time()

    return output, _start_t, _end_t, nms_end_t

def test(opt, model=None):
    """
    :param YOLOv3TestParams opt:
    :param model:
    :return:
    """
    weights_base = os.path.basename(opt.weights)
    weights_dir = os.path.dirname(opt.weights)
    test_path_base = os.path.basename(opt.test_path)
    if not opt.save_dir:
        prefix = os.path.splitext(weights_base)[0]
        if opt.out_suffix:
            prefix = '{}_{}'.format(prefix, opt.out_suffix)
        opt.save_dir = '{}_on_{}'.format(prefix, os.path.splitext(test_path_base)[0])

    opt.save_dir = os.path.join(weights_dir, opt.save_dir)
    if not os.path.isdir(opt.save_dir):
        os.makedirs(opt.save_dir)

    _tracker_types = opt.help['tracker_type']

    if opt.tracker_type:
        if opt.batch_size > 1:
            raise IOError('Batch size must be 1 to use tracking')

        _tracker_type = [k for k in _tracker_types if opt.tracker_type in _tracker_types[k]]
        if not _tracker_type:
            raise IOError('Invalid tracker_type: {}'.format(opt.tracker_type))
        _tracker_type = _tracker_type[0]

        # print('Tracking is enabled')
        if _tracker_type == 'SiamFC':
            create_tracker = functools.partial(SiamFC, params=opt.siam_fc)
            print('Using SiamFC tracker')
        elif _tracker_type == 'DaSiamRPN':
            create_tracker = functools.partial(DaSiamRPN, params=opt.da_siam_rpn, logger=None, )
            print('Using DaSiamRPN tracker')
        elif _tracker_type == 'SiamMask':
            create_tracker = functools.partial(SiamMask, params=opt.siam_mask, )
            print('Using SiamMask tracker')
        else:
            raise IOError('Invalid tracker_type: {}'.format(opt.tracker_type))

        max_target_id = 0

    print('Saving csv results to: {}'.format(opt.save_dir))

    nms_type = opt.nms_type.upper()
    conf_thresh = opt.conf_thresh
    nms_thresh = opt.nms_thresh

    if model is None:
        device = torch_utils.select_device()

        # Initialize model
        model = Darknet(opt.net_cfg, opt.img_size).to(device)

        # Load weights
        if opt.weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, opt.weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device

    # Configure run
    data_cfg = parse_data_cfg(opt.data_cfg)
    nc = int(data_cfg['classes'])  # number of classes
    # test_path = data_cfg['test']  # path to test images
    names = load_classes(data_cfg['names'])  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(opt.test_path, opt.img_size, opt.batch_size, rect=False, sort_files=True)
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    # seen = 0
    model.eval()
    # # coco91class = coco80_to_coco91_class()
    # print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
    loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
    # jdict, stats, ap, ap_class = [], [], [], []

    csv_raw = []
    prev_eval_seq = ''
    seq_id = 0
    n_frames = len(dataset.img_files)
    frame_id = 0
    avg_fps = avg_nms_fps = avg_tracking_fps = avg_overall_fps = 0
    seq_n_frames = 0
    prev_tracker_frame_id = -1
    trackers = []
    _pause = 1

    tracking_fps = 0

    print('Processing {} sequences'.format(dataset.n_sequences))
    for batch_i, (orig_imgs, imgs, targets, paths, shapes) in enumerate(dataloader):
        # tqdm(dataloader, desc='Running on {} image batches'.format(batch_size)))

        actual_batch_size, _, height, width = imgs.shape  # batch size, channels, height, width
        curr_frame = orig_imgs[0]
        imgs = imgs.to(device)

        if trackers:
            n_trackers = len(trackers)

            # n_ops = n_trackers + 1
            # n_threads = min(n_ops, multiprocessing.cpu_count())
            # print('Running {} trackers and detector using {} threads'.format(
            #     n_trackers, n_threads))
            # combined_start_t = time.time()
            # with closing(multiprocessing.Pool(n_threads)) as pool:
            #     combined_out_list = pool.map(functools.partial(
            #         runDetector,
            #         # model=model,
            #         # imgs=imgs,
            #         # device=device,
            #         # conf_thresh=opt.conf_thresh,
            #         # nms_type=nms_type,
            #         # nms_thresh=opt.nms_thresh,
            #         # trackers=trackers,
            #         # curr_frame=curr_frame,
            #     ), range(n_ops))
            # combined_end_t = time.time()
            # combined_fps = 1.0 / (combined_end_t - combined_start_t)
            # print('Combined fps: {:.4f}'.format(combined_fps))

            # output, _start_t, _end_t, nms_end_t = combined_out_list[0]

            output, _start_t, _end_t, nms_end_t = runDetector(0,
                                                              model, imgs, device,
                                                              conf_thresh, nms_type, nms_thresh
                                                              )

            removed_target_ids = []
            combined_start_t = time.time()
            # trackers_to_remove = []
            # print('before: n_trackers: {}'.format(n_trackers))
            for i, tracker in enumerate(trackers):
                tracker.update(curr_frame)
                if tracker.confidence < opt.track_thresh:
                    # trackers_to_remove.append(tracker)
                    removed_target_ids.append(tracker.target_id)
                    if opt.verbose:
                        print('Removing tracker {} with confidence: {}'.format(
                            tracker.target_id, tracker.confidence))
            combined_end_t = time.time()
            # track_t = (combined_end_t - combined_start_t) - (_end_t - _start_t)
            track_t = combined_end_t - combined_start_t
            try:
                tracking_fps = 1.0 / float(track_t)
            except ZeroDivisionError:
                tracking_fps = 0
            trackers = [tracker for tracker in trackers if tracker.target_id not in removed_target_ids]
        else:
            output, _start_t, _end_t, nms_end_t = runDetector(0,
                                                              model, imgs, device,
                                                              conf_thresh, nms_type, nms_thresh
                                                              )

        # print('\noutput:\n{}'.format(pformat(output)))
        # print('\npaths:\n{}'.format(pformat(paths)))

        # Statistics per image

        for si, pred in enumerate(output):
            # labels = targets[targets[:, 0] == si, 1:]
            # nl = len(labels)
            # tcls = labels[:, 0].tolist() if nl else []  # target class
            # seen += 1

            curr_frame = orig_imgs[si]
            eval_file = paths[si]
            eval_seq = os.path.dirname(eval_file)

            if not prev_eval_seq:
                prev_eval_seq = eval_seq
                seq_id += 1
                seq_n_frames = dataset.seq_to_n_frames[prev_eval_seq]
                print('\nProcessing sequence {}/{}: {}'.format(seq_id, dataset.n_sequences, prev_eval_seq))
                seq_frame_id = 0
                prev_tracker_frame_id = -1

            if eval_seq != prev_eval_seq:
                if prev_eval_seq:
                    # print('Done sequence {} with {} frames'.format(prev_eval_seq, len(csv_raw)))
                    eval_seq_name = os.path.basename(prev_eval_seq)
                    csv_file_name = os.path.join(opt.save_dir, '{}.csv'.format(eval_seq_name))
                    print('\nWriting csv data for {} frames to: {}\n'.format(len(csv_raw), csv_file_name))
                    pd.DataFrame(csv_raw).to_csv(csv_file_name)
                csv_raw = []
                trackers = []
                prev_eval_seq = eval_seq
                seq_id += 1
                prev_tracker_frame_id = -1
                seq_frame_id = 0
                seq_n_frames = dataset.seq_to_n_frames[prev_eval_seq]
                print('\nProcessing sequence {}/{}: {}'.format(seq_id, dataset.n_sequences, prev_eval_seq))

            if pred is None:
                print('\nNone pred for {}'.format(eval_file))
                pred = []
                n_raw_dets = 0
            else:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                height, width = imgs[si].shape[1:3]
                # image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape

                # box = xyxy2xywh(box)  # xywh
                # box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

                n_raw_dets = pred.shape[0]

                # print('\nProcessing eval_file {}'.format(eval_file))

            new_trackers = {}
            bboxes = []

            # print('n_raw_dets: {}'.format(n_raw_dets))
            for di, d in enumerate(pred):

                xmin, ymin, xmax, ymax = [float(x) for x in box[di]]
                label_id = int(d[6])
                label = names[label_id]
                confidence = float(d[4])

                bbox = [xmin, ymin, xmax, ymax, label, confidence, -1]

                # associated_bboxes = {}
                # bboxes_to_initialize = {}

                if opt.tracker_type:
                    _id = associate(trackers, [xmin, ymin, xmax, ymax], label, opt.assoc_thresh)
                    if _id < 0:
                        # unassociated detection
                        if prev_tracker_frame_id == -1 or seq_frame_id - prev_tracker_frame_id >= opt.track_diff:
                            # min frame difference between tracker creation
                            max_target_id += 1
                            new_tracker = create_tracker(target_id=max_target_id,
                                                         label=label,
                                                         confidence=confidence)
                            w = xmax - xmin
                            h = ymax - ymin

                            cx = xmin + w / 2.0
                            cy = ymin + h / 2.0

                            # bboxes_to_initialize[max_target_id] = [cx, cy, w, h]

                            new_trackers[max_target_id] = (new_tracker, [cx, cy, w, h])

                            if opt.filter_unassociated:
                                bbox = [xmin, ymin, xmax, ymax, label, confidence, new_tracker.target_id]

                            # associated_bboxes[new_tracker.target_id] = bbox
                    else:
                        _det_bbox = [xmin, ymin, xmax, ymax]

                        # _tracker_bbox = trackers[_id].bbox
                        # mean_bbox = [(_tracker_bbox[i] + _det_bbox[i]) / 2.0 for i in range(4)]
                        # mean_confidence = (trackers[_id].confidence + confidence) / 2.0

                        mean_bbox = _det_bbox
                        mean_confidence = confidence

                        bbox = mean_bbox + [label, mean_confidence, 0]
                        # associated_bboxes[trackers[_id].target_id] = bbox

                bboxes.append(bbox)

            n_det_bboxes = len(bboxes)
            removed_target_ids = []

            if opt.tracker_type:
                # trackers_to_remove = []
                for i, tracker in enumerate(trackers):
                    if tracker.associated:
                        tracker.associated_frames += 1
                        tracker.associated = 0
                        tracker.unassociated_frames = 0
                        continue

                    tracker.unassociated_frames += 1
                    if opt.unassoc_thresh and tracker.unassociated_frames > opt.unassoc_thresh:
                        if opt.verbose:
                            print('Removing tracker {} with unassociated_frames: {}'.format(
                                tracker.target_id, tracker.unassociated_frames))
                        # trackers_to_remove.append(tracker)
                        removed_target_ids.append(tracker.target_id)
                        continue

                    bboxes.append(tracker.bbox +
                                  [tracker.label, tracker.cumulative_confidence, tracker.target_id])
                    tracker.associated = 0

                # remove trackers that have gone unassociated for too long
                n_removed_trackers = len(removed_target_ids)
                # for tracker in trackers_to_remove:
                #     print('Removing tracker {} with unassociated_frames: {}'.format(
                #         tracker.target_id, tracker.unassociated_frames))

                trackers = [tracker for tracker in trackers if tracker.target_id not in removed_target_ids]

                if new_trackers:
                    prev_tracker_frame_id = seq_frame_id
                    trackers += [new_tracker[0] for _, new_tracker in new_trackers.items()]

                n_trackers = len(trackers)

                if opt.max_trackers and n_trackers > opt.max_trackers:
                    # remove trackers with least confidences
                    trackers.sort(
                        key=lambda x: x.cumulative_confidence * float(x.associated_frames + 1) / float(
                            1 + x.unassociated_frames))
                    n_trackers_to_remove = n_trackers - opt.max_trackers
                    for del_id in range(n_trackers_to_remove):
                        _tracker = trackers[del_id]
                        if opt.verbose:
                            print('Deleting tracker {} with:'
                                  ' cumulative_confidence: {}'
                                  ' associated_frames: {}'
                                  ' unassociated_frames: {}'.format(
                                _tracker.target_id,
                                _tracker.cumulative_confidence,
                                _tracker.associated_frames,
                                _tracker.unassociated_frames,
                            ))
                        removed_target_ids.append(_tracker.target_id)

                    trackers = trackers[-opt.max_trackers:]

                    n_trackers = opt.max_trackers
                # print('after: n_trackers: {}'.format(n_trackers))
                for _id in new_trackers:
                    if _id not in removed_target_ids:
                        _tracker, _bbox = new_trackers[_id]
                        _tracker.initialize(curr_frame, _bbox)

            bboxes = [bbox for bbox in bboxes if bbox[-1] not in removed_target_ids]

            n_bboxes = len(bboxes)
            n_track_bboxes = n_bboxes - n_det_bboxes

            filename = os.path.basename(eval_file)

            for di, bbox in enumerate(bboxes):
                xmin, ymin, xmax, ymax, label, confidence, target_id = bbox

                csv_raw.append({
                    'filename': filename,
                    'width': width,
                    'height': height,
                    'class': label,
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax),
                    'confidence': confidence,
                })
                if opt.vis:
                    w = xmax - xmin
                    h = ymax - ymin
                    if target_id == 0:
                        # associated detection
                        col = 'blue'
                    elif target_id > 0:
                        # tracked
                        col = 'red'
                    else:
                        # unassociated detection
                        col = 'green'
                    drawBox(curr_frame, np.array([xmin, ymin, w, h]), color=col,
                            _id=target_id,
                            label='{}:{:.1f}'.format(label, confidence * 100.0))

        overall_end_t = time.time()

        fps = float(actual_batch_size) / float(_end_t - _start_t)
        nms_fps = float(actual_batch_size) / float(nms_end_t - _start_t)
        overall_fps = float(actual_batch_size) / float(overall_end_t - _start_t)

        avg_fps += (fps - avg_fps) / float(batch_i + 1)
        avg_nms_fps += (nms_fps - avg_nms_fps) / float(batch_i + 1)
        avg_overall_fps += (overall_fps - avg_overall_fps) / float(batch_i + 1)

        frame_id += actual_batch_size
        seq_frame_id += actual_batch_size

        # print('actual_batch_size', actual_batch_size)

        txt = 'Done {:d}({:d})/{:d}({:d}) frames d_box: {:d}'.format(
            frame_id, seq_frame_id, n_frames, seq_n_frames, n_det_bboxes)
        if opt.tracker_type:
            avg_tracking_fps += (tracking_fps - avg_tracking_fps) / float(batch_i + 1)
            txt = '{} t_box: {:d} n_track: {:d} rm: {:d} t_fps: {:.2f}'.format(
                txt, n_track_bboxes, n_trackers, n_removed_trackers, avg_tracking_fps)
        txt = ' {} fps: {:.2f} nms: {:.2f} overall: {:.2f}'.format(
            txt, avg_fps, avg_nms_fps, avg_overall_fps
        )
        sys.stdout.write('\r' + txt)
        sys.stdout.flush()

        if opt.vis:
            win_name = 'blue: associated green: unassociated red: tracked'
            cv2.imshow(win_name, curr_frame)
            k = cv2.waitKey(1 - _pause)
            if k == 32:
                _pause = 1 - _pause
            elif k == 27:
                sys.exit(0)

    if csv_raw and prev_eval_seq:
        eval_seq_name = os.path.basename(prev_eval_seq)
        csv_file_name = os.path.join(opt.save_dir, '{}.csv'.format(eval_seq_name))
        print('\nWriting csv data for {} frames to: {}\n'.format(len(csv_raw), csv_file_name))
        pd.DataFrame(csv_raw).to_csv(csv_file_name)

    # if enable_eval:
    #     cmd = 'python3 ../tf_api/tf_api_eval.py ' \
    #           'labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt ' \
    #           'n_frames=0 show_img=0 n_classes=6 eval_every=0 ' \
    #           'root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  ' \
    #           'random_sampling=0 sleep_time=10 write_summary=1 save_det=1 ' \
    #           'load_det=1 ' \
    #           'load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv/ ' \
    #           'load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv ' \
    #           'score_thresholds=0:1:0.001 allow_seq_skipping=1'


def associate(_trackers, _bbox, _label, assoc_thresh):
    for _id, _tracker in enumerate(_trackers):
        if _tracker.associated or _tracker.label != _label:
            continue
        iou = bbox_iou_np(_bbox, _tracker.bbox)
        if iou > assoc_thresh:
            _tracker.associated = 1
            return _id
    return -1


if __name__ == '__main__':
    params = YOLOv3TestParams()
    paramparse.process(params)

    with torch.no_grad():
        test(params)
