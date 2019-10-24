import pandas as pd
import os
import ast
import sys, time, random
from pprint import pprint, pformat
import math
# import argparse
import paramparse

from PollingParams import PollingParams
from utilities import sortKey, resizeAR


def getIOU(bb, bbgt):
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
        ov = iw * ih / ua

        return ov
    return 0

def removeDuplicates(bboxes, iou_thresh=0.9):
    duplicate_ids = []
    n_bboxes = len(bboxes)
    for i in range(n_bboxes):
        if i in duplicate_ids:
            continue
        det_id_1, xmin_1, ymin_1, xmax_1, ymax_1, width_1, height_1, class_name_1, confidence_1 = bboxes[i]

        for j in range(i + 1, n_bboxes):
            if j in duplicate_ids:
                continue
            det_id_2, xmin_2, ymin_2, xmax_2, ymax_2, width_2, height_2, class_name_2, confidence_2 = bboxes[j]
            if det_id_1 == det_id_2:
                continue
            iou = getIOU([xmin_1, ymin_1, xmax_1, ymax_1], [xmin_2, ymin_2, xmax_2, ymax_2])
            if iou >= iou_thresh:
                if confidence_1 > confidence_2:
                    duplicate_ids.append(j)
                else:
                    duplicate_ids.append(i)
                    break
    n_duplicates = len(duplicate_ids)
    print('n_bboxes: {} n_duplicates: {}'.format(n_bboxes, n_duplicates))
    bboxes = [bbox for _id, bbox in enumerate(bboxes) if _id not in duplicate_ids]
    return bboxes


def main():
    # parser = argparse.ArgumentParser(description="CSV to TFRecord Converter")
    # parser.add_argument('--seq_paths', type=str, default='',
    #                     help='List of paths to image sequences')
    # parser.add_argument('--root_dir', type=str, default='',
    #                     help='Path to input files')
    # parser.add_argument('--csv_paths', type=str, default='',
    #                     help='List of paths to csv annotations')
    # parser.add_argument('--csv_root_dir', type=str, default='',
    #                     help='Path to csv files')
    # parser.add_argument('--output_path', type=str, default='',
    #                     help='Path to save the polled csv files')
    # parser.add_argument('--class_names_path', type=str, default='',
    #                     help='Path to file containing class names')
    # parser.add_argument('--shuffle_files', type=int, default=1,
    #                     help='shuffle files')
    # parser.add_argument('--n_frames', type=int, default=0,
    #                     help='n_frames')
    # parser.add_argument('--min_size', type=int, default=1,
    #                     help='min_size')
    # parser.add_argument('--fixed_ar', type=int, default=0,
    #                     help='pad images to have fixed aspect ratio')
    # parser.add_argument('--sampling_ratio', type=float, default=1.0,
    #                     help='proportion of images to include in the tfrecord file')
    # parser.add_argument('--random_sampling', type=int, default=0,
    #                     help='enable random sampling')
    # parser.add_argument('--inverted_sampling', type=int, default=0,
    #                     help='invert samples defined by the remaining sampling parameters')
    # parser.add_argument('--even_sampling', type=float, default=0.0,
    #                     help='use evenly spaced sampling (< 1 would draw samples from '
    #                          'only a fraction of the sequence; < 0 would invert the sampling)')
    # parser.add_argument('--allow_missing_detections', type=int, default=1,
    #                     help='allow_missing_detections')
    # parser.add_argument('--samples_per_class', type=int, default=0,
    #                     help='no. of samples to include per class; < 0 would sample from the end')
    # parser.add_argument('--only_sampling', type=int, default=0,
    #                     help='only_sampling')
    # parser.add_argument('--polling_type', type=int, default=0,
    #                     help='polling_type: '
    #                          '0: single highest confidence detection in each frame;'
    #                          '1: pool all detections and remove duplicates based on IOU;'
    #                          '')
    # parser.add_argument('--load_samples_root', type=str, default='', help='load_samples_root')
    #
    # parser.add_argument('--load_samples', type=int, default=0, help='load_samples')
    #
    # paramparse.fromParser(parser, 'PollingParams2')
    #
    # sys.exit()

    # params = parser.parse_args()

    params = PollingParams()
    paramparse.process(params)

    seq_paths = params.seq_paths
    root_dir = params.root_dir

    csv_paths = params.csv_paths
    csv_root_dir = params.csv_root_dir

    output_path = params.output_path
    n_frames = params.n_frames
    class_names_path = params.class_names_path
    min_size = params.min_size
    allow_missing_detections = params.allow_missing_detections

    inverted_sampling = params.inverted_sampling
    load_samples = params.load_samples
    load_samples_root = params.load_samples_root
    sampling_ratio = params.sampling_ratio
    random_sampling = params.random_sampling
    even_sampling = params.even_sampling
    samples_per_class = params.samples_per_class

    only_sampling = params.only_sampling
    polling_type = params.polling_type

    conf_thresh = params.conf_thresh
    discard_below_thresh = params.discard_below_thresh
    allow_seq_skipping = params.allow_seq_skipping

    sample_from_end = 0
    variable_sampling_ratio = 0
    if samples_per_class != 0:
        sampling_ratio = 0
        print('Using variable sampling ratio to include {} samples per class')
        variable_sampling_ratio = 1
        if samples_per_class < 0:
            samples_per_class = -samples_per_class
            print('Sampling from end')
            sample_from_end = 1

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

    class_names = {x.strip() for x in class_names}
    class_dict = {x: csv_id for (csv_id, x) in enumerate(class_names)}

    print('class_names: ', class_names)
    print('class_dict: ', class_dict)

    if seq_paths:
        if os.path.isfile(seq_paths):
            seq_paths = [x.strip() for x in open(seq_paths).readlines()]
        else:
            seq_paths = seq_paths.split(',')
        if root_dir:
            seq_paths = [os.path.join(root_dir, k) for k in seq_paths]
    elif root_dir:
        seq_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                     os.path.isdir(os.path.join(root_dir, name))]
        seq_paths.sort(key=sortKey)
    else:
        raise IOError('Either seq_paths or root_dir must be provided')
    seq_paths.sort(key=sortKey)
    n_seq = len(seq_paths)

    if csv_paths:
        if os.path.isfile(csv_paths):
            csv_paths = [x.strip() for x in open(csv_paths).readlines()]
        else:
            csv_paths = csv_paths.split(',')
        if csv_root_dir:
            csv_paths = [os.path.join(csv_root_dir, k) for k in csv_paths]
    elif csv_root_dir:
        csv_paths = [os.path.join(csv_root_dir, name) for name in os.listdir(csv_root_dir) if
                     os.path.isdir(os.path.join(csv_root_dir, name))]
        csv_paths.sort(key=sortKey)
    else:
        raise IOError('Either csv_paths or csv_root_dir must be provided')

    n_csv_paths = len(csv_paths)
    print('Polling over {} sets of detections:'.format(n_csv_paths))
    pprint(csv_paths)

    if conf_thresh:
        conf_thresh = [float(x) / 100.0 for x in conf_thresh.split(',')]
        n_conf_thresh = len(conf_thresh)
        if n_csv_paths != n_conf_thresh:
            raise IOError('Mismatch between n_csv_paths: {} and n_conf_thresh: {}'.format(
                n_csv_paths, n_conf_thresh))
        if any(k >= 1.0 for k in conf_thresh):
            raise IOError('One opr more invalid conf_thresh: {}'.format(pformat(conf_thresh)))
        print('conf_thresh: {}'.format(pformat(conf_thresh)))

    csv_files_dict = {}
    for csv_path in csv_paths:
        csv_files = [os.path.join(csv_path, name) for name in os.listdir(csv_path) if
                     os.path.isfile(os.path.join(csv_path, name)) and name.endswith('.csv')]
        csv_files.sort(key=sortKey)
        n_csv = len(csv_files)

        if n_csv != n_seq:
            msg = 'Mismatch between n_seq: {} and n_csv: {} for csv_path: {}'.format(
                n_seq, n_csv, csv_path)
            if allow_seq_skipping:
                print(msg)
            else:
                raise IOError(msg)
        csv_files_dict[csv_path] = csv_files

    img_ext = 'jpg'
    # csv_data = []
    total_samples = 0
    total_files = 0

    seq_to_src_files = {}
    class_to_n_files = {_class: 0 for _class in class_names}
    seq_to_n_files = {}

    def getClass(seq_path):
        for _class in class_names:
            if _class in os.path.basename(seq_path):
                return _class
        raise IOError('No class found for {}'.format(seq_path))

    seq_to_sampling_ratio = {k: sampling_ratio for k in seq_paths}
    seq_to_class = {k: getClass(k) for k in seq_paths}

    for idx, seq_path in enumerate(seq_paths):
        src_files = [os.path.join(seq_path, k) for k in os.listdir(seq_path) if
                     os.path.splitext(k.lower())[1][1:] == img_ext]
        src_files.sort(key=sortKey)

        n_files = len(src_files)

        seq_to_n_files[seq_path] = n_files
        seq_to_src_files[seq_path] = src_files
        class_to_n_files[seq_to_class[seq_path]] += n_files

        total_files += n_files

    print('class_to_n_files:')
    pprint(class_to_n_files)

    if variable_sampling_ratio:
        class_to_sampling_ratio = {k: float(samples_per_class) / class_to_n_files[k] for k in class_to_n_files}

        print('class_to_sampling_ratio:')
        pprint(class_to_sampling_ratio)

        seq_to_sampling_ratio = {
            k: class_to_sampling_ratio[seq_to_class[k]] for k in seq_paths
        }

    seq_to_samples = {}

    if len(load_samples) == 1:
        if load_samples[0] == 1:
            load_samples = ['seq_to_samples.txt', ]
        elif load_samples[0] == 0:
            load_samples = []

    if load_samples:
        # if load_samples == '1':
        #     load_samples = 'seq_to_samples.txt'
        print('load_samples: {}'.format(pformat(load_samples)))
        if load_samples_root:
            load_samples = [os.path.join(load_samples_root, k) for k in load_samples]
        print('Loading samples from : {}'.format(load_samples))
        for _f in load_samples:
            if os.path.isdir(_f):
                _f = os.path.join(_f, 'seq_to_samples.txt')
            with open(_f, 'r') as fid:
                curr_seq_to_samples = ast.literal_eval(fid.read())
                for _seq in curr_seq_to_samples:
                    if _seq in seq_to_samples:
                        seq_to_samples[_seq] += curr_seq_to_samples[_seq]
                    else:
                        seq_to_samples[_seq] = curr_seq_to_samples[_seq]

        print('Found samples for {} sequences'.format(len(seq_to_samples)))

    class_to_n_samples = {_class: 0 for _class in class_names}

    # Writer object for TFRecord creation
    if not output_path:
        output_path = os.path.join(os.path.dirname(csv_paths[0]), 'csv_polled')

    output_path += '_type_{}'.format(polling_type)
    print('output_path: ', output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    all_sampled_files = []
    valid_seq_paths = []

    skipped_seq = 0

    for idx, seq_path in enumerate(seq_paths):
        src_files = seq_to_src_files[seq_path]
        n_files = seq_to_n_files[seq_path]

        if load_samples:
            try:
                sampled_files = seq_to_samples[seq_path]
            except KeyError:
                sampled_files = []
            n_samples = len(sampled_files)
        else:
            sampling_ratio = seq_to_sampling_ratio[seq_path]
            n_samples = int(n_files * sampling_ratio)

            if sampling_ratio != 1.0:
                if random_sampling:
                    sampled_files = random.sample(src_files, n_samples)
                elif even_sampling:
                    if sampling_ratio > even_sampling:
                        raise SystemError('{} :: sampling_ratio: {} is greater than even_sampling: {}'.format(
                            seq_path, sampling_ratio, even_sampling))
                    sample_1_of_n = int(math.ceil(even_sampling / sampling_ratio))
                    end_file = int(n_files * even_sampling)

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

        if not sampled_files:
            msg = 'No sampled files found for {} with {} source files'.format(seq_path, n_files)
            if allow_seq_skipping:
                print('\n{}\n'.format(msg))
                skipped_seq += 1
                continue
            else:
                raise IOError(msg)

        # pprint(sampled_files)

        valid_seq_paths.append(seq_path)
        seq_to_samples[seq_path] = sampled_files
        actual_samples = len(sampled_files)
        class_to_n_samples[seq_to_class[seq_path]] += actual_samples
        total_samples += actual_samples

        all_sampled_files += sampled_files

        if only_sampling:
            continue

    seq_paths = valid_seq_paths
    n_seq = len(seq_paths)

    for csv_path in csv_files_dict:
        n_csv = len(csv_files_dict[csv_path])
        if n_csv != n_seq:
            msg = 'Mismatch between n_seq: {} and n_csv: {} for csv_path: {}'.format(
                n_seq, n_csv, csv_path)
            raise IOError(msg)

    for idx, seq_path in enumerate(seq_paths):

        seq_name = os.path.splitext(os.path.basename(seq_path))[0]
        src_files = seq_to_src_files[seq_path]
        n_files = seq_to_n_files[seq_path]
        sampled_files = seq_to_samples[seq_path]

        print('Processing sequence {}/{}: {} :: reading data for {}({})/{} images from {}'
              ' (total images: {}/{})'.format(idx + 1, n_seq, seq_name, actual_samples, n_samples,
                                              n_files, seq_path,
                                              total_samples, total_files))
        df_list = []
        for csv_path in csv_files_dict:
            csv_file = csv_files_dict[csv_path][idx]
            if seq_name not in csv_file:
                print('seq_paths:\n{}'.format(pformat(seq_paths)))
                raise IOError('Invalid csv_file: {} for seq {}: {}'.format(
                    csv_file, idx, seq_name))
            df = pd.read_csv(csv_file)
            df_list.append([df, csv_file, csv_path])

        # print('Getting csv data from:')
        # pprint([k[1] for k in df_list])

        csv_raw = []
        for file_path in sampled_files:
            filename = os.path.basename(file_path)

            # csv_data = []
            max_confidence = 0
            best_bbox = None

            # if not os.path.isfile(file_path):
            #     raise IOError('Image file not found: {}'.format(file_path))

            bboxes = []
            for csv_id in range(n_csv_paths):

                try:
                    df, csv_file, csv_path = df_list[csv_id]
                except ValueError as e:
                    print(df_list[csv_id])
                    sys.exit()

                if df.empty:
                    continue
                try:
                    df_bboxes = df.loc[df['filename'] == filename]
                except KeyError:
                    err_msg = '\nNo detections found for {} in {}\n'.format(file_path, csv_file)
                    # continue
                    if allow_missing_detections:
                        # print(err_msg)
                        continue
                    else:
                        raise IOError(err_msg)

                if df_bboxes.empty:
                    continue
                try:
                    _ = df_bboxes.iloc[0].loc['filename']
                except IndexError:
                    if allow_missing_detections:
                        print('df_bboxes:')
                        pprint(df_bboxes)
                        continue
                    else:
                        raise IOError(err_msg)

                n_df_bboxes = len(df_bboxes.index)

                for bbox_id in range(n_df_bboxes):
                    df_bbox = df_bboxes.iloc[bbox_id]
                    xmin = df_bbox.loc['xmin']
                    ymin = df_bbox.loc['ymin']
                    xmax = df_bbox.loc['xmax']
                    ymax = df_bbox.loc['ymax']
                    width = df_bbox.loc['width']
                    height = df_bbox.loc['height']
                    class_name = df_bbox.loc['class']
                    confidence = df_bbox.loc['confidence']

                    if conf_thresh:
                        if discard_below_thresh:
                            if confidence < conf_thresh[csv_id]:
                                continue
                            else:
                                confidence = (confidence - conf_thresh[csv_id]) / (1.0 - conf_thresh[csv_id])
                        else:
                            confidence = confidence / conf_thresh[csv_id]
                    bbox = (csv_id, xmin, ymin, xmax, ymax, width, height, class_name, confidence)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_bbox = bbox
                    bboxes.append(bbox)

                df_list[csv_id][0] = df.drop(df_bboxes.index[:n_df_bboxes])
                # csv_data.append((bboxes, csv_path))

            if best_bbox is None:
                pass
                # print('No boxes found for {}'.format(file_path))
            else:
                if polling_type == 0:
                    out_bboxes = [best_bbox, ]
                    # other boxes from highest confidence detector
                    out_bboxes += [k for k in bboxes if k[0] == best_bbox[0]]
                elif polling_type == 1:
                    out_bboxes = removeDuplicates(bboxes)
                elif polling_type == 2:
                    # all bboxes
                    out_bboxes = bboxes

                # n_out_bboxes = len(out_bboxes)
                # print('filename: {} out_bboxes: {} max_confidence: {}'.format(
                #     filename, n_out_bboxes, max_confidence))

                if max_confidence == 0:
                    raise SystemError('zero max_confidence encountered')

                for _bbox in out_bboxes:
                    csv_id, xmin, ymin, xmax, ymax, width, height, class_name, confidence = _bbox

                    if not discard_below_thresh:
                        confidence = confidence / max_confidence

                    if confidence > 1.0:
                        raise SystemError('Invalid confidence: {}'.format(confidence))

                    raw_data = {
                        'confidence': confidence,
                        'filename': filename,
                        'width': width,
                        'height': height,
                        'class': class_name,
                        'xmin': int(xmin),
                        'ymin': int(ymin),
                        'xmax': int(xmax),
                        'ymax': int(ymax),
                        'csv_id': int(csv_id),
                    }
                    csv_raw.append(raw_data)

        out_file_path = os.path.join(output_path, '{}.csv'.format(seq_name))
        pd.DataFrame(csv_raw).to_csv(out_file_path)

    print('Saved polled detections to: {}'.format(output_path))

    sampled_fname = os.path.join(output_path, 'sampled_files.txt')
    # print('Writing sampled_files list to: {}'.format(sampled_fname))
    with open(sampled_fname, 'w') as fid:
        # print('sampled_files:')
        pprint(all_sampled_files, fid)


if __name__ == '__main__':
    main()
