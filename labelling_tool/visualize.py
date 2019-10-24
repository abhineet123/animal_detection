import os, sys
import pandas as pd
import numpy as np
import logging
import paramparse
import cv2

from tracking.Visualizer import Visualizer, VisualizerParams, ImageWriter
from tracking.Utilities import processArguments2, stackImages_ptf, getDateTime
# from tracking.Server import ServerParams
from libs.frames_readers import get_frames_reader
from libs.pascal_voc_io import PascalVocReader
from libs.shape import Shape

# from argparse import Namespace

sys.path.append('..')
from tf_api.utilities import processArguments, sortKey, resizeAR


def visualize(vis_params, logger, img_path, csv_path, class_dict,
              init_frame_id=0, n_frames=0, request_roi=None,
              generator_mode=0, enable_masks=0, label='',
              only_boxes=0, crop_size=()):
    """

    :param vis_params:
    :param logger:
    :param img_path:
    :param csv_path:
    :param class_dict:
    :param init_frame_id:
    :param n_frames:
    :param request_roi:
    :param generator_mode:
    :return:
    """
    global _pause, _quit
    save_fname_templ = os.path.splitext(os.path.basename(img_path))[0]

    # csv_path = os.path.join(img_path, 'annotations.csv')

    df = pd.read_csv(csv_path)

    frames_reader = get_frames_reader(img_path, save_as_bin=False)
    if request_roi is not None:
        frames_reader.setROI(request_roi)
    class_labels = dict((v, k) for k, v in class_dict.items())

    if generator_mode:
        vis_params.show = 0
        vis_params.save = 0

    visualizer = Visualizer(vis_params, logger, class_labels)
    init_frame = frames_reader.get_frame(init_frame_id)

    height, width, _ = init_frame.shape
    frame_size = width, height
    visualizer.initialize(save_fname_templ, frame_size, _pause)

    if n_frames <= 0:
        n_frames = frames_reader.num_frames
    print('Reading {:d} images from {:s}...'.format(n_frames, img_path))

    for frame_id in range(init_frame_id, n_frames):
        try:
            curr_frame = frames_reader.get_frame(frame_id)
        except IOError as e:
            print('{}'.format(e))
            break

        if only_boxes:
            curr_frame = np.zeros_like(curr_frame)

        file_path = frames_reader.get_file_path()
        if file_path is None:
            print('Visualization is only supported on image sequence data')
            return

        filename = os.path.basename(file_path)
        multiple_instance = df.loc[df['filename'] == filename]
        # Total # of object instances in a file
        n_bboxes = len(multiple_instance.index)
        # Remove from df (avoids duplication)
        df = df.drop(multiple_instance.index[:n_bboxes])

        frame_data = []
        masks = []

        generic_target_id = -1

        if enable_masks:
            filename = os.path.basename(file_path)
            xml_path = os.path.join(img_path, 'annotations', os.path.splitext(filename)[0] + '.xml')
            if not os.path.isfile(xml_path):
                print('{} :: annotations xml file not found: {}'.format(filename, xml_path))
                continue
            xml_reader = PascalVocReader(xml_path)
            shapes = xml_reader.getShapes()
            n_shapes = len(shapes)

            if n_shapes != n_bboxes:
                raise IOError('Mismatch between n_bboxes in xml: {} and csv: {}'.format(n_shapes, n_bboxes))

        for box_id in range(n_bboxes):

            bbox = multiple_instance.iloc[box_id]
            try:
                target_id = bbox['target_id']
            except KeyError:
                target_id = generic_target_id
                generic_target_id -= 1

            xmin = bbox.loc['xmin']
            ymin = bbox.loc['ymin']
            xmax = bbox.loc['xmax']
            ymax = bbox.loc['ymax']
            class_name = bbox.loc['class']

            try:
                class_id = class_dict[str(class_name)]
            except KeyError:
                print('Ignoring annotation with invalid class: {}'.format(class_name))
                continue

            width = xmax - xmin
            height = ymax - ymin

            curr_frame_data = [frame_id, target_id, xmin, ymin, width, height, class_id]

            if enable_masks:
                mask = shapes[box_id][-2]
                if mask is not None:
                    _contour_pts = Shape.getContourPts(mask)
                    masks.append(_contour_pts)

            frame_data.append(curr_frame_data)

        frame_data = np.asarray(frame_data)
        res = visualizer.update(frame_id, curr_frame, frame_data, masks, label, crop_size)
        if generator_mode:
            yield res
        # elif not res:
        #     break

    _quit = visualizer._quit
    _pause = visualizer._pause

    visualizer.close()
    frames_reader.close()


class VisParams:
    def __init__(self):
        self.cfg = ('',)
        self.batch_size = 1
        self.class_names_path = '../labelling_tool/data//predefined_classes_orig.txt'
        self.codec = 'H264'
        self.csv_paths = ''
        self.csv_root_dir = ''
        self.data_type = 'annotations'
        self.enable_masks = 0
        self.fps = 20
        self.grid_size = ''
        self.img_ext = 'png'
        self.labels = []
        self.load_path = ''
        self.map_folder = ''
        self.n_classes = 4
        self.n_frames = 0
        self.n_vis = 1
        self.root_dir = ''
        self.save = 0
        self.save_dir = 'vis'
        self.save_file_name = ''
        self.save_fmt = 'jpg'
        self.save_video = 1
        self.seq_paths = ''
        self.seq_prefix = ''
        self.show_img = 1
        self.vis_size = ''
        self.only_boxes = 0
        self.crop_size = ''
        self.vis = VisualizerParams()

        self.help = {}


def main():
    global _pause, _quit
    params = VisParams()
    paramparse.process(params)

    # _args = [k for k in sys.argv[1:] if not k.startswith('vis.')]
    # vis_args = ['--{}'.format(k.replace('vis.', '')) for k in sys.argv[1:] if k.startswith('vis.')]

    # processArguments(_args, params)
    # params = _params

    seq_paths = params.seq_paths
    root_dir = params.root_dir
    csv_paths = params.csv_paths
    csv_root_dir = params.csv_root_dir
    class_names_path = params.class_names_path
    data_type = params.data_type
    n_frames = params.n_frames
    seq_prefix = params.seq_prefix
    n_vis = params.n_vis
    vis_size = params.vis_size
    enable_masks = params.enable_masks
    show_img = params.show_img
    save = params.save
    save_fmt = params.save_fmt
    save_dir = params.save_dir
    labels = params.labels
    grid_size = params.grid_size
    only_boxes = params.only_boxes
    crop_size = params.crop_size

    if crop_size:
        crop_size = tuple([int(x) for x in crop_size.split('x')])
        print('Cropping a region of size {}x{} around the box'.format(*crop_size))
    else:
        crop_size = ()

    if grid_size:
        grid_size = [int(k) for k in grid_size.split('x')]
        print('Using a grid size of {}x{}'.format(*grid_size))
    else:
        grid_size = None

    # params = Namespace(**params)

    if vis_size:
        vis_size = [int(x) for x in vis_size.split('x')]

    # get parameters
    # _params = ServerParams()
    # _params.processArguments()

    # print('vis_args: ', _params.vis.__dict__)
    # processArguments2(_params, vis_args)

    # print('_params: ', _params)

    # setup logger
    logging_fmt = '%(levelname)s::%(module)s::%(funcName)s::%(lineno)s :  %(message)s'
    logging_level = logging.INFO
    # logging_level = logging.DEBUG
    # logging_level = PROFILE_LEVEL_NUM
    logging.basicConfig(level=logging_level, format=logging_fmt)
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)

    if seq_paths:
        if os.path.isfile(seq_paths):
            seq_paths = [x.strip() for x in open(seq_paths).readlines() if x.strip()]
        else:
            seq_paths = seq_paths.split(',')
        if root_dir:
            seq_paths = [os.path.join(root_dir, name) for name in seq_paths]

    elif root_dir:
        seq_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                     os.path.isdir(os.path.join(root_dir, name))]
        seq_paths.sort(key=sortKey)
    else:
        raise IOError('Either seq_paths or root_dir must be provided')

    if csv_paths:
        if os.path.isfile(csv_paths):
            csv_paths = [x.strip() for x in open(csv_paths).readlines() if x.strip()]
        else:
            csv_paths = csv_paths.split(',')
        if csv_root_dir:
            csv_paths = [os.path.join(csv_root_dir, name) for name in csv_paths]
    elif csv_root_dir:
        csv_paths = [os.path.join(csv_root_dir, name) for name in os.listdir(csv_root_dir) if
                     os.path.isfile(os.path.join(csv_root_dir, name)) and name.endswith('.csv')]
        csv_paths.sort(key=sortKey)
    else:
        csv_paths = [os.path.join(seq_path, data_type + '.csv') for seq_path in seq_paths]

    seq_path_ids = []

    if seq_prefix:
        seq_path_ids = [_id for _id, seq_path in enumerate(seq_paths) if
                        os.path.basename(seq_path).startswith(seq_prefix)]
        seq_paths = [seq_paths[_id] for _id in seq_path_ids]
        csv_paths = [csv_paths[_id] for _id in seq_path_ids]

    n_seq, n_csv = len(seq_paths), len(csv_paths)
    if n_seq != n_csv:
        raise IOError('Mismatch between image {} and annotation {} lengths'.format(n_seq, n_csv))

    class_names = open(class_names_path, 'r').readlines()
    class_dict = {x.strip(): i for (i, x) in enumerate(class_names)}
    print('class_dict: ', class_dict)
    print('labels: ', labels)

    if n_vis > 0:
        if save:
            save_fname = '{:s}_{:s}.{:s}'.format(
                save_dir, getDateTime(), save_fmt)
            save_path = os.path.join('log', save_fname)
            writer = ImageWriter(save_path, _logger)
            _logger.info('Saving {:s} image sequence to {:s}'.format(save_fmt, save_path))

        if n_seq % n_vis != 0:
            raise AssertionError('n_seq: {} not multiple of n_vis: {}'.format(n_seq, n_vis))
        n_groups = int(n_seq / n_vis)
        seq_id = 0
        label = ''
        for i in range(n_groups):

            vis_gen = []
            for j in range(n_vis):
                if labels:
                    label = labels[j]
                vis_gen.append(visualize(
                    params.vis, _logger, seq_paths[seq_id], csv_paths[seq_id], class_dict,
                    n_frames=n_frames, generator_mode=1, enable_masks=enable_masks,
                    label=label, only_boxes=only_boxes, crop_size=crop_size
                ))
                seq_id += 1
            for imgs in zip(*vis_gen):
                # img_stacked = np.hstack(imgs)
                stack_params = {
                    'grid_size': grid_size,
                    'preserve_order': 1
                }
                if crop_size:
                    stack_params['annotations'] = labels
                img_stacked = stackImages_ptf(imgs, **stack_params)
                img_stacked = cv2.cvtColor(img_stacked, cv2.COLOR_RGB2BGR)

                if vis_size:
                    img_stacked = resizeAR(img_stacked, vis_size[0], vis_size[1])
                if save:
                    writer.write(img_stacked)

                if show_img:
                    cv2.imshow('img_stacked', img_stacked)
                    key = cv2.waitKey(1 - _pause) % 256
                    if key == 27:
                        break
                    elif key == ord('q'):
                        _quit = 1
                        break
                    elif key == 32:
                        _pause = 1 - _pause
            if _quit:
                break
    if save:
        writer.release()

    else:
        for i in range(n_seq):
            visualize(params.vis, _logger, seq_paths[i], csv_paths[i], class_dict,
                      n_frames=n_frames, enable_masks=enable_masks)
            if _quit:
                break


if __name__ == '__main__':
    _quit = 0
    _pause = 1

    main()
