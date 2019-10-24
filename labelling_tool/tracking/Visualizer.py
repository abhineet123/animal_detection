import sys

sys.path.append('../..')

try:
    import Utilities as utils
except ImportError:
    try:
        from labelling_tool.tracking import Utilities as utils
    except ImportError:
        import tracking.Utilities as utils

import numpy as np
import cv2
import os


class VisualizerParams:
    """
    :type mode: (int, int, int)
    :type tracked_cols: tuple(str,)
    :type lost_cols: tuple(str,)
    :type inactive_cols: tuple(str,)
    :type det_cols: tuple(str,)
    :type ann_cols: tuple(str,)
    :type text_fmt: tuple(str, int, float, int, int)
    :type gate_fmt: tuple(str, float, float, int, int)
    :type pause_after_frame: bool
    :type input: InputParams
    :type objects: ObjectsParams
    :type help: {str:str}
    """

    def __init__(self):
        """
        :rtype: None
        """
        self.mode = 0
        self.tracked_cols = ('green',)
        self.lost_cols = ('red',)
        self.inactive_cols = ('cyan',)
        self.det_cols = ('black',)
        self.ann_cols = ('green', 'red', 'blue', 'cyan', 'magenta', 'purple')

        self.convert_to_rgb = 0
        self.write_frame_id = 1
        self.pause_after_frame = 1

        self.show_trajectory = 0
        self.box_thickness = 2
        self.traj_thickness = 2
        self.resize_factor = 1.0
        # self._text.fmt = ('green', 'black', 0, 5, 1.0, 1)
        self.text = utils.CVText()
        self.gate_fmt = ('black', 2.0, 5, 1.0, 1)

        self.show = 1
        self.save = 0
        self.save_fmt = ('avi', 'XVID', 30)
        self.save_dir = 'log'
        self.save_prefix = ''

        self.help = {
            'mode': '0: annotations, 1: detections, 2: tracked',
            'tracked_cols': 'bounding box colors in which to show the tracking result for objects in tracked state; '
                            'if there are more objects than the number of specified colors, modulo indexing is used',
            'lost_cols': 'bounding box colors in which to show the tracking result for objects in lost state',
            'inactive_cols': 'bounding box colors in which to show the tracking result for objects in inactive state',
            'det_cols': 'bounding box colors in which to show the detections',
            'ann_cols': 'bounding box colors in which to show the annotations',
            'convert_to_rgb': 'convert the image to RGB before showing it; this is sometimes needed if the raw frame is'
                              ' in BGR format so that it does not show correctly (blue and red channels are '
                              'interchanged)',
            'write_frame_id': 'Write the current free mighty in the frames',
            'pause_after_frame': 'pause execution after each frame till a key is pressed to continue;'
                                 'Esc: exit the program'
                                 'Spacebar: toggle this parameter',
            'show_trajectory': 'show the trajectory of bounding boxes with associated unique IDs by drawing lines '
                               'connecting their centers across consecutive frames',
            'box_thickness': 'thickness of lines used to draw the bounding boxes',
            'traj_thickness': 'thickness of lines used to draw the trajectories',
            'resize_factor': 'multiplicative factor by which the images are resized before being shown or saved',

            'gate_fmt': '(color, thickness, font, font_size, font_thickness) of the lines and labels used '
                        'for showing the gates',
            'show': 'Show the images with drawn objects; this can be disabled when running in batch mode'
                    ' or on a system without GUI; the output can instead be saved as a video file',
            'save': 'Save the images with drawn objects as video files',
            'save_prefix': 'Prefix to be added to the name of the saved video files',
            'save_dir': 'Directory in which to save the video files',
            'save_fmt': '3 element tuple to specify the (extension, FOURCC format string, fps) of the saved video file;'
                        'refer http://www.fourcc.org/codecs.php for a list of valid FOURCC strings; '
                        'extension can be one of [jpg, bmp, png] to write to an image sequence instead of a video file',
        }


class ImageWriter:
    def __init__(self, file_path, logger=None):
        self.file_path = file_path
        self.logger = logger
        split_path = os.path.splitext(file_path)
        self.save_dir = split_path[0]
        try:
            self.ext = split_path[1][1:]
        except IndexError:
            self.ext = 'jpg'
        if not self.ext:
            self.ext = 'jpg'

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.frame_id = 0
        self.filename = 'image{:06d}.{:s}'.format(self.frame_id, self.ext)
        if self.logger is None:
            self.logger = utils.write
        else:
            self.logger = self.logger.info
        self.logger('Saving images of type {:s} to {:s}\n'.format(self.ext, self.save_dir))

    def write(self, frame, frame_id=None, prefix=''):
        if frame_id is None:
            self.frame_id += 1
        else:
            self.frame_id = frame_id
        if prefix:
            self.filename = '{:s}.{:s}'.format(prefix, self.ext)
        else:
            self.filename = 'image{:06d}.{:s}'.format(self.frame_id, self.ext)

        self.curr_file_path = os.path.join(self.save_dir, self.filename)
        cv2.imwrite(self.curr_file_path, frame, (cv2.IMWRITE_JPEG_QUALITY, 100))

    def release(self):
        pass


class Visualizer:
    """
    :type params: VisualizerParams
    :type logger: logging.RootLogger | logging.Logger
    :type traj_data: list[dict{int:list[int]}]
    """

    def __init__(self, params, logger, class_labels):
        self._params = params
        self._logger = logger
        self._class_labels = class_labels

        self._objects = [None] * 3
        self._pause = self._params.pause_after_frame

        # lost and inactive states to be shown in the same color as the tracked state
        # or alternatively, the color is independent of the state
        self._show_lost = 0
        self._show_inactive = 0
        if len(self._params.lost_cols) == 0:
            self._params.lost_cols = self._params.tracked_cols
        elif len(self._params.lost_cols) == 1 and self._params.lost_cols[0] == 'none':
            self._show_lost = 0
        if len(self._params.inactive_cols) == 0:
            self._params.inactive_cols = self._params.tracked_cols
        elif len(self._params.inactive_cols) == 1 and self._params.inactive_cols[0] == 'none':
            self._show_inactive = 0

        self.writer = None

        self.traj_data = [{}, None, {}]

        self._text = self._params.text

        # self._text.color = utils.col_rgb[self._params.text_fmt[0]]
        # if self._params.text_fmt[1]:
        #     self._text.bkg_color = utils.col_rgb[self._params.text_fmt[1]]
        # else:
        #     self._text.bkg_color = None
        # self._text.font = utils.CVConstants.fonts[self._params.text_fmt[3]]
        # self._text.font_size = self._params.text_fmt[4]
        # self._text.thickness = self._params.text_fmt[5]
        # self._text.location = (5, 15)
        # if cv2.__version__.startswith('2'):
        #     self._text.line_type = cv2.CV_AA
        # else:
        #     self._text.line_type = cv2.LINE_AA

        self.image_exts = ['jpg', 'bmp', 'png']
        self.win_name = 'Visualizer'
        self._quit = 0

    def initialize(self, save_fname_templ, frame_size, pause_after_frame=None):
        """
        :type save_fname_templ: str
        :type frame_size: tuple(int, int)
        :rtype: bool
        """
        # n_cols, n_rows = frame_size
        # print('n_cols: {:d}', n_cols)
        # print('n_rows: {:d}', n_rows)

        # if self._params.text_fmt[2] == 1:
        #     self._text.location = (n_cols - 200, 15)
        # elif self._params.text_fmt[2] == 2:
        #     self._text.location = (n_cols - 200, n_rows - 15)
        # elif self._params.text_fmt[2] == 3:
        #     self._text.location = (5, n_rows - 15)
        # else:
        #     self._text.location = (5, 15)

        self.win_name = '{} :: Press Space to pause/continue and Esc to exit'.format(save_fname_templ)
        if pause_after_frame is not None:
            self._pause = pause_after_frame

        if not self._params.save:
            return True

        if self._params.save_prefix:
            save_fname_templ = '{:s}_{:s}'.format(self._params.save_prefix, save_fname_templ)

        if self._params.mode == 0:
            save_postfix = 'ann'
            save_type = 'annotations'
        elif self._params.mode == 1:
            save_postfix = 'det'
            save_type = 'detections'
        elif self._params.mode == 2:
            save_postfix = 'res'
            save_type = 'tracking_result'

        if not os.path.exists(self._params.save_dir):
            os.makedirs(self._params.save_dir)

        save_fname = '{:s}_{:s}_{:s}.{:s}'.format(
            save_fname_templ, save_postfix, utils.getDateTime(), self._params.save_fmt[0])
        save_path = os.path.join(self._params.save_dir, save_fname)
        if self._params.save_fmt[0] in self.image_exts:
            writer = ImageWriter(save_path, self._logger)
            self._logger.info('Saving {:s} image sequence to {:s}'.format(save_type, save_path))
        else:
            if self._params.resize_factor != 1:
                frame_size = (int(frame_size[0] * self._params.resize_factor),
                              int(frame_size[1] * self._params.resize_factor))
            writer = cv2.VideoWriter()
            if cv2.__version__.startswith('3'):
                writer.open(filename=save_path, apiPreference=cv2.CAP_FFMPEG,
                            fourcc=cv2.VideoWriter_fourcc(*self._params.save_fmt[1]),
                            fps=int(self._params.save_fmt[2]), frameSize=frame_size)
            else:
                writer.open(filename=save_path, fourcc=cv2.cv.CV_FOURCC(*self._params.save_fmt[1]),
                            fps=self._params.save_fmt[2], frameSize=frame_size)
            if not writer.isOpened():
                self._logger.error('Video file {:s} could not be opened'.format(save_path))
                return False
            self._logger.info('Saving {:s} video to {:s}'.format(save_type, save_path))

        self.writer = writer
        return True

    def update(self, frame_id, frame, frame_data, masks=(), label='', crop_size=()):
        """
        :type frame_id: int
        :type frame: np.ndarray
        :type frame_data: np.ndarray
        :rtype: bool
        """
        if frame is None:
            self._logger.error('Frame is None')
            return False

        if frame.dtype != np.dtype('uint8'):
            self._logger.error('Invalid frame type')
            return False

        if len(frame.shape) == 3:
            if self._params.convert_to_rgb:
                curr_frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                curr_frame_disp = np.copy(frame)
        else:
            curr_frame_disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        img_text = ''
        if self._params.write_frame_id:
            img_text = 'frame {:d}'.format(frame_id)

        if label:
            img_text = '{} {}'.format(label, img_text) if img_text else label

        if img_text:
            # self._logger.info('img_text: {}'.format(img_text))
            self._text.put(curr_frame_disp, img_text)
        # annotations
        ann_frame = np.copy(curr_frame_disp)
        ann_data = frame_data
        for ann_id in range(ann_data.shape[0]):
            target_id = int(ann_data[ann_id, 1])
            if target_id < 0:
                col_id = -(1 + target_id) % len(self._params.ann_cols)
            else:
                col_id = (target_id - 1) % len(self._params.ann_cols)
            if self._params.show_trajectory:
                if target_id not in self.traj_data[2].keys():
                    self.traj_data[2][target_id] = []
                traj_point = np.array([ann_data[ann_id, 2] + ann_data[ann_id, 4] / 2.0,
                                       ann_data[ann_id, 3] + ann_data[ann_id, 5]])
                self.traj_data[2][target_id].append(traj_point)
                utils.drawTrajectory(ann_frame, self.traj_data[2][target_id], color=self._params.ann_cols[col_id],
                                     thickness=self._params.traj_thickness)
            kw_args = {
                'color': self._params.ann_cols[col_id],
                '_id': target_id, 'thickness': self._params.box_thickness,
                'label': self._class_labels[ann_data[ann_id, 6]]
            }
            if masks:
                kw_args['mask'] = masks[ann_id]

            utils.drawBox(ann_frame, ann_data[ann_id, 2:6], **kw_args)

        if crop_size:
            crop_w, crop_h = crop_size
            x, y, w, h = ann_data[ann_id, 2:6]
            cx, cy = x + w / 2.0, y + h / 2.0,
            minx, miny = int(cx - crop_w / 2.0), int(cy - crop_h / 2.0)
            maxx, maxy = int(cx + crop_w / 2.0), int(cy + crop_h / 2.0)

            if minx < 0:
                diff_x = -minx
                minx = 0

            if miny < 0:
                diff_y = -miny
                miny = 0

            img_h, img_w = ann_frame.shape[:2]

            print('ann_frame.shape before: {}'.format(ann_frame.shape))



            maxx = min(img_w - 1, maxx)
            maxy = min(img_h - 1, maxy)

            ann_frame = ann_frame[miny:maxy, minx:maxx, ...]

            print('ann_frame.shape after: {}'.format(ann_frame.shape))

        if self._params.resize_factor != 1:
            ann_frame = cv2.resize(ann_frame, (0, 0), fx=self._params.resize_factor,
                                   fy=self._params.resize_factor)
        if self.writer is not None:
            self.writer.write(ann_frame)
        if self._params.show:
            cv2.imshow(self.win_name, ann_frame)
            key = cv2.waitKey(1 - self._pause) % 256
            if key == 27:
                return False
            elif key == ord('q'):
                self._quit = 1
                return False
            elif key == 32:
                self._pause = 1 - self._pause
        else:
            return ann_frame

        return True

    def close(self):
        if self._params.show:
            cv2.destroyWindow(self.win_name)

        if self.writer is not None:
            self.writer.release()
