import os
import sys

sys.path.append('.')

import cv2
import numpy as np
import time

from Utilities import drawRegion, drawBox, col_rgb, CVConstants

try:
    import pyMTF

    mtf_available = 1
except ImportError as e:
    print('MTF unavailable: {}'.format(e))
    mtf_available = 0

from siamfc.SiamFC import SiamFC, SiamFCParams
from SiamMask.SiamMask import SiamMask, SiamMaskParams
from DaSiamRPN.DaSiamRPN import DaSiamRPN, DaSiamRPNParams

siamfc_available = 1


# try:
#     from siamfc.SiamFC import SiamFC
#
#     siamfc_available = 1
# except ImportError as e:
#     print('Siamese FC import error: {}'.format(e))
#     siamfc_available = 0


class PatchTrackerParams:
    """
    :type tracker_type: int
    :type mtf_cfg_dir: str
    :type cv_tracker_type: int
    :type show: int | bool
    :type save: int | bool
    :type box_color: str
    :type text_fmt: tuple(str, int, float, int, int)
    :type save_fmt: tuple(str, str, int)

    :param SiamFCParams siam_fc:
    :param SiamMaskParams siam_mask:
    :param DaSiamRPNParams da_siam_rpn:

    """

    def __init__(self):
        self.tracker_type = '2'
        self.mtf_cfg_dir = 'tracking/cfg/mtf'
        self.cv_tracker_type = 0

        self.show = 1
        self.convert_to_rgb = 0
        self.thickness = 2
        self.box_color = 'red'
        self.gt_color = 'green'
        self.resize_factor = 1.0
        self.show_text = 1
        self.text_fmt = ('green', 0, 5, 1.0, 1)
        self.save = 0
        self.save_fmt = ('avi', 'XVID', 30)
        self.save_dir = 'videos'

        self.pause_after_frame = 0

        self.siam_fc = SiamFCParams()
        self.siam_mask = SiamMaskParams()
        self.da_siam_rpn = DaSiamRPNParams()

        self.help = {
            'tracker_type': 'type of tracker to use:'
                            '0: OpenCV 3'
                            '1: MTF '
                            '2: Siamese FC '
                            '3: SiamMask '
                            '4: DaSiamRPN ',
            'mtf_cfg_dir': 'directory containing the cfg files for MTF',
            'cv_tracker_type': 'tracker type to use if use_mtf is disabled',
            'siam_fc': 'SiamFC tracker params',
            'siam_mask': 'SiamMask tracker params',
            'da_siam_rpn': 'DaSiamRPN tracker params',
            'show': 'show the tracked object location drawn on the input image',
            'convert_to_rgb': 'convert the image to RGB before showing it; this is sometimes needed if the raw frame is'
                              ' in BGR format so that it does not show correctly (blue and red channels are '
                              'interchanged)',
            'thickness': 'thickness of the bounding box lines drawn on the image',
            'box_color': 'color of the bounding box used to represent the tracked object location',
            'resize_factor': 'multiplicative factor by which the images are resized before being shown or saved',
            'show_text': 'write text in the top left corner of the image to indicate the frame number and FPS',
            'text_fmt': '(color, location, font, font_size, thickness) of the text used to '
                        'indicate the frame number and FPS; '
                        'Available fonts: '
                        '0: cv2.FONT_HERSHEY_SIMPLEX, '
                        '1: cv2.FONT_HERSHEY_PLAIN, '
                        '2: cv2.FONT_HERSHEY_DUPLEX, '
                        '3: cv2.FONT_HERSHEY_COMPLEX, '
                        '4: cv2.FONT_HERSHEY_TRIPLEX, '
                        '5: cv2.FONT_HERSHEY_COMPLEX_SMALL, '
                        '6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX ,'
                        '7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX; '
                        'Locations: 0: top left, 1: top right, 2: bottom right, 3: bottom left',
            'save': 'save the visualization result with tracked object location drawn on the'
                    ' input image as a video file',
            'save_fmt': '(extension, encoder, FPS) of the saved video',
            'save_dir': 'directory where to save the video',
        }


class PatchTracker:
    def __init__(self, params, logger, target_id, label='generic', show_only=False):
        """
        :type params: PatchTrackerParams
        :type logger: logging.logger
        :type target_id: int
        :rtype None
        """
        self._params = params
        self._logger = logger

        self.target_id = target_id
        self.label = label
        self.mtf_id = 0
        self.show_only = show_only

        self.is_created = False
        self.is_terminated = False
        self.is_initialized = False

        self.cv_tracker = None
        self.siamfc_tracker = None
        self.siam_mask_tracker = None
        self.da_siam_rpn_tracker = None

        self.tracker_type = None

        self.box_color = col_rgb[self._params.box_color]
        self.gt_color = col_rgb[self._params.gt_color]
        self.text_color = col_rgb[self._params.text_fmt[0]]
        self.text_font = CVConstants.fonts[self._params.text_fmt[2]]
        self.text_font_size = self._params.text_fmt[3]
        self.text_thickness = self._params.text_fmt[4]
        self.text_location = (5, 15)

        if cv2.__version__.startswith('2'):
            self.text_line_type = cv2.CV_AA
        else:
            self.text_line_type = cv2.LINE_AA
        if not self.show_only:
            if self._params.tracker_type in ('0', 'cv'):
                self.tracker_type = 0

                (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

                print('major_ver: {}'.format(major_ver))
                print('minor_ver: {}'.format(minor_ver))

                if int(major_ver) < 3:
                    self._logger.error('OpenCV trackers are not available')
                    return

                tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
                tracker_type = tracker_types[self._params.cv_tracker_type]

                self._logger.info('Using OpenCV {:s} tracker'.format(tracker_type))
                if int(minor_ver) < 3:
                    self.cv_tracker = cv2.Tracker_create(tracker_type)
                else:
                    if tracker_type == 'BOOSTING':
                        self.cv_tracker = cv2.TrackerBoosting_create()
                    if tracker_type == 'MIL':
                        self.cv_tracker = cv2.TrackerMIL_create()
                    if tracker_type == 'KCF':
                        self.cv_tracker = cv2.TrackerKCF_create()
                    if tracker_type == 'TLD':
                        self.cv_tracker = cv2.TrackerTLD_create()
                    if tracker_type == 'MEDIANFLOW':
                        self.cv_tracker = cv2.TrackerMedianFlow_create()
                    if tracker_type == 'GOTURN':
                        self.cv_tracker = cv2.TrackerGOTURN_create()
            elif self._params.tracker_type in ('1', 'mtf'):
                self.tracker_type = 1
                if not mtf_available:
                    self._logger.error('MTF is not available')
                    return
                self._logger.info('Using MTF tracker')
            elif self._params.tracker_type in ('2', 'siamfc'):
                self.tracker_type = 2
                if not siamfc_available:
                    self._logger.error('Siamese FC tracker is not available')
                    return
                self._logger.info('Using Siamese FC tracker')
                self.siamfc_tracker = SiamFC(self._params.siam_fc, label=self.label, target_id=self.target_id)
            elif self._params.tracker_type in ('3', 'siam_mask'):
                self.tracker_type = 3
                self._logger.info('Using SiamMask tracker')
                self.siam_mask_tracker = SiamMask(self._params.siam_mask, label=self.label, target_id=self.target_id)
            elif self._params.tracker_type in ('4', 'da_siam_rpn'):
                self.tracker_type = 4
                self._logger.info('Using DaSiamRPN tracker')
                self.da_siam_rpn_tracker = DaSiamRPN(self._params.da_siam_rpn, self._logger,
                                                     label=self.label, target_id=self.target_id)
            else:
                raise IOError('Invalid tracker_type: {}'.format(self._params.tracker_type))

        self.window_name = 'Target {:d} : Press space/esc to stop tracking, s/S to toggle pause'.format(self.target_id)

        if self._params.show:
            # window for displaying the tracking result
            cv2.namedWindow(self.window_name)

        self.curr_corners = np.zeros((2, 4), dtype=np.float64)
        self.out_bbox = None
        self.curr_mask_cropped = None
        self.curr_mask = None
        # self.curr_mask_pts = None
        self.score = 1
        self.is_created = True
        self.video_writer = None
        self.pause_after_frame = self._params.pause_after_frame

    def createTracker(self, init_frame, xmin, ymin, xmax, ymax):
        if self.tracker_type == 0:
            width = xmax - xmin + 1
            height = ymax - ymin + 1
            roi = (xmin, ymin, width, height)
            ok = self.cv_tracker.init(init_frame, roi)
            if not ok:
                self._logger.error('Tracker initialization was unsuccessful')
                return
        elif self.tracker_type == 1:
            # if len(init_frame.shape) == 3:
            #     init_frame_gs = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
            # else:
            #     init_frame_gs = init_frame

            init_corners = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ]
            init_corners = np.array(init_corners).T

            try:
                # initialize tracker with the first frame and the initial corners
                self.mtf_id = pyMTF.create(init_frame.astype(np.uint8), init_corners.astype(np.float64),
                                           self._params.mtf_cfg_dir)
                # print('mtf_id: ', self.mtf_id)
                # print('type(mtf_id): ', type(self.mtf_id))

                if not self.mtf_id:
                    tracker_created = False
                else:
                    tracker_created = True
            except:
                tracker_created = False

            if not tracker_created:
                self._logger.error('MTF tracker creation was unsuccessful')
                return
        elif self.tracker_type == 2:
            w = xmax - xmin
            h = ymax - ymin

            cx = xmin + w / 2.0
            cy = ymin + h / 2.0

            bbox = [cx, cy, w, h]
            self.siamfc_tracker.initialize(init_frame, bbox)
        elif self.tracker_type == 3:
            w = xmax - xmin
            h = ymax - ymin

            cx = xmin + w / 2.0
            cy = ymin + h / 2.0

            bbox = [cx, cy, w, h]
            self.siam_mask_tracker.initialize(init_frame, bbox)
        elif self.tracker_type == 4:
            w = xmax - xmin
            h = ymax - ymin

            cx = xmin + w / 2.0
            cy = ymin + h / 2.0

            bbox = [cx, cy, w, h]
            self.da_siam_rpn_tracker.initialize(init_frame, bbox)

    def initialize(self, init_frame, init_bbox):

        # extract the true corners in the first frame and place them into a 2x4 array
        xmin = init_bbox['xmin']
        xmax = init_bbox['xmax']
        ymin = init_bbox['ymin']
        ymax = init_bbox['ymax']

        shape = init_frame.shape
        # print('init_frame.shape: ', init_frame.shape)
        if len(shape) == 3:
            n_rows, n_cols, n_ch = shape
        else:
            n_rows, n_cols = shape

        if self._params.text_fmt[1] == 1:
            self.text_location = (n_cols - 100, 15)
        elif self._params.text_fmt[1] == 2:
            self.text_location = (n_cols - 100, n_rows - 15)
        elif self._params.text_fmt[1] == 3:
            self.text_location = (5, n_rows - 15)
        else:
            self.text_location = (5, 15)

        if not self.show_only:
            self.createTracker(init_frame, xmin, ymin, xmax, ymax)

        if self._params.save:
            time_str = time.strftime("%y%m%d_%H%M", time.localtime())
            save_fname = 'target_{:d}_{:s}.{:s}'.format(self.target_id, time_str, self._params.save_fmt[0])
            save_path = os.path.join(self._params.save_dir, save_fname)
            if not os.path.exists(self._params.save_dir):
                os.makedirs(self._params.save_dir)
            frame_size = (init_frame.shape[1], init_frame.shape[0])
            if self._params.resize_factor != 1:
                frame_size = (int(frame_size[0] * self._params.resize_factor),
                              int(frame_size[1] * self._params.resize_factor))
            self.video_writer = cv2.VideoWriter()
            if cv2.__version__.startswith('3'):
                self.video_writer.open(filename=save_path, apiPreference=cv2.CAP_FFMPEG,
                                       fourcc=cv2.VideoWriter_fourcc(*self._params.save_fmt[1]),
                                       fps=int(self._params.save_fmt[2]), frameSize=frame_size)
            else:
                self.video_writer.open(filename=save_path, fourcc=cv2.cv.CV_FOURCC(*self._params.save_fmt[1]),
                                       fps=self._params.save_fmt[2], frameSize=frame_size)

            if not self.video_writer.isOpened():
                self._logger.error('Video file {:s} could not be opened'.format(save_path))
                return
            print('Saving tracking output to {:s}'.format(save_path))

        self.is_initialized = True

    def update(self, frame, frame_id, file_path=None, gt_bbox=None):
        start_time = time.clock()

        if gt_bbox is not None:
            gt_xmin = gt_bbox['xmin']
            gt_xmax = gt_bbox['xmax']
            gt_ymin = gt_bbox['ymin']
            gt_ymax = gt_bbox['ymax']
            gt_corners = np.zeros((2, 4), dtype=np.float64)
            gt_corners[:, 0] = (gt_xmin, gt_ymin)
            gt_corners[:, 1] = (gt_xmax, gt_ymin)
            gt_corners[:, 2] = (gt_xmax, gt_ymax)
            gt_corners[:, 3] = (gt_xmin, gt_ymax)
        else:
            gt_corners = None

        if self.tracker_type == 0:
            ok, bbox = self.cv_tracker.update(frame)
            if not ok:
                self._logger.error('Tracker update was unsuccessful')
                self.out_bbox = None
                self.is_terminated = True
                return

            xmin, ymin, width, height = bbox
            xmax = xmin + width - 1
            ymax = ymin + height - 1
            self.curr_corners[:, 0] = (xmin, ymin)
            self.curr_corners[:, 1] = (xmax, ymin)
            self.curr_corners[:, 2] = (xmax, ymax)
            self.curr_corners[:, 3] = (xmin, ymax)
        elif self.tracker_type == 1:
            # if len(frame.shape) == 3:
            #     frame_gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # else:
            #     frame_gs = frame
            success = pyMTF.getRegion(frame, self.curr_corners, self.mtf_id)
            if not success:
                self._logger.error('Tracker update was unsuccessful')
                self.out_bbox = None
                self.is_terminated = True
                return
        elif self.tracker_type == 2:
            bbox = self.siamfc_tracker.update(frame, frame_id, file_path)
            # cx, cy, w, h = bbox
            # xmin = cx + w/2.0
            # ymin = cy + h/2.0

            xmin, ymin, w, h = bbox

            xmax = xmin + w
            ymax = ymin + h
            self.curr_corners[:, 0] = (xmin, ymin)
            self.curr_corners[:, 1] = (xmax, ymin)
            self.curr_corners[:, 2] = (xmax, ymax)
            self.curr_corners[:, 3] = (xmin, ymax)
        elif self.tracker_type == 3:
            bbox = self.siam_mask_tracker.update(frame)
            xmin, ymin, w, h = bbox

            xmax = xmin + w
            ymax = ymin + h
            self.curr_corners[:, 0] = (xmin, ymin)
            self.curr_corners[:, 1] = (xmax, ymin)
            self.curr_corners[:, 2] = (xmax, ymax)
            self.curr_corners[:, 3] = (xmin, ymax)

            mask = self.siam_mask_tracker.mask
            # self.curr_mask_pts = self.siam_mask_tracker.mask_pts

            cv2.imshow('mask', mask)

            if gt_bbox is not None:
                mask_cropped = mask[gt_ymin:gt_ymax, gt_xmin:gt_xmax, ...]
                cv2.imshow('mask_cropped', mask_cropped)
            else:
                mask_cropped = mask[int(ymin):int(ymax), int(xmin):int(xmax), ...]
                cv2.imshow('mask_cropped', mask_cropped)

            self.curr_mask = (mask * 255).astype(np.uint8)
            self.curr_mask_cropped = mask_cropped
            # self.curr_mask = (mask_cropped * 255).astype(np.uint8)
            self.score = self.siam_mask_tracker.score

        elif self.tracker_type == 4:
            bbox = self.da_siam_rpn_tracker.update(frame)
            xmin, ymin, w, h = bbox

            xmax = xmin + w
            ymax = ymin + h
            self.curr_corners[:, 0] = (xmin, ymin)
            self.curr_corners[:, 1] = (xmax, ymin)
            self.curr_corners[:, 2] = (xmax, ymax)
            self.curr_corners[:, 3] = (xmin, ymax)

        end_time = time.clock()
        # compute the tracking fps
        fps = 1.0 / (end_time - start_time)

        if self._params.show:
            self.show(frame, self.curr_corners, frame_id, fps,
                      gt_corners=gt_corners,
                      # mask_img=self.curr_mask
                      )

        # print('curr_corners: ', curr_corners)
        xmin = int(self.curr_corners[0, 0])
        ymin = int(self.curr_corners[1, 0])
        xmax = int(self.curr_corners[0, 2])
        ymax = int(self.curr_corners[1, 2])

        self.out_bbox = dict(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
        return fps

        # print('out_bbox: ', self.out_bbox)

    def show(self, frame, corners, frame_id, fps=None, remote_fps=None, gt_corners=None, mask_img=None):
        if self._params.convert_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # draw the tracker location
        kw_args = {
            'color': self.box_color,
            'thickness': self._params.thickness,
            'mask_img': mask_img,
        }
        # drawBox(frame, corners, **kw_args)
        drawRegion(frame, corners, **kw_args)


        if self._params.show_text:
            # write statistics (error and fps) to the image
            header_text = "frame {:d}".format(frame_id)
            if fps is not None:
                header_text = "{:s} {:5.2f} fps".format(header_text, fps)
            if remote_fps is not None:
                header_text = "{:s} {:5.2f} remote_fps".format(header_text, remote_fps)

        if gt_corners is not None:
            xmin, ymin = corners[:, 0]
            xmax, ymax = corners[:, 2]
            bb = [xmin, ymin, xmax, ymax]

            xmin, ymin = gt_corners[:, 0]
            xmax, ymax = gt_corners[:, 2]
            bb_gt = [xmin, ymin, xmax, ymax]

            bi = [max(bb[0], bb_gt[0]), max(bb[1], bb_gt[1]), min(bb[2], bb_gt[2]),
                  min(bb[3], bb_gt[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1

            if iw <= 0 or ih <= 0:
                iou = 0
            else:
                # compute overlap (IoU) = area of intersection / area of union
                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bb_gt[2] - bb_gt[0]
                                                                  + 1) * (
                             bb_gt[3] - bb_gt[1] + 1) - iw * ih
                iou = iw * ih / ua

            header_text = "{:s} {:5.2f} iou".format(header_text, iou)
            drawRegion(frame, gt_corners, self.gt_color, self._params.thickness)

        if self._params.show_text:
            cv2.putText(frame, header_text, self.text_location,
                        self.text_font, self.text_font_size, self.text_color, self.text_thickness,
                        self.text_line_type)

        if self._params.resize_factor != 1:
            frame = cv2.resize(frame, (0, 0), fx=self._params.resize_factor,
                               fy=self._params.resize_factor)
        # display the image
        cv2.imshow(self.window_name, frame)

        if self.video_writer is not None:
            self.video_writer.write(frame)

        key = cv2.waitKey(1 - self.pause_after_frame)
        if key == 27 or key == 32:
            self.is_terminated = True
        if key == ord('s') or key == ord('S'):
            self.pause_after_frame = 1 - self.pause_after_frame

    def close(self):
        if self._params.show:
            cv2.destroyWindow(self.window_name)
            if self.tracker_type == 3:
                cv2.destroyWindow('mask')
                cv2.destroyWindow('mask_cropped')

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if not self.show_only:
            if self.tracker_type == 1:
                pyMTF.remove(self.mtf_id)
            elif self.tracker_type == 2:
                self.siamfc_tracker.close()
            elif self.tracker_type == 3:
                self.siam_mask_tracker.close()
            elif self.tracker_type == 4:
                self.da_siam_rpn_tracker.close()
