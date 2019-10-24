import numpy as np
import cv2
import subprocess
import filecmp
import os
import shutil
import sys
import time
# from ast import literal_eval
import json
import argparse

# from ..libs.shape import Shape

# RGB values for different colors
col_rgb = {
    'snow': (250, 250, 255),
    'snow_2': (233, 233, 238),
    'snow_3': (201, 201, 205),
    'snow_4': (137, 137, 139),
    'ghost_white': (255, 248, 248),
    'white_smoke': (245, 245, 245),
    'gainsboro': (220, 220, 220),
    'floral_white': (240, 250, 255),
    'old_lace': (230, 245, 253),
    'linen': (230, 240, 240),
    'antique_white': (215, 235, 250),
    'antique_white_2': (204, 223, 238),
    'antique_white_3': (176, 192, 205),
    'antique_white_4': (120, 131, 139),
    'papaya_whip': (213, 239, 255),
    'blanched_almond': (205, 235, 255),
    'bisque': (196, 228, 255),
    'bisque_2': (183, 213, 238),
    'bisque_3': (158, 183, 205),
    'bisque_4': (107, 125, 139),
    'peach_puff': (185, 218, 255),
    'peach_puff_2': (173, 203, 238),
    'peach_puff_3': (149, 175, 205),
    'peach_puff_4': (101, 119, 139),
    'navajo_white': (173, 222, 255),
    'moccasin': (181, 228, 255),
    'cornsilk': (220, 248, 255),
    'cornsilk_2': (205, 232, 238),
    'cornsilk_3': (177, 200, 205),
    'cornsilk_4': (120, 136, 139),
    'ivory': (240, 255, 255),
    'ivory_2': (224, 238, 238),
    'ivory_3': (193, 205, 205),
    'ivory_4': (131, 139, 139),
    'lemon_chiffon': (205, 250, 255),
    'seashell': (238, 245, 255),
    'seashell_2': (222, 229, 238),
    'seashell_3': (191, 197, 205),
    'seashell_4': (130, 134, 139),
    'honeydew': (240, 255, 240),
    'honeydew_2': (224, 238, 244),
    'honeydew_3': (193, 205, 193),
    'honeydew_4': (131, 139, 131),
    'mint_cream': (250, 255, 245),
    'azure': (255, 255, 240),
    'alice_blue': (255, 248, 240),
    'lavender': (250, 230, 230),
    'lavender_blush': (245, 240, 255),
    'misty_rose': (225, 228, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'dark_slate_gray': (79, 79, 49),
    'dim_gray': (105, 105, 105),
    'slate_gray': (144, 138, 112),
    'light_slate_gray': (153, 136, 119),
    'gray': (190, 190, 190),
    'light_gray': (211, 211, 211),
    'midnight_blue': (112, 25, 25),
    'navy': (128, 0, 0),
    'cornflower_blue': (237, 149, 100),
    'dark_slate_blue': (139, 61, 72),
    'slate_blue': (205, 90, 106),
    'medium_slate_blue': (238, 104, 123),
    'light_slate_blue': (255, 112, 132),
    'medium_blue': (205, 0, 0),
    'royal_blue': (225, 105, 65),
    'blue': (255, 0, 0),
    'dodger_blue': (255, 144, 30),
    'deep_sky_blue': (255, 191, 0),
    'sky_blue': (250, 206, 135),
    'light_sky_blue': (250, 206, 135),
    'steel_blue': (180, 130, 70),
    'light_steel_blue': (222, 196, 176),
    'light_blue': (230, 216, 173),
    'powder_blue': (230, 224, 176),
    'pale_turquoise': (238, 238, 175),
    'dark_turquoise': (209, 206, 0),
    'medium_turquoise': (204, 209, 72),
    'turquoise': (208, 224, 64),
    'cyan': (255, 255, 0),
    'light_cyan': (255, 255, 224),
    'cadet_blue': (160, 158, 95),
    'medium_aquamarine': (170, 205, 102),
    'aquamarine': (212, 255, 127),
    'dark_green': (0, 100, 0),
    'dark_olive_green': (47, 107, 85),
    'dark_sea_green': (143, 188, 143),
    'sea_green': (87, 139, 46),
    'medium_sea_green': (113, 179, 60),
    'light_sea_green': (170, 178, 32),
    'pale_green': (152, 251, 152),
    'spring_green': (127, 255, 0),
    'lawn_green': (0, 252, 124),
    'chartreuse': (0, 255, 127),
    'medium_spring_green': (154, 250, 0),
    'green_yellow': (47, 255, 173),
    'lime_green': (50, 205, 50),
    'yellow_green': (50, 205, 154),
    'forest_green': (34, 139, 34),
    'olive_drab': (35, 142, 107),
    'dark_khaki': (107, 183, 189),
    'khaki': (140, 230, 240),
    'pale_goldenrod': (170, 232, 238),
    'light_goldenrod_yellow': (210, 250, 250),
    'light_yellow': (224, 255, 255),
    'yellow': (0, 255, 255),
    'gold': (0, 215, 255),
    'light_goldenrod': (130, 221, 238),
    'goldenrod': (32, 165, 218),
    'dark_goldenrod': (11, 134, 184),
    'rosy_brown': (143, 143, 188),
    'indian_red': (92, 92, 205),
    'saddle_brown': (19, 69, 139),
    'sienna': (45, 82, 160),
    'peru': (63, 133, 205),
    'burlywood': (135, 184, 222),
    'beige': (220, 245, 245),
    'wheat': (179, 222, 245),
    'sandy_brown': (96, 164, 244),
    'tan': (140, 180, 210),
    'chocolate': (30, 105, 210),
    'firebrick': (34, 34, 178),
    'brown': (42, 42, 165),
    'dark_salmon': (122, 150, 233),
    'salmon': (114, 128, 250),
    'light_salmon': (122, 160, 255),
    'orange': (0, 165, 255),
    'dark_orange': (0, 140, 255),
    'coral': (80, 127, 255),
    'light_coral': (128, 128, 240),
    'tomato': (71, 99, 255),
    'orange_red': (0, 69, 255),
    'red': (0, 0, 255),
    'hot_pink': (180, 105, 255),
    'deep_pink': (147, 20, 255),
    'pink': (203, 192, 255),
    'light_pink': (193, 182, 255),
    'pale_violet_red': (147, 112, 219),
    'maroon': (96, 48, 176),
    'medium_violet_red': (133, 21, 199),
    'violet_red': (144, 32, 208),
    'violet': (238, 130, 238),
    'plum': (221, 160, 221),
    'orchid': (214, 112, 218),
    'medium_orchid': (211, 85, 186),
    'dark_orchid': (204, 50, 153),
    'dark_violet': (211, 0, 148),
    'blue_violet': (226, 43, 138),
    'purple': (240, 32, 160),
    'medium_purple': (219, 112, 147),
    'thistle': (216, 191, 216),
    'green': (0, 255, 0),
    'magenta': (255, 0, 255)
}


class MDPStates:
    inactive, active, tracked, lost = range(4)


class TrackingStatus:
    success, failure, unstable = range(1, 4)


PY_DIST = -1


class CVConstants:
    similarity_types = {
        -1: PY_DIST,
        0: cv2.TM_CCOEFF_NORMED,
        1: cv2.TM_SQDIFF_NORMED,
        2: cv2.TM_CCORR_NORMED,
        3: cv2.TM_CCOEFF,
        4: cv2.TM_SQDIFF,
        5: cv2.TM_CCORR
    }
    interp_types = {
        0: cv2.INTER_NEAREST,
        1: cv2.INTER_LINEAR,
        2: cv2.INTER_AREA,
        3: cv2.INTER_CUBIC,
        4: cv2.INTER_LANCZOS4
    }
    fonts = {
        0: cv2.FONT_HERSHEY_SIMPLEX,
        1: cv2.FONT_HERSHEY_PLAIN,
        2: cv2.FONT_HERSHEY_DUPLEX,
        3: cv2.FONT_HERSHEY_COMPLEX,
        4: cv2.FONT_HERSHEY_TRIPLEX,
        5: cv2.FONT_HERSHEY_COMPLEX_SMALL,
        6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    }
    line_types = {
        0: cv2.LINE_4,
        1: cv2.LINE_8,
        2: cv2.LINE_AA,
    }


class CVText:
    def __init__(self):
        self.color = 'white'
        self.bkg_color = 'black'
        self.location = 0
        self.font = 5
        self.size = 1
        self.thickness = 1
        self.line_type = 2
        self.offset = (5, 15)

        self.help = {
            'font': 'Available fonts: '
                    '0: cv2.FONT_HERSHEY_SIMPLEX, '
                    '1: cv2.FONT_HERSHEY_PLAIN, '
                    '2: cv2.FONT_HERSHEY_DUPLEX, '
                    '3: cv2.FONT_HERSHEY_COMPLEX, '
                    '4: cv2.FONT_HERSHEY_TRIPLEX, '
                    '5: cv2.FONT_HERSHEY_COMPLEX_SMALL, '
                    '6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX ,'
                    '7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX; ',
            'location': '0: top left, 1: top right, 2: bottom right, 3: bottom left; ',
            'bkg_color': 'should be empty for no background',
        }

    def put(self, img, text):
        """

        :param img:
        :param text:
        :return:
        """

        n_rows, n_cols = img.shape[:2]
        size = self.size
        color = col_rgb[self.color]
        if self.bkg_color:
            bkg_color = col_rgb[self.bkg_color]
        else:
            bkg_color = None
        font = CVConstants.fonts[self.font]
        line_type = CVConstants.line_types[self.line_type]

        x, y = self.offset

        if self.location == 1:
            location = (n_cols - 200, y)
        elif self.location == 2:
            location = (n_cols - 200, n_rows - y)
        elif self.location == 3:
            location = (x, n_rows - y)
        else:
            location = (x, y)

        text_offset_x, text_offset_y = location
        if bkg_color is not None:
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=self.size, thickness=self.thickness)[0]
            box_coords = ((text_offset_x, text_offset_y + 5), (text_offset_x + text_width, text_offset_y - text_height))
            cv2.rectangle(img, box_coords[0], box_coords[1], bkg_color, cv2.FILLED)

        cv2.putText(img, text, location, font, size, color, self.thickness, line_type)


class DebugParams:
    """
    :type write_state_info: bool | int
    :type write_to_bin: bool | int
    :type write_thresh: (int, int)
    :type cmp_root_dirs: (str, str)
    """

    def __init__(self):
        self.write_state_info = 0
        self.write_thresh = (0, 0)
        self.write_to_bin = 1
        self.memory_tracking = 0
        self.cmp_root_dirs = ('/root/mdp_log', 'log')
        self.help = {
            'write_state_info': 'write matrices containing the target state information to files '
                                'on disk (for debugging purposes)',
            'write_thresh': 'two element tuple to indicate the minimum (iter_id, frame_id) after which '
                            'to start writing and comparing state info',
            'write_to_bin': 'write the matrices to binary files instead of human readable ASCII text files',
            'memory_tracking': 'track memory usage to find leaks',
            'cmp_root_dirs': 'root directories where the data files to be compared are written',
        }


# overlaps between two sets of labeled objects, typically the annotations and the detections
class CrossOverlaps:
    """
    :type iou: list[np.ndarray]
    :type ioa_1: list[np.ndarray]
    :type ioa_2: list[np.ndarray]
    :type max_iou_1: np.ndarray
    :type max_iou_1_idx: np.ndarray
    :type max_iou_2: np.ndarray
    :type max_iou_2_idx: np.ndarray
    """

    def __init__(self):
        # intersection over union
        self.iou = None
        # intersection over area of object 1
        self.ioa_1 = None
        # intersection over area of object 2
        self.ioa_2 = None
        # max iou of each object in first set over all objects in second set from the same frame
        self.max_iou_1 = None
        # index of the object in the second set that corresponds to the maximum iou
        self.max_iou_1_idx = None
        # max iou of each object in second set over all objects in first set from the same frame
        self.max_iou_2 = None
        # index of the object in the first set that corresponds to the maximum iou
        self.max_iou_2_idx = None

    def compute(self, objects_1, objects_2, index_1, index_2, n_frames):
        """
        :type objects_1: np.ndarray
        :type objects_2: np.ndarray
        :type index_1: list[np.ndarray]
        :type index_2: list[np.ndarray]
        :type n_frames: int
        :rtype: None
        """
        # for each frame, contains a matrix that stores the overlap between each pair of
        # annotations and detections in that frame
        self.iou = [None] * n_frames
        self.ioa_1 = [None] * n_frames
        self.ioa_2 = [None] * n_frames

        self.max_iou_1 = np.zeros((objects_1.shape[0],))
        self.max_iou_2 = np.zeros((objects_2.shape[0],))

        self.max_iou_1_idx = np.full((objects_1.shape[0],), -1, dtype=np.int32)
        self.max_iou_2_idx = np.full((objects_2.shape[0],), -1, dtype=np.int32)

        for frame_id in range(n_frames):
            idx1 = index_1[frame_id]
            idx2 = index_2[frame_id]

            if idx1 is None or idx2 is None:
                continue

            boxes_1 = objects_1[idx1, :]
            n1 = boxes_1.shape[0]
            ul_1 = boxes_1[:, :2]  # n1 x 2
            size_1 = boxes_1[:, 2:]  # n1 x 2
            br_1 = ul_1 + size_1 - 1  # n1 x 2
            area_1 = np.multiply(size_1[:, 0], size_1[:, 1]).reshape((n1, 1))  # n1 x 1

            boxes_2 = objects_2[idx2, :]
            n2 = boxes_2.shape[0]
            ul_2 = boxes_2[:, :2]  # n2 x 2
            size_2 = boxes_2[:, 2:]  # n2 x 2
            br_2 = ul_2 + size_2 - 1  # n2 x 2
            area_2 = np.multiply(size_2[:, 0], size_2[:, 1]).reshape((n2, 1))  # n2 x 1

            ul_1_rep = np.tile(np.reshape(ul_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            ul_2_rep = np.tile(np.reshape(ul_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            ul_inter = np.maximum(ul_1_rep, ul_2_rep)  # n2 x 2 x n1

            # box size is defined in terms of  no. of pixels
            br_1_rep = np.tile(np.reshape(br_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            br_2_rep = np.tile(np.reshape(br_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            br_inter = np.minimum(br_1_rep, br_2_rep)  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

            size_inter = br_inter - ul_inter + 1  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            size_inter[size_inter < 0] = 0  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
            area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1]).reshape((n1, n2))

            area_1_rep = np.tile(area_1, (1, n2))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
            area_2_rep = np.tile(area_2.transpose(), (n1, 1))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
            area_union = area_1_rep + area_2_rep - area_inter  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)

            # self.iou[frame_id] = np.divide(area_inter, area_union).reshape((n1, n2), order='F')  # n1 x n2
            # self.ioa_1[frame_id] = np.divide(area_inter, area_1_rep).reshape((n1, n2), order='F')  # n1 x n2
            # self.ioa_2[frame_id] = np.divide(area_inter, area_2_rep).reshape((n1, n2), order='F')  # n1 x n2

            self.iou[frame_id] = np.divide(area_inter, area_union)  # n1 x n2
            self.ioa_1[frame_id] = np.divide(area_inter, area_1_rep)  # n1 x n2
            self.ioa_2[frame_id] = np.divide(area_inter, area_2_rep)  # n1 x n2

            max_idx_1 = np.argmax(self.iou[frame_id], axis=1)
            max_idx_2 = np.argmax(self.iou[frame_id], axis=0).transpose()

            self.max_iou_1[idx1] = self.iou[frame_id][np.arange(n1), max_idx_1]
            self.max_iou_2[idx2] = self.iou[frame_id][max_idx_2, np.arange(n2)]

            # indices wrt the overall object arrays rather than their frame-wise subsets
            self.max_iou_1_idx[idx1] = idx2[max_idx_1]
            self.max_iou_2_idx[idx2] = idx1[max_idx_2]


# overlaps between each labeled object in a set with all other objects in that set from the same frame
class SelfOverlaps:
    """
    :type iou: np.ndarray
    :type ioa: np.ndarray
    :type max_iou: np.ndarray
    :type max_ioa: np.ndarray
    """

    def __init__(self):
        # intersection over union
        self.iou = None
        # intersection over area of object
        self.ioa = None
        # max iou of each object over all other objects from the same frame
        self.max_iou = None
        # max ioa of each object over all other objects from the same frame
        self.max_ioa = None

        self.br = None
        self.areas = None

    def compute(self, objects, index, n_frames):
        """
        :type objects: np.ndarray
        :type index: list[np.ndarray]
        :type n_frames: int
        :rtype: None
        """
        self.iou = [None] * n_frames
        self.ioa = [None] * n_frames

        self.max_ioa = np.zeros((objects.shape[0],))
        self.areas = np.zeros((objects.shape[0],))
        self.br = np.zeros((objects.shape[0], 2))

        for frame_id in range(n_frames):
            if index[frame_id] is None:
                continue

            end_id = index[frame_id]
            boxes = objects[index[frame_id], :]

            n = boxes.shape[0]

            ul = boxes[:, :2]  # n x 2
            ul_rep = np.tile(np.reshape(ul, (n, 1, 2)), (1, n, 1))  # np(n x n x 2) -> std(n x 2 x n)
            ul_2_rep = np.tile(np.reshape(ul, (1, n, 2)), (n, 1, 1))  # np(n x n x 2) -> std(n x 2 x n)
            ul_inter = np.maximum(ul_rep, ul_2_rep)  # n x 2 x n

            size = boxes[:, 2:]  # n1 x 2
            br = ul + size - 1  # n x 2

            # size_ = boxes[:, 2:]  # n x 2
            # br = ul + size_ - 1  # n x 2
            br_rep = np.tile(np.reshape(br, (n, 1, 2)), (1, n, 1))  # np(n x n x 2) -> std(n x 2 x n)
            br_2_rep = np.tile(np.reshape(br, (1, n, 2)), (n, 1, 1))  # np(n x n x 2) -> std(n x 2 x n)
            br_inter = np.minimum(br_rep, br_2_rep)  # n x 2 x n

            size_inter = br_inter - ul_inter + 1  # np(n x n x 2) -> std(n x 2 x n)
            size_inter[size_inter < 0] = 0
            # np(n x n x 1) -> std(n x 1 x n)
            area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1])

            area = np.multiply(size[:, 0], size[:, 1]).reshape((n, 1))  # n1 x 1
            # area = np.multiply(size_[:, :, 0], size_[:, :, 1])  # n x 1
            area_rep = np.tile(area, (1, n))  # np(n x n x 1) -> std(n x 1 x n)
            area_2_rep = np.tile(area.transpose(), (n, 1))  # np(n x n x 1) -> std(n x 1 x n)
            area_union = area_rep + area_2_rep - area_inter  # np(n x n x 1) -> std(n x 1 x n)

            # self.iou[frame_id] = np.divide(area_inter, area_union).reshape((n, n), order='F')  # n x n
            # self.ioa[frame_id] = np.divide(area_inter, area_rep).reshape((n, n), order='F')  # n x n

            self.iou[frame_id] = np.divide(area_inter, area_union)  # n x n
            self.ioa[frame_id] = np.divide(area_inter, area_rep)  # n x n

            # set box overlap with itself to 0
            idx = np.arange(n)
            self.ioa[frame_id][idx, idx] = 0
            self.iou[frame_id][idx, idx] = 0

            for i in range(n):
                invalid_idx = np.flatnonzero(np.greater(br[i, 1], br[:, 1]))
                self.ioa[frame_id][i, invalid_idx] = 0

            self.max_ioa[index[frame_id]] = np.amax(self.ioa[frame_id], axis=1)

            self.areas[index[frame_id]] = area.reshape((n,))
            self.br[index[frame_id], :] = br


def computeOverlapsMulti(iou, ioa_1, ioa_2, objects_1, objects_2, logger=None):
    """
    :type iou: np.ndarray | None
    :type ioa_1: np.ndarray | None
    :type ioa_2: np.ndarray | None
    :type object_1: np.ndarray
    :type objects_2: np.ndarray
    :type logger: logging.RootLogger | None
    :rtype: None
    """
    # handle annoying singletons
    if len(objects_1.shape) == 1:
        objects_1 = objects_1.reshape((1, 4))

    if len(objects_2.shape) == 1:
        objects_2 = objects_2.reshape((1, 4))

    n1 = objects_1.shape[0]
    n2 = objects_2.shape[0]

    ul_1 = objects_1[:, :2]  # n1 x 2
    ul_1_rep = np.tile(np.reshape(ul_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
    ul_2 = objects_2[:, :2]  # n2 x 2
    ul_2_rep = np.tile(np.reshape(ul_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

    size_1 = objects_1[:, 2:]  # n1 x 2
    size_2 = objects_2[:, 2:]  # n2 x 2

    # if logger is not None:
    #     logger.debug('objects_1.shape: %(1)s', {'1': objects_1.shape})
    #     logger.debug('objects_2.shape: %(1)s', {'1': objects_2.shape})
    #     logger.debug('objects_1: %(1)s', {'1': objects_1})
    #     logger.debug('objects_2: %(1)s', {'1': objects_2})
    #     logger.debug('ul_1: %(1)s', {'1': ul_1})
    #     logger.debug('ul_2: %(1)s', {'1': ul_2})
    #     logger.debug('size_1: %(1)s', {'1': size_1})
    #     logger.debug('size_2: %(1)s', {'1': size_2})

    br_1 = ul_1 + size_1 - 1  # n1 x 2
    br_1_rep = np.tile(np.reshape(br_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
    br_2 = ul_2 + size_2 - 1  # n2 x 2
    br_2_rep = np.tile(np.reshape(br_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

    size_inter = np.minimum(br_1_rep, br_2_rep) - np.maximum(ul_1_rep, ul_2_rep) + 1  # n2 x 2 x n1
    size_inter[size_inter < 0] = 0
    # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1])

    area_1 = np.multiply(size_1[:, 0], size_1[:, 1]).reshape((n1, 1))  # n1 x 1
    area_1_rep = np.tile(area_1, (1, n2))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_2 = np.multiply(size_2[:, 0], size_2[:, 1]).reshape((n2, 1))  # n2 x 1
    area_2_rep = np.tile(area_2.transpose(), (n1, 1))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_union = area_1_rep + area_2_rep - area_inter  # n2 x 1 x n1

    if iou is not None:
        # write('iou.shape: {}\n'.format(iou.shape))
        # write('area_inter.shape: {}\n'.format(area_inter.shape))
        # write('area_union.shape: {}\n'.format(area_union.shape))
        iou[:] = np.divide(area_inter, area_union)  # n1 x n2
    if ioa_1 is not None:
        ioa_1[:] = np.divide(area_inter, area_1_rep)  # n1 x n2
    if ioa_2 is not None:
        ioa_2[:] = np.divide(area_inter, area_2_rep)  # n1 x n2


# compute overlap of a single object with one or more objects
# specialized version for greater speed
def computeOverlap(iou, ioa_1, ioa_2, object_1, objects_2, logger=None, debug=False):
    """
    :type iou: np.ndarray | None
    :type ioa_1: np.ndarray | None
    :type ioa_2: np.ndarray | None
    :type object_1: np.ndarray
    :type objects_2: np.ndarray
    :type logger: logging.RootLogger | None
    :rtype: None
    """

    n = objects_2.shape[0]

    ul_coord_1 = object_1[0, :2].reshape((1, 2))
    ul_coords_2 = objects_2[:, :2]  # n x 2
    ul_coords_inter = np.maximum(ul_coord_1, ul_coords_2)  # n x 2

    size_1 = object_1[0, 2:].reshape((1, 2))
    sizes_2 = objects_2[:, 2:]  # n x 2

    br_coord_1 = ul_coord_1 + size_1 - 1
    br_coords_2 = ul_coords_2 + sizes_2 - 1  # n x 2
    br_coords_inter = np.minimum(br_coord_1, br_coords_2)  # n x 2

    sizes_inter = br_coords_inter - ul_coords_inter + 1
    sizes_inter[sizes_inter < 0] = 0

    # valid_idx = np.flatnonzero((sizes_inter >= 0).all(axis=1))
    # valid_count = valid_idx.size
    # if valid_count == 0:
    #     if iou is not None:
    #         iou.fill(0)
    #     if ioa_1 is not None:
    #         ioa_1.fill(0)
    #     if ioa_2 is not None:
    #         ioa_2.fill(0)
    #     return

    areas_inter = np.multiply(sizes_inter[:, 0], sizes_inter[:, 1]).reshape((n, 1))  # n x 1

    # if logger is not None:
    #     logger.debug('object_1.shape: %(1)s', {'1': object_1.shape})
    #     logger.debug('objects_2.shape: %(1)s', {'1': objects_2.shape})
    #     logger.debug('object_1: %(1)s', {'1': object_1})
    #     logger.debug('objects_2: %(1)s', {'1': objects_2})
    #     logger.debug('ul_coord_1: %(1)s', {'1': ul_coord_1})
    #     logger.debug('ul_coords_2: %(1)s', {'1': ul_coords_2})
    #     logger.debug('size_1: %(1)s', {'1': size_1})
    #     logger.debug('sizes_2: %(1)s', {'1': sizes_2})
    #     logger.debug('areas_inter: %(1)s', {'1': areas_inter})
    #     logger.debug('sizes_inter: %(1)s', {'1': sizes_inter})

    areas_2 = None
    if iou is not None:
        # iou.fill(0)
        areas_2 = np.multiply(sizes_2[:, 0], sizes_2[:, 1]).reshape((n, 1))  # n x 1
        area_union = size_1[0, 0] * size_1[0, 1] + areas_2 - areas_inter
        # if logger is not None:
        #     logger.debug('iou.shape: %(1)s', {'1': iou.shape})
        #     logger.debug('area_union.shape: %(1)s', {'1': area_union.shape})
        #     logger.debug('area_union: %(1)s', {'1': area_union})
        iou[:] = np.divide(areas_inter, area_union)
    if ioa_1 is not None:
        # ioa_1.fill(0)
        ioa_1[:] = np.divide(areas_inter, size_1[0, 0] * size_1[0, 1])
    if ioa_2 is not None:
        # ioa_2.fill(0)
        if areas_2 is None:
            areas_2 = np.multiply(sizes_2[:, 0], sizes_2[:, 1])
        ioa_2[:] = np.divide(areas_inter, areas_2)
    if debug:
        logger.debug('paused')


# faster version for single frame operations
def computeSelfOverlaps(iou, ioa, boxes):
    """
    :type iou: np.ndarray | None
    :type ioa: np.ndarray | None
    :type boxes: np.ndarray
    :rtype: None
    """
    n = boxes.shape[0]

    ul = boxes[:, :2].reshape((n, 2))  # n x 2
    ul_rep = np.tile(np.reshape(ul, (n, 1, 2)), (1, n, 1))  # np(n x n x 2) -> std(n x 2 x n)
    ul_2_rep = np.tile(np.reshape(ul, (1, n, 2)), (n, 1, 1))  # np(n x n x 2) -> std(n x 2 x n)
    ul_inter = np.maximum(ul_rep, ul_2_rep)  # n x 2 x n

    sizes = boxes[:, 2:].reshape((n, 2))  # n1 x 2
    br = ul + sizes - 1  # n1 x 2
    # size_ = boxes[:, 2:]  # n x 2
    # br = ul + size_ - 1  # n x 2
    br_rep = np.tile(np.reshape(br, (n, 1, 2)), (1, n, 1))  # np(n x n x 2) -> std(n x 2 x n)
    br_2_rep = np.tile(np.reshape(br, (1, n, 2)), (n, 1, 1))  # np(n x n x 2) -> std(n x 2 x n)
    br_inter = np.minimum(br_rep, br_2_rep)  # n x 2 x n

    size_inter = br_inter - ul_inter + 1  # np(n x n x 2) -> std(n x 2 x n)
    size_inter[size_inter < 0] = 0
    # np(n x n x 1) -> std(n x 1 x n)
    area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1])

    area = np.multiply(sizes[:, 0], sizes[:, 1]).reshape((n, 1))  # n x 1
    area_rep = np.tile(area, (1, n))  # np(n x n x 1) -> std(n x 1 x n)
    area_2_rep = np.tile(area.transpose(), (n, 1))  # np(n x n x 1) -> std(n x 1 x n)
    area_union = area_rep + area_2_rep - area_inter  # n x 1 x n

    if iou is not None:
        iou[:] = np.divide(area_inter, area_union)  # n x n
        idx = np.arange(n)
        iou[idx, idx] = 0
    if ioa is not None:
        ioa[:] = np.divide(area_inter, area)  # n x n
        idx = np.arange(n)
        ioa[idx, idx] = 0


def logDebugMulti(logger, vars, names):
    log_str = ''
    log_dict = {}
    for i in range(len(vars)):
        log_str += '{:s}: %%({:d})s'.format(names[i], i + 1)
        log_dict['{:d}'.format(i + 1)] = vars[i]
    logger.debug(log_str, log_dict)


def drawRegion(img, corners, color, thickness=1, mask_img=None):
    # draw the bounding box specified by the given corners
    for i in range(4):
        p1 = (int(corners[0, i]), int(corners[1, i]))
        p2 = (int(corners[0, (i + 1) % 4]), int(corners[1, (i + 1) % 4]))
        cv2.line(img, p1, p2, color, thickness)

    # if mask_img is not None:
    #     # threshold = 128
    #     # _, mask_img = cv2.threshold(mask_img, threshold, 255, cv2.THRESH_BINARY)
    #
    #     if len(mask_img.shape) == 3:
    #         mask_img_gs = np.squeeze(mask_img[:, :, 0]).copy().astype(np.uint8)
    #     else:
    #         mask_img_gs = mask_img.copy().astype(np.uint8)
    #
    #
    #     ret = cv2.findContours(mask_img_gs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    #     _contour_pts, _ = ret
    #     if not _contour_pts:
    #         raise IOError('No contour_pts found in mask')
    #     mask_pts = list(np.squeeze(_contour_pts[0]))
    #
    #     xmin, ymin = corners[0, 0], corners[1, 0]
    #     mask_pts = [(x + xmin, y + ymin) for x, y in mask_pts]
    #
    #     n_pts = len(mask_pts)
    #     if not n_pts:
    #         raise IOError('No mask_pts found in mask')
    #     print('{} mask_pts found'.format(n_pts))
    #
    #     # cv2.imshow('mask_img', mask_img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyWindow('mask_img')
    #
    #     # mask = np.array(mask).reshape((-1, 1, 2)).astype(np.int32)
    #     # mask = np.array(mask)
    #     # print('mask: {}'.format(mask))
    #     cv2.drawContours(img, _contour_pts, -1, (255, 255, 255), thickness=thickness)


def drawBox(frame, box, _id=None, color='black', thickness=2, label=None, mask=()):
    """
    :type frame: np.ndarray
    :type box: np.ndarray
    :type _id: int | None
    :param color: indexes into col_rgb
    :type color: str
    :type label: str | None
    :type thickness: int
    :rtype: None
    """

    color = col_rgb[color]
    box = box.squeeze()
    pt1 = (int(box[0]), int(box[1]))
    pt2 = (int(box[0] + box[2]),
           int(box[1] + box[3]))
    cv2.rectangle(frame, pt1, pt2, color, thickness=thickness)
    if _id is not None:
        if cv2.__version__.startswith('2'):
            font_line_type = cv2.CV_AA
        else:
            font_line_type = cv2.LINE_AA

        if _id >= 0:
            box_str = '{}'.format(_id)
            if label is not None:
                box_str = '{}:{}'.format(box_str, label)
        else:
            if label is not None:
                box_str = '{}'.format(label)
            else:
                box_str = ''

        if box[1] > 10:
            y_loc = int(box[1] - 1)
        else:
            y_loc = int(box[3] - 1)
        if box_str:
            cv2.putText(frame, box_str, (int(box[0] - 1), y_loc), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, font_line_type)
        if mask:
            mask = np.array(mask).reshape((-1, 1, 2)).astype(np.int32)
            # mask = np.array(mask)
            # print('mask: {}'.format(mask))
            cv2.drawContours(frame, mask, -1, color, thickness=thickness)


def drawTrajectory(frame, trajectory, color='black', thickness=2):
    """
    :type frame: np.ndarray
    :type trajectory: list[np.ndarray]
    :param color: indexes into col_rgb
    :type color: str
    :type thickness: int
    :rtype: None
    """

    n_traj = len(trajectory)
    for i in range(1, n_traj):
        pt1 = tuple(trajectory[i - 1].astype(np.int64))
        pt2 = tuple(trajectory[i].astype(np.int64))
        try:
            cv2.line(frame, pt1, pt2, col_rgb[color], thickness=thickness)

        except TypeError:
            print('frame.dtype', frame.dtype)
            print('pt1', pt1)
            print('pt2', pt2)


def writeToFiles(root_dir, write_to_bin, entries):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    if write_to_bin:
        file_ext = 'bin'
    else:
        file_ext = 'txt'
    for entry in entries:
        array = entry[0]
        fname = '{:s}/{:s}.{:s}'.format(root_dir, entry[1], file_ext)
        if write_to_bin:
            dtype = entry[2]
            array.astype(dtype).tofile(open(fname, 'wb'))
        else:
            fmt = entry[3]
            np.savetxt(fname, array, delimiter='\t', fmt=fmt)


def compareFiles(read_from_bin, fnames, dirs=None, sync_id=-1, msg=''):
    """
    :type target: Target
    :type read_from_bin: bool | int
    :type fnames: list[str]
    :type dirs: (str, ...) | None
    :type sync_id: int
    :type msg: str
    :rtype: bool
    """
    if dirs is None:
        params = DebugParams()
        dirs = params.cmp_root_dirs

    # if files is None:
    #     files = (
    #         ('active_train_features', 'active/train_features', 0),
    #         ('active_train_labels', 'active/train_labels', 0),
    #         ('lost_train_features', 'lost/train_features', 0),
    #         ('lost_train_labels', 'lost/train_labels', 0),
    #         ('locations', 'templates/locations', 0),
    #         ('features', 'templates/features', 0),
    #         ('scores', 'templates/scores', 0),
    #         ('indices', 'templates/indices', 0),
    #         ('overlaps', 'templates/overlaps', 0),
    #         ('ratios', 'templates/ratios', 0),
    #         ('angles', 'templates/angles', 0),
    #         ('bb_overlaps', 'templates/bb_overlaps', 0),
    #         ('similarity', 'templates/similarity', 0),
    #         ('scores', 'templates/scores', 0),
    #         ('roi', 'templates/roi', 0),
    #         ('patterns', 'templates/patterns', 0),
    #         ('history_locations', 'history/locations', 0),
    #         ('history_scores', 'history/scores', 0),
    #         ('ids', 'history/ids', 0),
    #         ('frame_ids', 'history/frame_ids', 0),
    #         ('states', 'history/states', 0),
    #         ('lk_out', 'lkcv/lk_out', 1),
    #     )

    if read_from_bin:
        file_ext = 'bin'
    else:
        file_ext = 'txt'

    if sync_id >= 0:
        sync_fname = '{:s}/write_{:d}.sync'.format(dirs[0], sync_id)
        sys.stdout.write('{:s} Waiting for {:s}...'.format(msg, sync_fname))
        sys.stdout.flush()
        iter_id = 0
        while not os.path.isfile(sync_fname):
            time.sleep(0.5)
            # iter_id += 1
            # if iter_id==10:
            #     return False
            # sys.stdout.write('.')
            # sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        os.remove(sync_fname)

    files_are_same = True
    for fname in fnames:
        path_1 = '{:s}/{:s}.{:s}'.format(dirs[0], fname, file_ext)
        path_2 = '{:s}/{:s}.{:s}'.format(dirs[1], fname, file_ext)

        if not os.path.isfile(path_1):
            print('{:s} does not exist'.format(path_1))
            continue
        if not os.path.isfile(path_2):
            print('{:s} does not exist'.format(path_2))
            continue
        if not read_from_bin:
            subprocess.call('dos2unix -q {:s}'.format(path_1), shell=True)
            subprocess.call('dos2unix -q {:s}'.format(path_2), shell=True)
            subprocess.call('sed -i -e \'s/NaN/nan/g\' {:s}'.format(path_1), shell=True)
        if not filecmp.cmp(path_1, path_2):
            print('Files {:s} and {:s} are different'.format(path_1, path_2))
            files_are_same = False
            if not read_from_bin:
                subprocess.call('diff {:s} {:s} > {:s}/{:s}.diff'.format(
                    path_1, path_2, dirs[1], fname), shell=True)
    if not files_are_same:
        print('paused')
    if sync_id >= 0:
        sync_fname = '{:s}/read_{:d}.sync'.format(dirs[0], sync_id)
        open(sync_fname, 'w').close()
    return files_are_same


def resizeAR(src_img, width=0, height=0, return_factors=False,
             placement_type=0):
    src_height, src_width, n_channels = src_img.shape
    src_aspect_ratio = float(src_width) / float(src_height)

    if width <= 0 and height <= 0:
        raise AssertionError('Both width and height cannot be zero')
    elif height <= 0:
        height = int(width / src_aspect_ratio)
    elif width <= 0:
        width = int(height * src_aspect_ratio)

    aspect_ratio = float(width) / float(height)

    if src_aspect_ratio == aspect_ratio:
        dst_width = src_width
        dst_height = src_height
        start_row = start_col = 0
    elif src_aspect_ratio > aspect_ratio:
        dst_width = src_width
        dst_height = int(src_width / aspect_ratio)
        start_row = int((dst_height - src_height) / 2.0)
        if placement_type == 0:
            start_row = 0
        elif placement_type == 1:
            start_row = int((dst_height - src_height) / 2.0)
        elif placement_type == 2:
            start_row = int(dst_height - src_height)
        start_col = 0
    else:
        dst_height = src_height
        dst_width = int(src_height * aspect_ratio)
        start_col = int((dst_width - src_width) / 2.0)
        if placement_type == 0:
            start_col = 0
        elif placement_type == 1:
            start_col = int((dst_width - src_width) / 2.0)
        elif placement_type == 2:
            start_col = int(dst_width - src_width)
        start_row = 0

    dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8)

    dst_img[start_row:start_row + src_height, start_col:start_col + src_width, :] = src_img
    dst_img = cv2.resize(dst_img, (width, height))
    if return_factors:
        resize_factor = float(height) / float(dst_height)
        return dst_img, resize_factor, start_row, start_col
    else:
        return dst_img


def stackImages_ptf(img_list, grid_size=None, stack_order=0, borderless=1,
                    preserve_order=0, return_idx=0, annotations=None,
                    ann_fmt=CVText()):
    n_images = len(img_list)
    # print('grid_size: {}'.format(grid_size))

    if grid_size is None:
        n_cols = n_rows = int(np.ceil(np.sqrt(n_images)))
    else:
        n_rows, n_cols = grid_size
    target_ar = 1920.0 / 1080.0
    if n_cols <= n_rows:
        target_ar /= 2.0
    shape_img_id = 0
    min_ar_diff = np.inf
    img_heights = np.zeros((n_images,), dtype=np.int32)
    for _img_id in range(n_images):
        height, width = img_list[_img_id].shape[:2]
        img_heights[_img_id] = height
        img_ar = float(n_cols * width) / float(n_rows * height)
        ar_diff = abs(img_ar - target_ar)
        if ar_diff < min_ar_diff:
            min_ar_diff = ar_diff
            shape_img_id = _img_id

    img_heights_sort_idx = np.argsort(-img_heights)
    row_start_idx = img_heights_sort_idx[:n_rows]
    img_idx = img_heights_sort_idx[n_rows:]
    # print('img_heights: {}'.format(img_heights))
    # print('img_heights_sort_idx: {}'.format(img_heights_sort_idx))
    # print('img_idx: {}'.format(img_idx))

    # grid_size = [n_rows, n_cols]
    img_size = img_list[shape_img_id].shape
    height, width = img_size[:2]

    # grid_size = [n_rows, n_cols]
    # print 'img_size: ', img_size
    # print 'n_images: ', n_images
    # print 'grid_size: ', grid_size

    # print()
    stacked_img = None
    list_ended = False
    img_idx_id = 0
    inner_axis = 1 - stack_order
    stack_idx = []
    stack_locations = []
    start_row = 0
    curr_ann = ''
    for row_id in range(n_rows):
        start_id = n_cols * row_id
        curr_row = None
        start_col = 0
        for col_id in range(n_cols):
            img_id = start_id + col_id
            if img_id >= n_images:
                curr_img = np.zeros(img_size, dtype=np.uint8)
                list_ended = True
            else:
                if preserve_order:
                    _curr_img_id = img_id
                elif col_id == 0:
                    _curr_img_id = row_start_idx[row_id]
                else:
                    _curr_img_id = img_idx[img_idx_id]
                    img_idx_id += 1

                curr_img = img_list[_curr_img_id]
                if annotations:
                    curr_ann = annotations[_curr_img_id]
                stack_idx.append(_curr_img_id)
                # print(curr_img.shape[:2])

                if curr_ann:
                    ann_fmt.put(curr_img, curr_ann)

                if not borderless:
                    curr_img = resizeAR(curr_img, width, height)
                if img_id == n_images - 1:
                    list_ended = True
            if curr_row is None:
                curr_row = curr_img
            else:
                if borderless:
                    curr_img = resizeAR(curr_img, 0, curr_row.shape[0])
                # print('curr_row.shape: ', curr_row.shape)
                # print('curr_img.shape: ', curr_img.shape)
                curr_row = np.concatenate((curr_row, curr_img), axis=inner_axis)

            curr_h, curr_w = curr_img.shape[:2]
            stack_locations.append((start_row, start_col, start_row + curr_h, start_col + curr_w))
            start_col += curr_w

        if stacked_img is None:
            stacked_img = curr_row
        else:
            if borderless:
                resize_factor = float(curr_row.shape[1]) / float(stacked_img.shape[1])
                curr_row = resizeAR(curr_row, stacked_img.shape[1], 0)
                new_start_col = 0
                for _i in range(n_cols):
                    _start_row, _start_col, _end_row, _end_col = stack_locations[_i - n_cols]
                    _w, _h = _end_col - _start_col, _end_row - _start_row
                    w_resized, h_resized = _w / resize_factor, _h / resize_factor
                    stack_locations[_i - n_cols] = (
                        _start_row, new_start_col, _start_row + h_resized, new_start_col + w_resized)
                    new_start_col += w_resized
            # print('curr_row.shape: ', curr_row.shape)
            # print('stacked_img.shape: ', stacked_img.shape)
            stacked_img = np.concatenate((stacked_img, curr_row), axis=stack_order)

        curr_h, curr_w = curr_row.shape[:2]
        start_row += curr_h

        if list_ended:
            break
    if return_idx:
        return stacked_img, stack_idx, stack_locations
    else:
        return stacked_img


def stackImages(img_list, stack_order=0, is_list=0):
    if is_list:
        n_images = len(img_list)
        img_shape = img_list[0].shape
    else:
        n_images = img_list.shape[0]
        img_shape = img_list.shape[1:]
    grid_size = int(np.ceil(np.sqrt(n_images)))
    stacked_img = None
    list_ended = False
    inner_axis = 1 - stack_order
    for row_id in range(grid_size):
        start_id = grid_size * row_id
        curr_row = None
        for col_id in range(grid_size):
            img_id = start_id + col_id
            if img_id >= n_images:
                curr_img = np.zeros(img_shape, dtype=np.uint8)
                list_ended = True
            else:
                if is_list:
                    curr_img = img_list[img_id]
                else:
                    curr_img = img_list[img_id, :, :].squeeze()
                if img_id == n_images - 1:
                    list_ended = True
            if curr_row is None:
                curr_row = curr_img
            else:
                if curr_row.shape[1 - inner_axis] != curr_img.shape[1 - inner_axis]:
                    if inner_axis == 1:
                        curr_img = resizeAR(curr_img, height=curr_row.shape[0])
                    else:
                        curr_img = resizeAR(curr_img, width=curr_row.shape[1])
                curr_row = np.concatenate((curr_row, curr_img), axis=inner_axis)
        if stacked_img is None:
            stacked_img = curr_row
        else:
            stacked_img = np.concatenate((stacked_img, curr_row), axis=stack_order)
        if list_ended:
            break
    return stacked_img


def stackImages1D(img_list, stack_order=0):
    # stack into a single row or column
    stacked_img = None
    inner_axis = 1 - stack_order
    for img in img_list:
        if stacked_img is None:
            stacked_img = img
        else:
            stacked_img = np.concatenate((stacked_img, img), axis=inner_axis)
    return stacked_img


def removeSubFolders(dir_name, sub_dir_prefix):
    folders = [os.path.join(dir_name, name) for name in os.listdir(dir_name) if
               name.startswith(sub_dir_prefix) and
               os.path.isdir(os.path.join(dir_name, name))]
    for folder in folders:
        shutil.rmtree(folder)


def write(str):
    sys.stdout.write(str)
    sys.stdout.flush()


def getDateTime():
    return time.strftime("%y%m%d_%H%M", time.localtime())


def parseSeqIDs(ids):
    out_ids = []
    if isinstance(ids, int):
        out_ids.append(ids)
    else:
        for _id in ids:
            if isinstance(_id, list):
                if len(_id) == 1:
                    out_ids.extend(range(_id[0]))
                if len(_id) == 2:
                    out_ids.extend(range(_id[0], _id[1]))
                elif len(_id) == 3:
                    out_ids.extend(range(_id[0], _id[1], _id[2]))
            else:
                out_ids.append(_id)
    return tuple(out_ids)


def listParseer(arg):
    arg_vals = arg.split(',')
    arg_vals_parsed = []
    for _val in arg_vals:
        try:
            _val_parsed = int(_val)
        except ValueError:
            try:
                _val_parsed = float(_val)
            except ValueError:
                _val_parsed = _val
        arg_vals_parsed.append(_val_parsed)
    return arg_vals_parsed


def addParamsToParser(parser, obj, root_name='', obj_name=''):
    members = tuple([attr for attr in dir(obj) if not callable(getattr(obj, attr))
                     and not attr.startswith("__")])
    if obj_name:
        if root_name:
            root_name = '{:s}.{:s}'.format(root_name, obj_name)
        else:
            root_name = '{:s}'.format(obj_name)
    for member in members:
        if member == 'help':
            continue

        default_val = getattr(obj, member)
        if isinstance(default_val, (int, bool, float, str, tuple, list, dict)):
            if root_name:
                member_param_name = '{:s}.{:s}'.format(root_name, member)
            else:
                member_param_name = '{:s}'.format(member)
            try:
                obj_help = obj.help[member]
            except KeyError:
                obj_help = member

            # print('member: {} help: {}'.format(member, obj_help))

            if isinstance(default_val, (tuple, list)):
                parser.add_argument('--{:s}'.format(member_param_name), type=listParseer,
                                    default=default_val, help=obj_help, metavar='')
            elif isinstance(default_val, dict):
                parser.add_argument('--{:s}'.format(member_param_name), type=json.loads, default=default_val,
                                    help=obj_help, metavar='')
            else:
                parser.add_argument('--{:s}'.format(member_param_name), type=type(default_val), default=default_val,
                                    help=obj_help, metavar='')
        else:
            # parameter is itself an instance of some other parmeter class so its members must
            # be processed recursively
            addParamsToParser(parser, getattr(obj, member), root_name, member)


def processArguments(obj, description="Labeling Tool"):
    parser = argparse.ArgumentParser(usage='%(prog)s [options]', description=description)
    addParamsToParser(parser, obj)
    args_in = []
    if hasattr(obj, 'cfg'):
        # check for a custom cfg file specified at command line
        if len(sys.argv) > 1 and '--cfg' in sys.argv[1]:
            _, arg_val = sys.argv[1].split('=')
            obj.cfg = arg_val
        if os.path.isfile(obj.cfg):
            # print('Reading parameters from {:s}'.format(obj.cfg))
            file_args = open(obj.cfg, 'r').readlines()
            # lines starting with # in the cfg file are regarded as comments and thus ignored
            file_args = ['--{:s}'.format(arg.strip()) for arg in file_args if arg.strip() and not arg.startswith('#')]
            args_in += file_args
            # command line arguments override those in the cfg file
    args_in = list(sys.argv[1:]) + args_in + list(sys.argv[1:])
    args = parser.parse_args(args_in)

    processArgsFromParser(obj, args)


def processArguments2(obj, args_in):
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')
    addParamsToParser(parser, obj)
    args = parser.parse_args(args_in)
    processArgsFromParser(obj, args)


def list2str(in_list):
    out_str = '{:d}'.format(in_list[0])
    for x in in_list[1:]:
        out_str = '{:s},{:d}'.format(out_str, x)
    return out_str


def str2list(in_str):
    if not in_str:
        return None
    split_str = in_str.strip().split(',')
    # print('in_str: ', in_str)
    # print('split_str: ', split_str)
    return [int(x) for x in split_str]


def assignArg(obj, arg, id, val):
    if id >= len(arg):
        raise IOError('Invalid arg: {}'.format(arg))
        # return
    obj_attr = getattr(obj, arg[id])
    if isinstance(obj_attr, (int, bool, float, str, tuple, list, dict)):
        if val == '#':
            if isinstance(obj_attr, str):
                # empty string
                val = ''
            elif isinstance(obj_attr, tuple):
                # empty tuple
                val = ()
            elif isinstance(obj_attr, list):
                # empty list
                val = []
        setattr(obj, arg[id], val)
    else:
        # parameter is itself an instance of some other parmeter class so its members must
        # be processed recursively
        assignArg(obj_attr, arg, id + 1, val)


def processArgsFromParser(obj, args):
    members = vars(args)
    for key in members.keys():
        val = members[key]
        key_parts = key.split('.')
        assignArg(obj, key_parts, 0, val)


def scalarToString(val, add_quotes=False):
    if isinstance(val, (int, bool)):
        return '{:d}'.format(int(val))
    elif isinstance(val, float):
        return '{:f}'.format(val)
    elif isinstance(val, str):
        if add_quotes:
            return '"{:s}"'.format(val)
        else:
            return val
    print('Invalid scalar: ', val)
    return None


def tupleToString(vals):
    _str = ''
    for val in vals:
        if isinstance(val, (int, bool, float, str)):
            _str = '{:s}{:s},'.format(_str, scalarToString(val, True))
        elif isinstance(val, tuple):
            _str = '{:s}{:s},'.format(_str, tupleToString(val))
        elif isinstance(val, dict):
            _str = '{:s}{:s},'.format(_str, dictToString(val))
    return '({:s})'.format(_str)


def dictToString(vals):
    _str = '{{'
    for key in vals.keys():
        val = vals[key]
        key_str = scalarToString(key)
        if isinstance(val, (int, bool, float, str)):
            _str = '{:s}{:s}:{:s},'.format(_str, key_str, scalarToString(val))
        elif isinstance(val, tuple):
            _str = '{:s}{:s}:{:s},'.format(_str, key_str, tupleToString(val))
        elif isinstance(val, dict):
            _str = '{:s}{:s}:{:s},'.format(_str, key_str, dictToString(val))
    _str += '}}'
    return _str


def getIntersectionArea(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the dimensions of intersection rectangle
    height = (yB - yA + 1)
    width = (xB - xA + 1)

    if height > 0 and width > 0:
        return height * width
    return 0
