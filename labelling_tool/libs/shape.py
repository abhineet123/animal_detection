#!/usr/bin/python
# -*- coding: utf-8 -*-


# try:
from PyQt5.QtGui import *
from PyQt5.QtCore import *
# except ImportError:
#     from PyQt4.QtGui import *
#     from PyQt4.QtCore import *

import time
import os
import sys
import cv2
import numpy as np
from pprint import pprint
from collections import OrderedDict
from PIL import Image, ImageDraw
from skimage import feature, filters
from scipy import ndimage as ndi

from libs.lib import distance, col_rgb

DEFAULT_LINE_COLOR_GROUND_TRUTH = QColor(0, 255, 0, 128)
DEFAULT_LINE_COLOR_OBJECT_DETECTOR = QColor(255, 215, 0, 128)
DEFAULT_LINE_COLOR_TRACKER = QColor(0, 255, 255, 128)
DEFAULT_LINE_COLOR_SINGLE_TRACKER = QColor(0, 255, 255, 128)
DEFAULT_FILL_COLOR = QColor(255, 255, 0, 128)
DEFAULT_SELECT_LINE_COLOR = QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QColor(255, 128, 255, 155)
DEFAULT_VERTEX_FILL_COLOR_GROUND_TRUTH = QColor(0, 255, 0, 255)
DEFAULT_HVERTEX_FILL_COLOR = QColor(255, 0, 0)
DEFAULT_LINE_COLOR_GATE = QColor(255, 255, 0, 128)


class Shape(object):
    P_SQUARE, P_ROUND = range(2)

    MOVE_VERTEX, NEAR_VERTEX = range(2)

    # The following class variables influence the drawing
    # of _all_ shape objects.
    line_color = DEFAULT_LINE_COLOR_GROUND_TRUTH
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR_GROUND_TRUTH
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    cols = {
        1: 'white',
        2: 'red',
        3: 'green',
        4: 'blue',
        5: 'magenta',
        6: 'cyan',
        7: 'yellow',
        8: 'forest_green',
        9: 'orange',
        10: 'purple',
    }

    mask_help = {
        'left_button+drag / ctrl+shift+drag': 'draw mask boundary',
        'shift+left_button+drag': 'delete mask within the given radius of the pointer',
        'shift+drag': 'show the region within which mask would be deleted when left_button is pressed',
        'ctrl+left_button': 'add a single point to the mask',
        'ctrl+drag': 'show where and how next point would be added if left_button is clicked',
        'alt+left_button+drag / arrow keys': 'move the entire mask around',
        'ctrl+right_button / a': 'run augmentation without background selection dialogue (unless necessary)',
        'alt+right_button / A': 'run augmentation with background selection dialogue',
        'ctrl+shift+right_button  / c': 'clean mask points to generate a single contour',
        'shift+right_button': 'delete all mask points',
        'right_button / b': 'start paint mode',
        'middle_button / enter': 'exit, clean mask points to generate a single contour '
                                 'and apply changes including bounding box',
        'alt+middle_button / ctrl+enter': 'exit, clean mask points to generate a single contour '
                                          'and apply changes excluding bounding box',
        'shift+middle_button / q': 'exit and discard changes',
        'ctrl+shift+middle_button / esc': 'exit and apply changes',
        'wheel': 'change the drawing window size',
        'shift+wheel / +, - / >, <': 'change the mask deletion radius',
        'ctrl+shift+wheel': 'change the magnified patch area without changing the magnified window size',
        'ctrl+alt+shift+wheel': 'change the magnified window size without changing the magnified patch area',
        'm': 'toggle the magnified window visibility',
    }

    def __init__(self, label=None, line_color=None, difficult=False, bbox_source="ground_truth",
                 id_number=None, is_gate=False, score=-1, n_intersections=0, mask=None, mask_img=None
                 ):
        self.label = label
        self.points = []
        self.fill = False
        self.selected = False
        self.difficult = difficult
        self.bbox_source = bbox_source
        self.id_number = id_number
        self.score = score

        self.mask = mask
        if self.mask is None:
            self.mask = []
        self.mask_img = mask_img

        if self.label == 'gate':
            is_gate = True

        self.is_gate = is_gate

        if self.is_gate:
            self.max_points = 2
            self.label = 'gate'
            self.has_text = True
        else:
            self.has_text = False
            self.max_points = 4

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }
        self._closed = False
        self.set_colors(line_color)
        self.is_hidden = False
        # intersections between gates and targets
        self.n_intersections = n_intersections

        if self.is_gate and self.n_intersections > 0:
            print('Setting n_intersections for gate {:d} to {:d}'.format(self.id_number, self.n_intersections))

    def set_colors(self, line_color=None):
        if line_color:
            self.line_color = line_color
        elif self.is_gate:
            self.line_color = DEFAULT_LINE_COLOR_GATE
        else:
            self.line_color = dict(
                ground_truth=DEFAULT_LINE_COLOR_GROUND_TRUTH,
                object_detector=DEFAULT_LINE_COLOR_OBJECT_DETECTOR,
                tracker=DEFAULT_LINE_COLOR_TRACKER,
                single_object_tracker=DEFAULT_LINE_COLOR_SINGLE_TRACKER,
            )[self.bbox_source]
        r, g, b, f = self.line_color.getRgb()
        self._vertex_fill_color = QColor(r, g, b, min(f * 2, 255))
        self.fill_color = QColor(r, g, b, min(f // 2, 255))

    # def getNearestPt(pts, pt, dist_thresh, get_diff=0, only_end_pt=0):
    #     if not pts:
    #         if get_diff:
    #             # print('returning -1, 1')
    #             return -1, 1
    #         # print('returning -1')
    #         return -1
    #     n_pts = len(pts)
    #     x, y = pt
    #     dist = [(x - _x) ** 2 + (y - _y) ** 2 if _f else np.inf for _x, _y, _f in pts]
    #     min_dist = min(dist)
    #
    #     # print('min_dist: {}'.format(min_dist))
    #     # print('get_diff: {}'.format(get_diff))
    #
    #     dist_thresh *= dist_thresh
    #
    #     if dist_thresh > 0 and min_dist > dist_thresh:
    #         if get_diff:
    #             # print('returning -1, 1')
    #             return -1, 1
    #         # print('returning -1')
    #         return -1
    #     min_id = dist.index(min_dist)
    #     if only_end_pt:
    #         end_id_1, end_id_2 = getNearestEndPts(min_id, pts, n_pts)
    #         diff_1, diff_2 = abs(min_id - end_id_1), abs(min_id - end_id_2)
    #         if diff_1 < diff_2:
    #             min_id = end_id_1
    #         else:
    #             min_id = end_id_2
    #
    #     # print('min_id: {}'.format(min_id))
    #
    #     if get_diff:
    #         if min_id < n_pts - 1:
    #             next_pt_x, next_pt_y, next_f = pts[min_id + 1]
    #             # print('next_pt_x: {}'.format(next_pt_x))
    #             if not next_f:
    #                 # print('returning {}, 1'.format(min_id))
    #                 return min_id, 1
    #             min_pt_x, min_pt_y, _ = pts[min_id]
    #             next_to_min_dist = (next_pt_x - min_pt_x) ** 2 + (next_pt_y - min_pt_y) ** 2
    #
    #             next_to_curr_dist = dist[min_id + 1]
    #             # print('next_to_min_dist: {}'.format(next_to_min_dist))
    #             # print('min_dist: {}'.format(min_dist))
    #             # print('next_to_curr_dist: {}'.format(next_to_curr_dist))
    #             if next_to_min_dist > next_to_curr_dist:
    #                 # print('returning {}, 1'.format(min_id))
    #                 return min_id, 1
    #             else:
    #                 # print('returning {}, 0'.format(min_id))
    #                 return min_id, 0
    #         else:
    #             # print('returning {}, 1'.format(min_id))
    #             return min_id, 1
    #     else:
    #         # print('returning {}'.format(min_id))
    #         return min_id
    #
    #     # print('Something weird going on here')

    def runHED(self, shape_patch, hed_net):
        print('Running HED...')

        hed_start_t = time.time()
        (H, W) = shape_patch.shape[:2]
        blob = cv2.dnn.blobFromImage(shape_patch, scalefactor=1.0, size=(W, H),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
        hed_net.setInput(blob)
        hed_img = hed_net.forward()
        hed_img = cv2.resize(hed_img[0, 0], (W, H))
        hed_img = (255 * hed_img).astype("uint8")
        hed_end_t = time.time()
        print('time taken: {} secs'.format(hed_end_t - hed_start_t))
        cv2.imshow('hed_img', hed_img)

        threshold = 50
        hed_mask = hed_pts = None

        def update_threshold(x):
            nonlocal threshold, hed_mask, hed_pts
            threshold = x
            _, hed_binary = cv2.threshold(hed_img, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('hed_binary', hed_binary)

            hed_pts, _ = self.contourPtsFromMask(hed_binary)
            hed_mask, _ = self.contourPtsToMask(hed_pts, shape_patch)
            cv2.imshow('hed_mask', hed_mask)

        update_threshold(threshold)
        cv2.createTrackbar('threshold', 'hed_binary', threshold, 255, update_threshold)

        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyWindow('hed_img')
            cv2.destroyWindow('hed_binary')
            cv2.destroyWindow('hed_mask')
            return None

        cv2.destroyWindow('hed_img')
        cv2.destroyWindow('hed_binary')
        cv2.destroyWindow('hed_mask')

        return hed_mask

    @staticmethod
    def getContourPts(mask_pts=None, shape=None, show_img=0, verbose=1,
                      patch_img=None, mag_patch_size=50, mag_win_size=800, show_magnified_window=1):
        # if mask_pts is None:
        #     if not self.mask:
        #         print('No mask has been added')
        #         return []
        #     mask_pts = self.mask

        if shape is None:
            show_img = 0
        else:
            height, width = shape

        n_cols = len(Shape.cols)

        n_pts = len(mask_pts)
        mask_segments = []
        curr_segment = []

        for pt_id in range(n_pts):
            _pt = mask_pts[pt_id]
            if not curr_segment:
                if _pt[2]:
                    curr_segment.append(pt_id)
                continue
            if not _pt[2]:
                curr_segment.append(pt_id)
                if curr_segment[1] - curr_segment[0] > 2:
                    mask_segments.append(curr_segment)
                else:
                    if verbose:
                        print('Discarding too short segment {}'.format(curr_segment))
                curr_segment = []
        if len(curr_segment) == 1:
            curr_segment.append(n_pts)
            if curr_segment[1] - curr_segment[0] > 2:
                mask_segments.append(curr_segment)
            else:
                if verbose:
                    print('Discarding too short segment {}'.format(curr_segment))

        n_segments = len(mask_segments)
        if n_segments == 0:
            if verbose:
                print('No segments found')
            return []

        if verbose:
            print('Found {} segments: {}'.format(n_segments, mask_segments))

        end_pts_to_segment_dict = {mask_segments[i][0]: (i, 0) for i in range(n_segments)}
        end_pts_to_segment_dict.update({mask_segments[i][1] - 1: (i, 1) for i in range(n_segments)})

        # print('end_pts_to_segment_dict: {}'.format(end_pts_to_segment_dict))

        free_end_pts = list(end_pts_to_segment_dict.keys())

        # print('free_end_pts: {}'.format(free_end_pts))

        contour_pts = [(x, y, 1) for x, y, _ in mask_pts[mask_segments[0][0]:mask_segments[0][1]]]
        free_end_pts.remove(mask_segments[0][0])
        free_end_pts.remove(mask_segments[0][1] - 1)
        # to_remove = mask_segments[0][1] - 1
        min_seg_id = 0
        _id = 0
        while True:
            # try:
            #     free_end_pts.remove(to_remove)
            # except ValueError as e:
            #     print('{} not in free_end_pts'.format(to_remove))
            #     # raise ValueError(e)

            if not free_end_pts:
                break

            # print('free_end_pts: {}'.format(free_end_pts))
            # print('end_pts_to_segment_dict: {}'.format(end_pts_to_segment_dict))

            x, y, _ = contour_pts[-1]
            # x, y, _ = mask_pts[to_remove]

            curr_dists = [abs(x - mask_pts[k][0]) + abs(y - mask_pts[k][1]) for k in free_end_pts]
            min_dist = min(curr_dists)

            _min_id = curr_dists.index(min_dist)
            min_end_pt = free_end_pts[_min_id]

            min_id = end_pts_to_segment_dict[min_end_pt]

            min_seg_id = min_id[0]
            rev_pts = min_id[1]

            # print('_min_id: {} min_end_pt: {} min_id: {}'.format(_min_id, min_end_pt, min_id))

            # if min_id != i:
            #     mask_segments[min_id], mask_segments[i] = mask_segments[i], mask_segments[min_id]
            col_id = (_id % n_cols) + 1
            _id += 1

            curr_pts = [(x, y, col_id) for x, y, _ in
                        mask_pts[mask_segments[min_seg_id][0]:mask_segments[min_seg_id][1]]]
            if rev_pts:
                curr_pts = reversed(curr_pts)
                # free_end_pts.remove(mask_segments[min_seg_id][1] - 1)
                # to_remove = mask_segments[min_seg_id][0]
            else:
                pass
                # free_end_pts.remove(mask_segments[min_seg_id][0])
                # to_remove = mask_segments[min_seg_id][1] - 1

            contour_pts += curr_pts

            # print('to_remove: {}'.format(to_remove))
            # print('min_seg_id: {}'.format(min_seg_id))

            free_end_pts.remove(mask_segments[min_seg_id][0])
            free_end_pts.remove(mask_segments[min_seg_id][1] - 1)

        # contour_pts += [(x, y, col_id) for x, y, _ in mask_pts[mask_segments[-1][0]:mask_segments[-1][1]]]

        # close the contour
        if contour_pts[-1] != contour_pts[0]:
            contour_pts.append(contour_pts[0])

        n_contour_pts = len(contour_pts)
        if verbose:
            print('Found {} contour_pts'.format(n_contour_pts))

        if show_img:
            bin_img = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(n_contour_pts - 1):
                pt1, pt2 = contour_pts[i], contour_pts[i + 1]
                col_id = pt2[2]
                pt1 = (int(pt1[0]), int(pt1[1]))
                pt2 = (int(pt2[0]), int(pt2[1]))

                col = col_rgb[Shape.cols[col_id]]
                # col = (255, 255, 255)

                bin_img = cv2.line(bin_img, pt1, pt2, col, thickness=2)

            cv2.imshow('mask segments', bin_img)

        # bin_img_gs = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        # im2, contours, hierarchy = cv2.findContours(bin_img_gs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)

        # bin_img2 = np.zeros((height, width, 3), dtype=np.uint8)
        # cv2.drawContours(bin_img2, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        # cv2.imshow('bin_img2', bin_img2)

        contour_pts = [[x, y] for x, y, _ in contour_pts]

        return contour_pts

    @staticmethod
    def contourPtsToMask(contour_pts, patch_img, col=(255, 255, 255)):
        # np.savetxt('contourPtsToMask_mask_pts.txt', contour_pts, fmt='%.6f')

        mask_img = np.zeros_like(patch_img, dtype=np.uint8)
        # if not isinstance(contour_pts, list):
        #     raise SystemError('contour_pts must be a list rather than {}'.format(type(contour_pts)))
        if len(contour_pts) > 0:
            mask_img = cv2.fillPoly(mask_img, np.array([contour_pts, ], dtype=np.int32), col)
        blended_img = np.array(Image.blend(Image.fromarray(patch_img), Image.fromarray(mask_img), 0.5))

        return mask_img, blended_img

    @staticmethod
    def contourPtsFromMask(mask_img):
        # print('Getting contour pts from mask...')
        if len(mask_img.shape) == 3:
            mask_img_gs = np.squeeze(mask_img[:, :, 0]).copy()
        else:
            mask_img_gs = mask_img.copy()

        ret = cv2.findContours(mask_img_gs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
        _contour_pts, _ = ret
        if not _contour_pts:
            return [], []
        contour_pts = list(np.squeeze(_contour_pts[0]))

        n_contours = len(_contour_pts)
        # print('n_contours: {}'.format(n_contours))
        # print('_contour_pts: {}'.format(_contour_pts))
        # print('contour_pts: {}'.format(type(contour_pts)))

        if n_contours > 1:
            max_len = len(contour_pts)
            for _pts in _contour_pts[1:]:
                # print('_pts: {}'.format(_pts))
                _pts = np.squeeze(_pts)
                _len = len(_pts)
                if max_len < _len:
                    contour_pts = _pts
                    max_len = _len

        # print('contour_pts len: {}'.format(len(contour_pts)))
        mask_pts = [[x, y, 1] for x, y in contour_pts]

        return contour_pts, mask_pts

    def addMask(self, image, params, augment=None, hed_net=None):

        disp_size = params.disp_size
        border_size = params.border_size
        del_thresh = params.del_thresh
        show_magnified_window = params.show_magnified_window
        mag_patch_size = params.mag_patch_size
        mag_win_size = params.mag_win_size
        mag_thresh_t = params.mag_thresh_t
        show_pts = params.show_pts

        start_t = time.time()
        try:
            xmin = int(self.points[0].x())
            ymin = int(self.points[0].y())
            xmax = int(self.points[2].x())
            ymax = int(self.points[2].y())
        except BaseException as e:
            print('Something weird going on: {}'.format(e))
            return

        # width = image.width()
        # height = image.height()
        #
        # ptr = image.bits()
        # ptr.setsize(image.byteCount())
        #
        # image = np.array(ptr).reshape(height, width, 3)

        height, width = image.shape[:2]

        border_x, border_y = border_size

        if border_x > 0:
            xmin = max(0, xmin - border_x)
            xmax = min(width - 1, xmax + border_x)

        if border_y > 0:
            ymin = max(0, ymin - border_y)
            ymax = min(height - 1, ymax + border_y)

        shape_patch_orig = image[ymin:ymax, xmin:xmax, :]

        h, w = shape_patch_orig.shape[:2]
        scale_x, scale_y = disp_size[0] / w, disp_size[1] / h
        scale_factor = min(scale_x, scale_y)
        shape_patch = cv2.resize(shape_patch_orig, (0, 0), fx=scale_factor, fy=scale_factor)

        print('disp_size: ', disp_size)
        print('scale_factor: ', scale_factor)
        print('shape_patch: ', shape_patch.shape)

        # print('disp_size: {}'.format(disp_size))
        # print('self.mask: {}'.format(self.mask))

        mask_pts = [[(k[0] - xmin) * scale_factor, (k[1] - ymin) * scale_factor, k[2]] for k in self.mask]

        sel_pt_id = -1
        is_continuous = 0
        _exit_mask = 0
        draw_mask = 0
        start_painting_mode = 0
        paint_mode = 0
        discard_changes = 0
        clean_mask_pts = 0
        prev_mouse_pt = []
        prev_rect_pts = []
        mouse_x = mouse_y = 0
        end_pts = []
        start_id = 0
        mag_prev_t = 0
        blended_img = mask_img = disp_img = None

        max_dist = del_thresh * del_thresh

        mouse_whl_keys_to_flags = {
            'none': (7864320, -7864320),
            'ctrl': (7864328, -7864312),
            'alt': (7864352, -7864288),
            'shift': (7864336, -7864304),
            'ctrl+alt': (7864360, -7864280),
            'ctrl+shift': (7864344, -7864296),
            'alt+shift': (7864368, -7864272),
            'ctrl+alt+shift': (7864376, -7864264),
        }

        def drawContour(img, mask_pts, curr_pt=None,
                        cursor_thickness=0, show_centroid=0,
                        start_id=0, show_pts=0, contour_img=None):
            nonlocal end_pts

            n_pts = len(mask_pts)

            if contour_img is None:
                if start_id == 0:
                    _img = np.copy(img)
                else:
                    _img = img
                # print('mask_pts: {}'.format(mask_pts))
                pt1 = pt2 = None
                # pts = []
                end_pts = []
                _is_continuous = 0
                # min_dist = np.inf
                # _nearest_end_pt = mask_pts[0]
                for i in range(start_id, n_pts - 1):
                    pt1, pt2 = mask_pts[i], mask_pts[i + 1]
                    if not pt1[2] or not pt2[2]:
                        if _is_continuous:
                            end_pts.append(pt1)
                            # if curr_pt:
                            #     dist = abs(curr_pt[0] - pt1[0]) + abs(curr_pt[1] - pt1[1])
                            #     if dist < min_dist:
                            #         min_dist = dist
                            #         _nearest_end_pt = pt1
                        _is_continuous = 0
                        continue

                    # pts.append(pt1[:2])
                    # pts.append(pt2[:2])

                    _img = cv2.line(_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                                    (0, 255, 0), thickness=2)
                    if show_pts:
                        _img = cv2.drawMarker(_img, (int(pt1[0]), int(pt1[1])), (0, 255, 0), cv2.MARKER_STAR,
                                              markerSize=10)

                    if not _is_continuous:
                        _is_continuous = 1
                        end_pts.append(pt1)
                        # if curr_pt:
                        #     dist = abs(curr_pt[0] - pt1[0]) + abs(curr_pt[1] - pt1[1])
                        #     if dist < min_dist:
                        #         min_dist = dist
                        #         _nearest_end_pt = pt1
                        # end_ptsa.append(pt1)
                contour_img = np.copy(_img)
                if curr_pt is not None:
                    if _is_continuous:
                        end_pts.append(pt2)

                    curr_pt = (int(curr_pt[0]), int(curr_pt[1]))
                    if cursor_thickness > 0:
                        cv2.circle(_img, curr_pt, cursor_thickness, (0, 0, 255), 1)
                    else:
                        if n_pts == 0:
                            _img = cv2.drawMarker(_img, curr_pt, (0, 255, 0),
                                                  cv2.MARKER_STAR, markerSize=10)
                        elif n_pts == 1 and mask_pts[0][2]:
                            prev_pt = (int(mask_pts[0][0]), int(mask_pts[0][1]))
                            _img = cv2.line(_img, prev_pt, curr_pt,
                                            (0, 255, 0), thickness=2)
                        elif end_pts:
                            if len(end_pts) == 1:
                                _nearest_end_pt = end_pts[0]
                            else:
                                dists = [abs(curr_pt[0] - pt[0]) + abs(curr_pt[1] - pt[1]) for pt in end_pts]
                                _nearest_end_pt = end_pts[dists.index(min(dists))]
                            prev_pt = (int(_nearest_end_pt[0]), int(_nearest_end_pt[1]))
                            _img = cv2.line(_img, prev_pt, curr_pt, (0, 255, 0), thickness=2)
                if show_centroid:
                    _h, _w = _img.shape[:2]
                    mask_centroid = [int(_w / 2), int(_h / 2)]
                    # mask_centroid = np.mean(mask_pts, axis=0).astype(np.int32)
                    cv2.circle(_img, (mask_centroid[0], mask_centroid[1]),
                               5, (0, 255, 0), -1)

                if _is_continuous:
                    if show_pts:
                        _img = cv2.drawMarker(_img, (int(pt2[0]), int(pt2[1])), (0, 255, 0),
                                              cv2.MARKER_STAR, markerSize=10)
            else:
                _img = np.copy(contour_img)
                if cursor_thickness > 0:
                    # pass

                    # x, y = curr_pt
                    # img_h, img_w = _img.shape[:2]
                    # min_x, min_y = max(0, x - cursor_thickness), max(0, y - cursor_thickness)
                    # max_x, max_y = min(img_w, x + cursor_thickness), min(img_h, y + cursor_thickness)
                    # _img[min_y:max_y, min_x:max_x, :] += 10

                    cv2.circle(_img, curr_pt, cursor_thickness, (0, 0, 255), 1)

            return _img, contour_img

        _shape_patch, _contour_patch = drawContour(shape_patch, mask_pts, show_pts=show_pts)

        def showMagnifiedWindow(x, y, source_patch, draw_marker=1,
                                win_name='Magnified', marker_col=(0, 0, 255)):
            nonlocal mag_prev_t, mag_thresh_t
            # mag_t = time.time() - mag_prev_t
            # print('mag_t: ', mag_t)
            # if mag_t < mag_thresh_t:
            #     return

            _h, _w = _shape_patch.shape[:2]
            min_x, max_x = max(0, x - mag_patch_size), min(_w - 1, x + mag_patch_size)
            min_y, max_y = max(0, y - mag_patch_size), min(_h - 1, y + mag_patch_size)

            _x, _y = x - min_x, y - min_y

            mag_patch = np.copy(source_patch[min_y:max_y, min_x:max_x, :])
            if draw_marker == 1:
                mag_patch = cv2.circle(mag_patch, (_x, _y), 1, marker_col, -1)
            elif draw_marker == 2:
                _min_x, _min_y = _x - del_thresh, _y - del_thresh
                _max_x, _max_y = _x + del_thresh, _y + del_thresh
                # mag_patch[_min_y:_max_y, _min_x:_max_x, :] += 10
                cv2.rectangle(mag_patch, (_min_x, _min_y), (_max_x, _max_y), marker_col, 1)

            __h, __w = mag_patch.shape[:2]
            border_x, border_y = int(mag_patch_size - __w / 2), int(mag_patch_size - __h / 2)
            if border_x or border_y:
                mag_patch = cv2.copyMakeBorder(mag_patch, border_y, border_y, border_x, border_x,
                                               cv2.BORDER_CONSTANT)

            mag_patch = cv2.resize(mag_patch, (mag_win_size, mag_win_size))

            cv2.imshow(win_name, mag_patch)

            # mag_prev_t = time.time()

        def paintMouseHandler(event, x, y, flags=None, param=None):
            nonlocal mask_pts, contour_pts, blended_img, mask_img, disp_img, del_thresh, mag_patch_size, mag_win_size, \
                mouse_x, mouse_y, prev_mouse_pt, draw_mask_kb, paint_mode

            # def getNearestPts(x, y, max_dist, max_dist_sqr):
            #     min_x, min_y = x - max_dist, y - max_dist
            #     max_x, max_y = x + max_dist, y + max_dist
            #     valid_pts = [(_x, _y) for _x in range(min_x, max_x) for _y in range(min_y, max_y)
            #                  # if ((x - _x) ** 2 + (y - _y) ** 2) <= max_dist_sqr
            #                  ]
            #     return valid_pts

            paint_mode = 1
            mouse_x, mouse_y = x, y
            # refresh_paint_win = 1
            draw_marker = 1
            marker_col = (0, 255, 0)
            # _show_magnified_window = 0

            if event == cv2.EVENT_MOUSEMOVE:
                # print('flags: {}'.format(flags))
                # refresh_paint_win = 1
                if flags == 1 or flags == 25:
                    # left button
                    min_x, min_y = x - del_thresh, y - del_thresh
                    max_x, max_y = x + del_thresh, y + del_thresh
                    mask_img[min_y:max_y, min_x:max_x, :] = 255
                    blended_img[min_y:max_y, min_x:max_x, :] = (shape_patch[min_y:max_y, min_x:max_x,
                                                                :] + 255.0) / 2.0
                    marker_col = (0, 255, 0)
                    # _show_magnified_window = 1
                    # pts = getNearestPts(x, y, max_dist, max_dist_sqr)
                    # for x, y in pts:
                    #     mask_img[y, x, :] = 255
                    #     blended_img[y, x, 0] = int((255+patch_img[y, x, 0])/2)
                    #     blended_img[y, x, 1] = int((255+patch_img[y, x, 1])/2)
                    #     blended_img[y, x, 2] = int((255+patch_img[y, x, 2])/2)
                elif flags == 17:
                    # shift + left button
                    min_x, min_y = x - del_thresh, y - del_thresh
                    max_x, max_y = x + del_thresh, y + del_thresh
                    mask_img[min_y:max_y, min_x:max_x, :] = 0
                    blended_img[min_y:max_y, min_x:max_x, :] = (shape_patch[min_y:max_y, min_x:max_x, :]) / 2.0
                    # _show_magnified_window = 1

                    marker_col = (0, 0, 255)

                    # pts = getNearestPts(x, y, max_dist, max_dist_sqr)
                    # for x, y in pts:
                    #     mask_img[y, x, :] = 0
                    #     blended_img[y, x, 0] = int((0+patch_img[y, x, 0])/2)
                    #     blended_img[y, x, 1] = int((0+patch_img[y, x, 1])/2)
                    #     blended_img[y, x, 2] = int((0+patch_img[y, x, 2])/2)
                elif flags == 32:
                    draw_marker = 0
                elif flags == 16:
                    marker_col = (0, 0, 255)
                elif flags == 8:
                    draw_marker = 0
                # else:
                #     draw_marker = 0
            elif event == cv2.EVENT_LBUTTONDOWN:
                # print('flags: {}'.format(flags))
                pass
            elif event == cv2.EVENT_RBUTTONUP:
                # print('flags: {}'.format(flags))
                pass
            elif event == cv2.EVENT_MBUTTONDOWN:
                contour_pts, mask_pts = self.contourPtsFromMask(mask_img)
                draw_mask_kb = 1
                # print('flags: {}'.format(flags))
            elif event == cv2.EVENT_MOUSEWHEEL:
                if flags > 0:
                    if flags == mouse_whl_keys_to_flags['ctrl+alt+shift'][0]:
                        mag_win_size += 10
                        print('magnified window size increased to {}'.format(mag_win_size))
                    elif flags == mouse_whl_keys_to_flags['ctrl+shift'][0]:
                        mag_patch_size -= 1
                        if mag_patch_size < 5:
                            mag_patch_size = 5
                    # elif flags == mouse_whl_keys_to_flags['shift'][0]:
                    #     pass
                    else:
                        if del_thresh < 10:
                            del_thresh += 1
                        else:
                            del_thresh += 5
                        print('del_thresh increased to {}'.format(del_thresh))
                else:
                    if flags == mouse_whl_keys_to_flags['ctrl+alt+shift'][1]:
                        mag_win_size -= 10
                        if mag_win_size < 100:
                            mag_win_size = 100
                        print('magnified window size decreased to {}'.format(mag_win_size))

                    elif flags == mouse_whl_keys_to_flags['ctrl+shift'][1]:
                        mag_patch_size += 1
                    # elif flags == mouse_whl_keys_to_flags['shift'][1]:
                    #     pass
                    else:
                        if del_thresh < 10:
                            del_thresh = max(del_thresh - 1, 1)
                        else:
                            del_thresh -= 5
                        print('del_thresh decreased to {}'.format(del_thresh))

            # if disp_img is None or not prev_mouse_pt:
            #     disp_img = np.copy(blended_img)
            # else:
            #     _x, _y = prev_mouse_pt
            #     min_x, min_y = _x - del_thresh, _y - del_thresh
            #     max_x, max_y = _x + del_thresh, _y + del_thresh
            #     disp_img[min_y:max_y, min_x:max_x, :] = blended_img[min_y:max_y, min_x:max_x, :]
            # prev_rect_pts = [()]

            if draw_marker:
                disp_img = np.copy(blended_img)
                # disp_img = blended_img
                min_x, min_y = x - del_thresh, y - del_thresh
                max_x, max_y = x + del_thresh, y + del_thresh
                # disp_img = Image.fromarray(disp_img)
                # ImageDraw.Draw(disp_img).rectangle(
                #     [min_x, min_y, max_x, max_y], outline=marker_col)
                # disp_img = np.asarray(disp_img, dtype=np.uint8)

                # disp_img = cv2.drawMarker(disp_img, (x, y), (0, 0, 255),
                #                           cv2.MARKER_SQUARE, markerSize=del_thresh)
                # disp_img[min_y:max_y, min_x:max_x, :] += 10
                cv2.rectangle(disp_img, (min_x, min_y), (max_x, max_y), marker_col, 1)
            else:
                disp_img = blended_img
            cv2.imshow(paint_win_name, disp_img)

            # disp_img = cv2.drawMarker(disp_img, (x, y), (0, 0, 255),
            #                           cv2.MARKER_STAR, markerSize=5)

            # else:
            #     disp_img = blended_img

            # cv2.rectangle(_shape_patch, (min_x, min_y), (max_x, max_y), marker_col, 1)

            if show_magnified_window:
                showMagnifiedWindow(x, y, _shape_patch, draw_marker=2,
                                    marker_col=marker_col,
                                    # win_name='Paint Magnified'
                                    )
            # cv2.imshow('binary mask', mask_img)
            prev_mouse_pt = (x, y)

        def drawMouseHandler(event, x, y, flags=None, param=None):
            nonlocal mask_pts, sel_pt_id, disp_size, shape_patch, scale_factor, prev_mouse_pt, mouse_x, mouse_y, \
                is_continuous, _exit_mask, del_thresh, draw_mask, max_dist, clean_mask_pts, discard_changes, \
                _shape_patch, mag_patch_size, mag_win_size, start_id, start_painting_mode, cursor_thickness, show_centroid, \
                blended_img, mask_img, paint_mode, _contour_patch

            paint_mode = 0

            _h, _w = shape_patch.shape[:2]
            x = max(min(x, _w - 1), 0)
            y = max(min(y, _h - 1), 0)

            mouse_x, mouse_y = x, y

            if draw_mask > 1:
                draw_mask = 1
            else:
                draw_mask = 0

            if sel_pt_id >= len(mask_pts):
                sel_pt_id = -1

            single_pt_mode = 0
            start_id = 0

            def getNearestPts(pts, pt, max_dist):
                if not pts:
                    return []
                n_pts = len(pts)
                x, y = pt
                dist = [(x - _x) ** 2 + (y - _y) ** 2 if _f == 1 else np.inf for _x, _y, _f in pts]
                valid_pt_ids = [i for i in range(n_pts) if dist[i] < max_dist]

                return valid_pt_ids

            # print('flags: {}'.format(flags))

            continuity_broken = 1
            if event == cv2.EVENT_MOUSEMOVE:
                # print('flags: {}'.format(flags))
                if flags == 1 or flags == 25 or flags == 24:
                    continuity_broken = 0
                    if not is_continuous:
                        is_continuous = 1
                        mask_pts.append([0, 0, 0])
                        mask_pts.append([x, y, 1])
                    else:
                        mask_pts.append([x, y, 1])
                        start_id = len(mask_pts) - 2
                    draw_mask = 1
                    is_continuous = 1
                elif flags == 17:
                    # shift
                    pt_ids = getNearestPts(mask_pts, (x, y), max_dist)
                    for _id in pt_ids:
                        mask_pts[_id][2] = 0
                    draw_mask = 5
                elif flags == 33:
                    # alt + left mouse
                    if prev_mouse_pt:
                        tx, ty = x - prev_mouse_pt[0], y - prev_mouse_pt[1]
                        mask_pts = [[_x + tx, _y + ty, f] for _x, _y, f in mask_pts]
                    prev_mouse_pt = [x, y]
                    draw_mask = 1
                    single_pt_mode = 1
                elif flags == 16:
                    # delete_mode = 1
                    draw_mask = 3
                elif flags == 8:
                    draw_mask = 2
            elif event == cv2.EVENT_LBUTTONDOWN:
                # print('flags: {}'.format(flags))
                if flags == 9:
                    if end_pts:
                        dists = [abs(x - pt[0]) + abs(y - pt[1]) for pt in end_pts]
                        _nearest_end_pt = end_pts[dists.index(min(dists))]
                        if _nearest_end_pt != mask_pts[-1]:
                            mask_pts.append([0, 0, 0])
                            mask_pts.append(_nearest_end_pt)
                    mask_pts.append([x, y, 1])
            elif event == cv2.EVENT_RBUTTONUP:
                print('flags: {}'.format(flags))
                if flags == 24:
                    # ctrl + shift
                    _contour_pts = self.getContourPts(mask_pts)
                    print('Cleaning up the mask points...')
                    mask_pts = [[x, y, 1] for x, y in _contour_pts]
                    draw_mask = 1
                elif flags == 16:
                    # shift
                    print('Deleting all mask points ...')
                    mask_pts = []
                    draw_mask = 1
                elif flags == 8:
                    # ctrl
                    run_augmentation(use_prev=1)
                elif flags == 32:
                    # alt
                    run_augmentation(use_prev=0)
                else:
                    # if not delete_mode:
                    _contour_pts = self.getContourPts(mask_pts, shape_patch.shape[:2], show_img=1)
                    mask_img, blended_img = self.contourPtsToMask(_contour_pts, shape_patch)
                    start_painting_mode = 1
            elif event == cv2.EVENT_MBUTTONDOWN:
                print('flags: {}'.format(flags))
                if flags == 4:
                    clean_mask_pts = 2
                    _exit_mask = 1
                if flags == 36:
                    clean_mask_pts = 1
                    _exit_mask = 1
                elif flags == 28:
                    _exit_mask = 1
                elif flags == 20:
                    discard_changes = 1
                    _exit_mask = 1
            elif event == cv2.EVENT_MOUSEWHEEL:
                # flags -= 7864320
                print('flags: {}'.format(flags))
                _disp_size = disp_size
                if flags > 0:
                    if flags == mouse_whl_keys_to_flags['ctrl+alt+shift'][0]:
                        mag_win_size += 10
                        print('magnified window size increased to {}'.format(mag_win_size))
                    elif flags == mouse_whl_keys_to_flags['ctrl+shift'][0]:
                        mag_patch_size -= 1
                        if mag_patch_size < 5:
                            mag_patch_size = 5
                        # show_magnified_window = 1
                    elif flags == mouse_whl_keys_to_flags['shift'][0]:
                        if del_thresh < 10:
                            del_thresh += 1
                        else:
                            del_thresh += 5
                        print('del_thresh increased to {}'.format(del_thresh))
                        max_dist = del_thresh * del_thresh
                        # delete_mode = 1
                        draw_mask = 3
                    elif flags == mouse_whl_keys_to_flags['none'][0]:
                        _disp_size = tuple([min(2000, x + 10) for x in disp_size])
                else:
                    if flags == mouse_whl_keys_to_flags['ctrl+alt+shift'][1]:
                        mag_win_size -= 10
                        if mag_win_size < 100:
                            mag_win_size = 100
                        print('magnified window size decreased to {}'.format(mag_win_size))

                    elif flags == mouse_whl_keys_to_flags['ctrl+shift'][1]:
                        mag_patch_size += 1
                        # show_magnified_window = 1
                    elif flags == mouse_whl_keys_to_flags['shift'][1]:
                        if del_thresh > 10:
                            del_thresh -= 5
                        else:
                            del_thresh -= 1
                        if del_thresh < 1:
                            del_thresh = 1
                        max_dist = del_thresh * del_thresh
                        print('del_thresh decreased to {}'.format(del_thresh))
                        # delete_mode = 1
                        draw_mask = 3
                    elif flags == mouse_whl_keys_to_flags['none'][1]:
                        _disp_size = tuple([max(200, x - 10) for x in disp_size])
                if disp_size != _disp_size:
                    scale_x, scale_y = _disp_size[0] / w, _disp_size[1] / h
                    _scale_factor = min(scale_x, scale_y)
                    shape_patch = cv2.resize(shape_patch_orig, (0, 0), fx=_scale_factor, fy=_scale_factor)
                    print('_disp_size: ', _disp_size)
                    print('_scale_factor: ', _scale_factor)
                    print('shape_patch: ', shape_patch.shape)
                    k = _scale_factor / scale_factor
                    mask_pts = [[x * k, y * k, f] for x, y, f in mask_pts]
                    draw_mask = 1
                    disp_size = _disp_size
                    scale_factor = _scale_factor

                # print('mask_pts: {}'.format(mask_pts))
                # print('mask: {}'.format(self.mask))

            # k = cv2.waitKey(1)
            # print('k: {}'.format(k))
            # if k == 27:
            #     cv2.destroyWindow(draw_win_name)
            #     return
            if not single_pt_mode:
                prev_mouse_pt = []

            if continuity_broken:
                is_continuous = 0

            if draw_mask:
                contour_patch = None
                curr_pt = None
                cursor_thickness = 0
                show_centroid = 0
                if draw_mask == 2:
                    curr_pt = (x, y)
                elif draw_mask == 3:
                    curr_pt = (x, y)
                    cursor_thickness = del_thresh
                    contour_patch = _contour_patch
                elif draw_mask == 4:
                    show_centroid = 1
                elif draw_mask == 5:
                    curr_pt = (x, y)
                    cursor_thickness = del_thresh
                if start_id > 0:
                    _img = _shape_patch
                else:
                    _img = shape_patch
                _shape_patch, _contour_patch = drawContour(
                    _img, mask_pts, curr_pt, cursor_thickness, show_centroid,
                    start_id, show_pts, contour_img=contour_patch)
                cv2.imshow(draw_win_name, _shape_patch)

            if show_magnified_window:
                showMagnifiedWindow(x, y, _shape_patch)

            if start_painting_mode:
                print('Initial mask creation took {} secs'.format(
                    time.time() - start_t))
                cv2.imshow(paint_win_name, blended_img)
                cv2.setMouseCallback(paint_win_name, paintMouseHandler)
                start_painting_mode = 0

            # print('continuity_broken: {}'.format(continuity_broken))
            # print('start_id: {}'.format(start_id))

        def run_augmentation(use_prev=1):
            if augment is not None:
                mask = [(xmin + x / scale_factor, ymin + y / scale_factor, f)
                        for (x, y, f) in mask_pts]
                augment(None, mask, use_prev=use_prev, save_seq=0)

        paint_win_name = 'Paint the mask'
        draw_win_name = 'Draw the mask'
        cv2.imshow(draw_win_name, _shape_patch)
        cv2.setMouseCallback(draw_win_name, drawMouseHandler,
                             # param=(mask_pts, shape_patch)
                             )
        draw_mask_kb = 0
        while not _exit_mask:
            k = cv2.waitKeyEx(100)
            if not draw_mask_kb and k < 0:
                continue
            print('k: {}'.format(k))

            if k == ord('p'):
                show_pts = 1 - show_pts
                draw_mask_kb = 1
            elif k == ord('l'):
                print('Running Laplacian filtering...')
                shape_patch_gs = cv2.cvtColor(shape_patch, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(shape_patch_gs, cv2.CV_64F)
                threshold = 0

                def update_threshold(x):
                    nonlocal threshold
                    threshold = x
                    _, laplacian_binary = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)
                    cv2.imshow('laplacian_binary', laplacian_binary)

                _, laplacian_binary = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)
                cv2.imshow('laplacian', laplacian)
                cv2.imshow('laplacian_binary', laplacian_binary)

                cv2.createTrackbar('threshold', 'laplacian_binary', int(threshold), 20, update_threshold)

            elif k == ord('s'):
                print('Running Sobel filtering...')
                shape_patch_gs = cv2.cvtColor(shape_patch, cv2.COLOR_BGR2GRAY)
                # shape_patch_gs = cv2.GaussianBlur(shape_patch_gs, (k, k), 0)
                edgeX = cv2.Sobel(shape_patch_gs, cv2.CV_16S, 1, 0)
                edgeY = cv2.Sobel(shape_patch_gs, cv2.CV_16S, 0, 1)

                edgeX = np.uint8(np.absolute(edgeX))
                edgeY = np.uint8(np.absolute(edgeY))
                edge = cv2.bitwise_or(edgeX, edgeY)

                threshold = 50

                def update_threshold(x):
                    nonlocal threshold
                    threshold = x
                    _, sobel_binary = cv2.threshold(edge, threshold, 255, cv2.THRESH_BINARY)
                    cv2.imshow('sobel_binary', sobel_binary)

                _, sobel_binary = cv2.threshold(edge, threshold, 255, cv2.THRESH_BINARY)
                cv2.imshow('edge', edge)
                cv2.imshow('sobel_binary', sobel_binary)
                cv2.createTrackbar('threshold', 'sobel_binary', threshold, 100, update_threshold)

            elif k == ord('o'):
                print('Running Otsu thresholding...')
                shape_patch_gs = cv2.cvtColor(shape_patch, cv2.COLOR_BGR2GRAY)
                val = filters.threshold_otsu(shape_patch_gs)
                otsu_mask = np.array(shape_patch_gs < val, dtype=np.uint8) * 255
                cv2.imshow('otsu_mask', otsu_mask)
            elif k == ord('E'):
                shape_patch_gs = cv2.cvtColor(shape_patch, cv2.COLOR_BGR2GRAY)

                # shape_patch_gs = ndi.gaussian_filter(shape_patch_gs, 4)
                sigma = 3
                edges_canny = feature.canny(shape_patch_gs, sigma=sigma)
                edges_canny = np.array(edges_canny, dtype=np.uint8) * 255
                cv2.imshow('edges_canny', edges_canny)

                def update_sigma(x):
                    nonlocal sigma
                    sigma = x
                    edges_canny = feature.canny(shape_patch_gs, sigma=sigma)
                    edges_canny = np.array(edges_canny, dtype=np.uint8) * 255
                    cv2.imshow('edges_canny', edges_canny)

                cv2.createTrackbar('sigma', 'edges_canny', sigma, 100, update_sigma)

            elif k == ord('e'):
                print('Running Canny edge detection...')
                threshold1 = 50
                threshold2 = 25
                canny_edge_patch = cv2.Canny(shape_patch, threshold1, threshold2)
                cv2.imshow('canny_edge_patch', canny_edge_patch)

                def update_threshold1(x):
                    nonlocal threshold1
                    threshold1 = x
                    canny_edge_patch = cv2.Canny(shape_patch, threshold1, threshold2)
                    cv2.imshow('canny_edge_patch', canny_edge_patch)

                def update_threshold2(x):
                    nonlocal threshold2
                    threshold2 = x
                    canny_edge_patch = cv2.Canny(shape_patch, threshold1, threshold2)
                    cv2.imshow('canny_edge_patch', canny_edge_patch)

                cv2.createTrackbar('threshold1', 'canny_edge_patch', threshold1, 1000, update_threshold1)
                cv2.createTrackbar('threshold2', 'canny_edge_patch', threshold2, 1000, update_threshold2)

            elif k == ord('h'):
                if hed_net is not None:
                    hed_mask = self.runHED(shape_patch, hed_net)
                    if hed_mask is not None:
                        _, _mask_pts = self.contourPtsFromMask(hed_mask)
                        mask_pts = [[x, y, 1] for x, y in _mask_pts]
                        self.mask = [(xmin + x / scale_factor, ymin + y / scale_factor, f)
                                     for (x, y, f) in _mask_pts]
            elif k == ord('H'):
                pprint(self.mask_help)
            elif k == ord('m'):
                show_magnified_window = 1 - show_magnified_window
                if not show_magnified_window:
                    cv2.destroyWindow('Magnified')
            elif k == ord('c'):
                print('Cleaning up the mask points...')
                contour_pts = self.getContourPts(mask_pts)
                mask_pts = [[x, y, 1] for x, y in contour_pts]
                draw_mask_kb = 1
            elif k == ord('a'):
                run_augmentation(use_prev=1)
            elif k == ord('A'):
                run_augmentation(use_prev=0)
            elif k == ord('b'):
                _contour_pts = self.getContourPts(mask_pts, shape_patch.shape[:2], show_img=1)
                mask_img, blended_img = self.contourPtsToMask(_contour_pts, shape_patch)
                start_painting_mode = 1
            elif k == ord('q'):
                discard_changes = 1
                _exit_mask = 1
            elif k == 10:
                _exit_mask = 1
                clean_mask_pts = 1
            elif k == 13:
                _exit_mask = 1
                clean_mask_pts = 2
            elif k == 27:
                _exit_mask = 1
            elif k == ord('+') or k == ord('>'):
                del_thresh += 5
                print('del_thresh increased to {}'.format(del_thresh))
                max_dist = del_thresh * del_thresh
            elif k == ord('-') or k == ord('<'):
                del_thresh -= 5
                if del_thresh < 1:
                    del_thresh = 1
                print('del_thresh decreased to {}'.format(del_thresh))
                max_dist = del_thresh * del_thresh
            elif k == 2490368:
                # up
                mask_pts = [[_x, _y - 1, f] for _x, _y, f in mask_pts]
                draw_mask_kb = 1
            elif k == 2621440:
                # down
                mask_pts = [[_x, _y + 1, f] for _x, _y, f in mask_pts]
                draw_mask_kb = 1
            elif k == 2555904:
                # right
                mask_pts = [[_x + 1, _y, f] for _x, _y, f in mask_pts]
                draw_mask_kb = 1
            elif k == 2424832:
                # left
                mask_pts = [[_x - 1, _y, f] for _x, _y, f in mask_pts]
                draw_mask_kb = 1

            if start_painting_mode:
                print('Initial mask creation took {} secs'.format(
                    time.time() - start_t))
                cv2.imshow(paint_win_name, blended_img)
                cv2.setMouseCallback(paint_win_name, paintMouseHandler)
                start_painting_mode = 0

            if draw_mask_kb:
                curr_pt = None
                cursor_thickness = 0
                show_centroid = 0
                if draw_mask_kb == 2:
                    curr_pt = (mouse_x, mouse_y)
                elif draw_mask_kb == 3:
                    curr_pt = (mouse_x, mouse_y)
                    cursor_thickness = del_thresh
                elif draw_mask_kb == 4:
                    show_centroid = 1
                if start_id > 0:
                    _img = _shape_patch
                else:
                    _img = shape_patch
                _shape_patch, _contour_patch = drawContour(_img, mask_pts, curr_pt, cursor_thickness,
                                                           show_centroid, start_id, show_pts)
                cv2.imshow(draw_win_name, _shape_patch)
                draw_mask_kb = 0

        end_t = time.time()
        time_taken = end_t - start_t

        if clean_mask_pts:
            print('Cleaning up the mask points...')
            contour_pts = self.getContourPts(mask_pts)
            mask_pts = [[x, y, 1] for x, y in contour_pts]
        if not discard_changes:
            if paint_mode:
                contour_pts, mask_pts = self.contourPtsFromMask(mask_img)
            self.mask = [(xmin + x / scale_factor, ymin + y / scale_factor, f)
                         for (x, y, f) in mask_pts]

            min_border_x, min_border_y = params.min_box_border

            if min_border_x > 0 or min_border_x > 0:
                min_x, min_y = min_border_x, min_border_y
                max_x, max_y = width - min_border_x, height - min_border_y

                print('Clamping mask to be between {} and {}'.format(
                    (min_x, min_y), (max_x, max_y)))

                self.mask = [(min(max_x, max(min_x, x)), min(max_y, max(min_y, y)), f)
                             for (x, y, f) in self.mask]

            if clean_mask_pts == 2 and len(self.mask) > 0:
                print('Fixing bounding box using the updated mask')
                mask_arr = np.asarray([(x, y) for x, y, _ in self.mask])
                xmin, ymin = np.min(mask_arr, axis=0)
                xmax, ymax = np.max(mask_arr, axis=0)

                if xmin < xmax and ymin < ymax:
                    self.points[0] = QPointF(xmin, ymin)
                    self.points[1] = QPointF(xmax, ymin)
                    self.points[2] = QPointF(xmax, ymax)
                    self.points[3] = QPointF(xmin, ymax)
        else:
            print('Discarding changes ...')
        # cv2.destroyWindow(draw_win_name)
        cv2.destroyAllWindows()
        print('Mask creation took {} secs with {} mask points'.format(
            time_taken, len(self.mask)))

    def close(self):
        self._closed = True

    def reachMaxPoints(self):
        if len(self.points) >= self.max_points:
            return True
        return False

    def addPoint(self, point):
        if not self.reachMaxPoints():
            self.points.append(point)

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def paint(self, painter):
        if self.points and not self.is_hidden:
            color = self.select_line_color if self.selected else self.line_color
            pen = QPen(color)
            # Try using integer sizes for smoother drawing(?)
            if self.is_gate:
                pen.setWidth(max(1, int(round(4.0 / self.scale))))
            else:
                pen.setWidth(max(1, int(round(2.0 / self.scale))))
            painter.setPen(pen)

            line_path = QPainterPath()
            vrtx_path = QPainterPath()

            line_path.moveTo(self.points[0])
            # Uncommenting the following line will draw 2 paths
            # for the 1st vertex, and make it non-filled, which
            # may be desirable.
            # self.drawVertex(vrtx_path, 0)

            # print('self.points: {}'.format(self.points))

            for i, p in enumerate(self.points):
                line_path.lineTo(p)
                self.drawVertex(vrtx_path, i)
            if self.isClosed():
                line_path.lineTo(self.points[0])

            painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            painter.fillPath(vrtx_path, self.vertex_fill_color)
            if self.fill:
                color = self.select_fill_color if self.selected else self.fill_color
                painter.fillPath(line_path, color)

            if self.mask:
                line_path = QPainterPath()
                n_pts = len(self.mask)
                pt_id = 0
                line_segment_start = 1
                while pt_id < n_pts:
                    x, y, f = self.mask[pt_id]
                    pt_id += 1
                    if not f:
                        line_segment_start = 1
                        continue
                    _pt = QPointF(x, y)
                    if line_segment_start:
                        line_path.moveTo(_pt)
                        line_segment_start = 0
                    else:
                        line_path.lineTo(_pt)
                painter.drawPath(line_path)
            if self.has_text:
                self.drawText(painter)

    def drawText(self, painter):
        color = self.select_line_color if self.selected else self.line_color
        painter.setPen(color)

        # label = self.label if self.label else "undefined"
        id_number = str(self.id_number) if self.id_number else "N/A"
        heading_text = id_number

        if self.is_gate:
            if self.n_intersections > 0:
                heading_text = "{:s}:{:d}".format(heading_text, self.n_intersections)
            painter.setFont(QFont('Decorative', 5))
            mid_point = QPoint()
            if len(self.points) > 1:
                mid_point.setX((self.points[0].x() + self.points[1].x()) / 2)
                mid_point.setY((self.points[0].y() + self.points[1].y()) / 2)
            else:
                mid_point.setX(self.points[0].x())
                mid_point.setY(self.points[0].y())
            painter.drawText(mid_point, heading_text)
        else:
            painter.setFont(QFont('Decorative', 8))
            painter.drawText(self.boundingRect(), Qt.AlignTop, heading_text)
            # painter.drawText(self.boundingRect(), Qt.AlignBottom, self.bbox_source)

    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self.vertex_fill_color = self.hvertex_fill_color
        else:
            self.vertex_fill_color = self._vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        for i, p in enumerate(self.points):
            if distance(p - point) <= epsilon:
                return i
        return None

    def containsPoint(self, point):
        return self.makePath().contains(point)

    def makePath(self):
        path = QPainterPath(self.points[0])
        for p in self.points[1:]:
            path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        self._highlightIndex = None

    def copy(self):
        shape = Shape("%s" % self.label, bbox_source='ground_truth', is_gate=self.is_gate)
        shape.points = [p for p in self.points]
        shape.selected = self.selected
        shape._closed = self._closed
        shape.difficult = self.difficult
        shape.score = self.score
        return shape

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value

    def set_as_ground_truth(self):
        self.bbox_source = "ground_truth"
        self.set_colors()

    def set_hidden(self, bool):
        self.is_hidden = bool
