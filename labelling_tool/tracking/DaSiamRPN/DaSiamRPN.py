import os
import sys
import time
import math
import inspect
import copy
import logging

import numpy as np
import cv2
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from .DaSiamRPN_net import SiamRPNvot, SiamRPNBIG, SiamRPNotb
from .run_SiamRPN import generate_anchor, tracker_eval
from .DaSiamRPN_utils import get_subwindow_tracking


class DaSiamRPNParams:
    """

    :param int model: 0: SiamRPNvot 1: SiamRPNBIG 2: SiamRPNotb,
    :param str windowing: to penalize large displacements [cosine/uniform]
    :param int exemplar_size: input z size
    :param int instance_size: input x size (search region)
    :param float context_amount: context amount for the exemplar
    :param bool adaptive: adaptive change search region
    :param int score_size: size of score map
    :param int anchor_num: number of anchors
    """

    def __init__(self):
        self.windowing = 'cosine'
        self.exemplar_size = 127
        self.instance_size = 271
        self.total_stride = 8
        self.context_amount = 0.5
        self.ratios = (0.33, 0.5, 1, 2, 3)
        self.scales = (8,)

        self.penalty_k = 0.055
        self.window_influence = 0.42
        self.lr = 0.295
        self.adaptive = 0
        self.visualize = 0

        self.anchor_num = len(self.ratios) * len(self.scales)
        self.score_size = int((self.instance_size - self.exemplar_size) / self.total_stride + 1)

        self.gpu_id = 0
        self.model = 0
        self.update_location = 1
        self.rel_path = 1
        self.pretrained_wts_dir = 'pretrained_weights'

        self.help = {
        }

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = int((self.instance_size - self.exemplar_size) / self.total_stride + 1)
        self.anchor_num = len(self.ratios) * len(self.scales)


class DaSiamRPN:
    """
    :type params: DaSiamRPNParams
    :type logger: logging.RootLogger
    :type states: list[dict]
    """

    def __init__(self, params, logger, target_id=0,
                 label='generic', confidence=1.0):
        """
        :type params: DaSiamRPNParams
        :type logger: logging.RootLogger | None
        :type target_id: int
        :rtype: None
        """

        # self.tf_graph = tf.Graph()
        # avoid printing TF debugging information

        self._params = params
        self._logger = logger

        self.target_id = target_id
        self.label = label
        self.confidence = confidence
        self.cumulative_confidence = confidence

        if self._logger is None:
            self._logger = logging.getLogger()
            self._logger.setLevel(logging.INFO)
            # self.logger.handlers[0].setFormatter(logging.Formatter(
            #     '%(levelname)s::%(module)s::%(funcName)s::%(lineno)s :  %(message)s'))

        self.anchor = []

        # self.params.update(cfg={})

        self.associated_frames = 1
        self.unassociated_frames = 0
        self.associated = 0

        # self.is_initialized = 0

        self.bbox = None
        self.gpu_id = self._params.gpu_id

        self.pretrained_wts_dir = self._params.pretrained_wts_dir
        if self._params.rel_path:
            self.pretrained_wts_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.pretrained_wts_dir)

        self.net = None
        self.score_sz = self._params.score_size
        self.final_score_sz = self._params.score_size

        if self._params.update_location == 0:
            self._logger.info('Location updating is disabled')

        self.state = None

    def initialize(self, init_frame, init_bbox):
        """

        :param np.ndarray init_frame:
        :param np.ndarray | list | tuple init_bbox:
        :return:
        """

        if self.net is None:
            if self._params.model == 0:
                net = SiamRPNvot()
                net.load_state_dict(torch.load(os.path.join(self.pretrained_wts_dir, 'SiamRPNVOT.model')))
                # self._logger.info('Using SiamRPNVOT model')
            elif self._params.model == 1:
                net = SiamRPNBIG()
                net.load_state_dict(torch.load(os.path.join(self.pretrained_wts_dir, 'SiamRPNBIG.model')))
                # self._logger.info('Using SiamRPNBIG model')
            elif self._params.model == 2:
                net = SiamRPNotb()
                net.load_state_dict(torch.load(os.path.join(self.pretrained_wts_dir, 'SiamRPNOTB.model')))
                # self._logger.info('Using SiamRPNOTB model')
            else:
                raise IOError('Invalid model_type: {}'.format(self._params.model))

            net.eval().cuda(self.gpu_id)
            self.net = net

        cx, cy, target_w, target_h = init_bbox

        target_pos = np.array([cx, cy])
        target_sz = np.array([target_w, target_h])

        self._params.update(self.net.cfg)

        state = dict()
        state['im_h'] = init_frame.shape[0]
        state['im_w'] = init_frame.shape[1]

        if self._params.adaptive:
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                self._params.instance_size = 287  # small object big search region
            else:
                self._params.instance_size = 271

            self._params.score_size = (
                                              self._params.instance_size - self._params.exemplar_size) / self._params.total_stride + 1

        self.anchor = generate_anchor(self._params.total_stride, self._params.scales, self._params.ratios,
                                      int(self._params.score_size))

        avg_chans = np.mean(init_frame, axis=(0, 1))

        wc_z = target_sz[0] + self._params.context_amount * sum(target_sz)
        hc_z = target_sz[1] + self._params.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        z_crop = get_subwindow_tracking(init_frame, target_pos, self._params.exemplar_size, s_z, avg_chans)

        z = Variable(z_crop.unsqueeze(0))
        self.net.temple(z.cuda(self.gpu_id))

        if self._params.windowing == 'cosine':
            window = np.outer(np.hanning(self.score_sz), np.hanning(self.score_sz))
        elif self._params.windowing == 'uniform':
            window = np.ones((self.score_sz, self.score_sz))
        else:
            raise IOError('Invalid windowing type: {}'.format(self._params.windowing))
        window = np.tile(window.flatten(), self._params.anchor_num)

        # state['p'] = self.params

        pos_x, pos_y = target_pos
        target_w, target_h = target_sz

        xmin, ymin = pos_x - target_w / 2, pos_y - target_h / 2
        xmax, ymax = xmin + target_w, ymin + target_h

        bbox = [xmin, ymin, target_w, target_h]

        state['net'] = self.net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        self.bbox = [xmin, ymin, xmax, ymax]
        self.state = state

    def update(self, frame):
        state = self.state

        # p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        wc_z = target_sz[1] + self._params.context_amount * sum(target_sz)
        hc_z = target_sz[0] + self._params.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = self._params.exemplar_size / s_z
        d_search = (self._params.instance_size - self._params.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        x_crop = Variable(get_subwindow_tracking(frame, target_pos, self._params.instance_size,
                                                 round(s_x), avg_chans).unsqueeze(0))

        target_pos, target_sz, score, pscore, delta, score_id = tracker_eval(net, x_crop.cuda(self.gpu_id), target_pos,
                                                                             target_sz * scale_z, window,
                                                                             scale_z, self._params, self.anchor)

        score_map = np.reshape(score, (-1, self.score_sz, self.score_sz))
        pscore_map = np.reshape(pscore, (-1, self.score_sz, self.score_sz))
        delta_map = np.reshape(delta, (-1, self.score_sz, self.score_sz))

        unravel_id = np.unravel_index(score_id, score_map.shape)
        best_pscore_map = pscore_map[unravel_id[0], :, :].squeeze()

        best_pscore_map_max_idx = np.argmax(best_pscore_map)
        best_pscore_map_max_idx_ur = np.unravel_index(best_pscore_map_max_idx, best_pscore_map.shape)

        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

        if self._params.update_location:
            state['target_pos'] = target_pos
            state['target_sz'] = target_sz
        state['score'] = score
        state['pscore'] = pscore

        best_score = pscore[score_id]

        pos_x, pos_y = target_pos
        target_w, target_h = target_sz

        xmin, ymin = pos_x - target_w / 2, pos_y - target_h / 2
        xmax, ymax = xmin + target_w, ymin + target_h

        state['net'] = self.net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        self.bbox = [xmin, ymin, xmax, ymax]
        self.confidence = best_score
        self.cumulative_confidence *= best_score

        # self._logger.info('confidence: {}'.format(self.confidence))
        # self._logger.info('cumulative_confidence: {}'.format(self.cumulative_confidence))

        bbox = [xmin, ymin, target_w, target_h]

        return bbox

    def close(self):
        pass
