import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import os
import torch

from .SiamMask_utils.load_helper import load_pretrain
from .SiamMask_utils.config_helper import load_config

from .test_siammask import siamese_init, siamese_track

class SiamMaskParams:
    """
    """

    def __init__(self):
        self.arch = 'Custom'

        self.model_root = 'SiamMask/experiments/siammask_sharp'
        self.config = 'config_vot.json'
        self.resume = 'SiamMask_VOT.pth'
        self.mask = 1
        self.refine = 1
        self.mask_enable = True
        self.refine_enable = True
        self.cpu = 0
        self.help = {}



class SiamMask:
    """
    :type params: SiamMaskParams
    """

    def __init__(self, params, target_id=0,
                 label='generic', confidence=1.0):

        if params.model_root:
            params.config = os.path.join(params.model_root, params.config)
            params.resume = os.path.join(params.model_root, params.resume)

        self.mask_enable = params.mask_enable
        self.refine_enable = params.refine_enable

        self.target_id = target_id
        self.label = label
        self.confidence = confidence

        cfg = load_config(params)

        self.cfg = cfg
        self.hp = cfg['hp'] if 'hp' in cfg.keys() else None

        # setup model
        if params.arch == 'Custom':
            from .experiments.siammask_sharp.custom import Custom
            model = Custom(anchors=cfg['anchors'])
        else:
            raise IOError('invalid architecture: {}'.format(params.arch))

        if params.resume:
            assert os.path.isfile(params.resume), '{} is not a valid file'.format(params.resume)
            model = load_pretrain(model, params.resume)

        model.eval()
        self.device = torch.device('cuda' if (torch.cuda.is_available() and not params.cpu) else 'cpu')
        print('running on device: {}'.format(self.device))

        model = model.to(self.device)


        self.model = model
        self.state = None
        self.score = None
        self.mask = self.mask_pts = None
        self.target_pos = self.target_sz = None

        if torch.cuda.is_available():
            print('Using GPU: {}'.format(torch.cuda.get_device_name(0)))

    def initialize(self, im, bbox):
        cx, cy, w, h = bbox
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        self.state = siamese_init(im, target_pos, target_sz, self.model, self.hp, device=self.device)  # init tracker

    def update(self, im):
        # print(torch.cuda.is_available())
        # print(torch.cuda.current_device())
        # print(torch.cuda.device(0))
        # print(torch.cuda.device_count())
        # print(torch.cuda.get_device_name(0))

        self.state = siamese_track(self.state, im,
                                   self.mask_enable, self.refine_enable, device=self.device)  # track
        self.target_pos = self.state['target_pos']
        self.target_sz = self.state['target_sz']
        self.score = self.state['score']
        self.mask = self.state['mask']

        self.mask_pts = self.state['ploygon']

        cx, cy = self.target_pos
        w, h = self.target_sz

        x = cx - w / 2
        y = cy - h / 2

        return x, y, w, h

    def close(self):
        pass


