def x11_available():
    from subprocess import Popen, PIPE
    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0


from sys import platform

if platform in ("linux", "linux2") and not x11_available():
    """get rid of annoying error on ssh:
    Unable to init server: Could not connect: Connection refused
    Gdk-CRITICAL **: gdk_cursor_new_for_display: assertion 'GDK_IS_DISPLAY (display)' failed
    """
    import matplotlib as mpl

    mpl.use('Agg')

import numpy as np
import cv2
import subprocess
import filecmp
import os
import shutil
import copy
import sys
import time
from ast import literal_eval
from pprint import pformat
import json
import math
import functools
import inspect
import logging
from io import StringIO
from contextlib import contextmanager
from datetime import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate

import psutil

from colorlog import ColoredFormatter

import paramparse
from paramparse import MultiCFG, MultiPath

from tf_api.utilities import resizeAR

"""RGB values for different colors"""
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


class BaseParams:
    tee_log = ''


class IBTParams(BaseParams):
    """
    Iterative Batch Train Parameters
    has to be defined here instead of IBT to prevent circular dependency between Params and IBT

    :type cfgs: MultiCFG
    :type test_cfgs: MultiCFG
    :type async_dir: MultiPath
    :type states: MultiPath


    :ivar async_dir: 'Directory for saving the asynchronous training data',
    :ivar test_cfgs: 'cfg files and sections from which to read iteration specific configuration data '
                 'for testing and evaluation phases; '
                 'cfg files for different iterations must be separated by double colons followed by the '
                 'iteration id and a single colon; cfg files for any iteration can be provided in multiple '
                 'non contiguous units in which case they would be concatenated; '
                 'commas separate different cfg files for the same iteration and '
                 'single colons separate different sections for the same cfg file as usual; '
                 'configuration in the last provided iteration would be used for all subsequent '
                 'iterations as well unless an underscore (_) is used to revert to the global '
                 '(non iteration-specific) parameters; ',
    :ivar cfgs: 'same as test_cfgs except for the data generation and training phases '
            'which are specific to each '
            'state so that the iteration ID here includes both the iteration itself as well as the state; '
            'e.g. with 2 states:  iter 0, state 1  -> id = 01; iter 2, state 0 -> id = 20',
    :ivar start_iter: 'Iteration at which the start the training process',
    :ivar load: '0: Train from scratch '
            '1: load previously saved weights from the last iteration and continue training;'
            'Only applies if iter_id>0',
    :ivar states: 'states to train: one or more of [active, tracked, lost]',
    :ivar load_weights: '0: Train from scratch; '
                    '1: load previously saved weights and test; '
                    '2: load previously saved weights and continue training; ',
    :ivar min_samples: 'minimum number of samples generated in data_from_tester '
                   'for the policy to be considered trainable',
    :ivar accumulative: 'decides if training data from all previous iterations is added to that from '
                    'the current iteration for training',
    :ivar start: 'iteration at which the start the training process in the phase specified by start_phase',
    :ivar start_phase: 'Phase at which the start the training process in the iteration specified by start_id:'
                   '0: data generation / evaluation of previous iter'
                   '1: batch training '
                   '2: testing / evaluation of policy classifier '
                   '3: testing / evaluation of tracker ',
    :ivar start: 'single string specifying both start_id and start_phase by simple concatenation;'
             'e.g  start=12 means start_id=1 and start_phase=2; '
             'overrides both if provided',

    :ivar load_prev: continue training in the start iteration by loading weights from the same iteration
    saved in a previous run instead of loading them from previous iteration (if start iter > 0)

    """

    def __init__(self):
        self.start = ''
        self.start_iter = 0
        self.start_phase = 0
        self.start_state = 0
        self.data_from_tester = 0
        self.load = 0
        self.states = []
        self.skip_states = []
        self.n_iters = 5
        self.min_samples = 100
        self.accumulative = 0
        self.load_weights = 2
        self.save_suffix = ''
        self.load_prev = 0
        self.phases = ()
        self.test_iters = ()
        self.async_dir = MultiPath()
        self.cfgs = MultiCFG()
        self.test_cfgs = MultiCFG()

    def process(self):
        # self.async_dir = '_'.join(self.async_dir)
        if self.start:
            if ',' in self.start:
                start = list(map(int, self.start.split(',')))

                if len(start) == 3:
                    self.start_iter, self.start_phase, self.start_state = start
                elif len(start) == 2:
                    self.start_iter, self.start_phase = start
                else:
                    raise AssertionError(f'Invalid start IDs: {self.start}')
            else:
                if len(self.start) == 3:
                    self.start_iter = int(self.start[0])
                    self.start_phase = int(self.start[1])
                    self.start_state = int(self.start[2])
                elif len(self.start) == 2:
                    self.start_iter = int(self.start[0])
                    self.start_phase = int(self.start[1])
                elif len(self.start) == 1:
                    self.start_iter = int(self.start)
                else:
                    raise AssertionError(f'Invalid start IDs: {self.start}')

    def get_cfgs(self):
        n_states = len(self.states)

        valid_cfgs = [f'{iter_id}{state_id}' for iter_id in range(self.n_iters) for state_id in range(n_states)]
        return MultiCFG.to_dict(self.cfgs, valid_cfgs)

    def get_test_cfgs(self):
        valid_test_cfgs = list(map(str, range(self.n_iters)))
        return MultiCFG.to_dict(self.test_cfgs, valid_test_cfgs)


class MDPStates:
    inactive, active, tracked, lost = range(4)
    to_str = {
        0: 'inactive',
        1: 'active',
        2: 'tracked',
        3: 'lost',
    }
    # inactive, active, tracked, lost = ('inactive', 'active', 'tracked', 'lost')


class TrackingStatus:
    success, failure, unstable = range(1, 4)
    to_str = {
        1: 'success',
        2: 'failure',
        3: 'unstable',
    }


class AnnotationStatus:
    types = (
        'combined',
        'fp_background',
        'fp_deleted',
        'fp_apart',
        'fp_concurrent',
        'tp',
    )
    combined, fp_background, fp_deleted, fp_apart, fp_concurrent, tp = types


class SaveModes:
    valid = range(3)
    none, all, error = valid
    to_str = {
        0: 'none',
        1: 'all',
        2: 'error',
    }


def disable_vis(obj, args_in=None, prefix=''):
    """

    :param obj:
    :param list args_in:
    :param str prefix:
    :return:
    """
    if args_in is not None:
        for i, _arg in enumerate(args_in):
            if '+=' in _arg:
                _sep = '+='
            else:
                _sep = '='
            _arg_name, _arg_val = _arg.split(_sep)

            if _arg_name.endswith('.vis') or _arg_name.endswith('.visualize'):
                print(f'Disabling {_arg_name}')
                args_in[i] = '{}{}0'.format(_arg_name, _sep)

    obj_members = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]
    for member in obj_members:
        if member == 'help':
            continue
        member_val = getattr(obj, member)
        member_name = '{:s}.{:s}'.format(prefix, member) if prefix else member
        if not isinstance(member_val, (int, bool, float, str, tuple, list, dict,
                                       paramparse.MultiCFG, paramparse.MultiPath)):
            disable_vis(member_val, prefix=member_name)
        else:
            if member in ('vis', 'visualize') and isinstance(member_val, (int, bool)) and member_val:
                print(f'Disabling {member_name}')
                setattr(obj, member, 0)


def set_recursive(obj, name, val, prefix='', check_existence=1):
    """

    :param obj:
    :param list args_in:
    :param str prefix:
    :param check_existence int:
    :return:
    """
    if obj is None:
        return

    if not check_existence or hasattr(obj, name):
        setattr(obj, name, val)

    obj_members = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]

    for member in obj_members:
        if member == 'help':
            continue

        member_val = getattr(obj, member)
        member_name = '{:s}.{:s}'.format(prefix, member) if prefix else member
        if not isinstance(member_val, (int, bool, float, str, tuple, list, dict,
                                       paramparse.MultiCFG, paramparse.MultiPath)):
            set_recursive(member_val, name, val, prefix=member_name, check_existence=check_existence)


def list_to_str(vals, fmt='', sep='\t'):
    """

    :param list vals:
    :param fmt:
    :param sep:
    :return:
    """
    type_to_fmt = {
        int: '%d',
        float: '%.3f',
        bool: '%r',
        str: '%s',
    }
    if fmt:
        fmts = [fmt, ] * len(vals)
    else:
        fmts = []
        for val in vals:
            try:
                fmt = type_to_fmt[type(val)]
            except KeyError:
                fmt = type_to_fmt[type(val.item())]
            fmts.append(fmt)

    return sep.join(fmt % val for fmt, val in zip(fmts, vals))


def dict_to_str(vals, fmt='%.3f', sep='\t'):
    """

    :param dict vals:
    :param fmt:
    :param sep:
    :return:
    """
    return sep.join('{}: '.format(key) + fmt % val for key, val in vals.items())


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


class CustomLogger:
    """
    :type _backend: logging.RootLogger | logging.logger

    """

    def __init__(self, logger, names, key='custom_module'):
        """
        modify the custom module name header to append one or more names

        :param CustomLogger | logging.RootLogger logger:
        :param tuple | list names:
        """
        try:
            self._backend = logger.get_backend()
        except AttributeError:
            self._backend = logger

        self.handlers = self._backend.handlers
        self.addHandler = self._backend.addHandler
        self.removeHandler = self._backend.removeHandler

        try:
            k = logger.info.keywords['extra'][key]
        except BaseException as e:
            custom_log_header_str = '{}'.format(':'.join(names))
        else:
            custom_log_header_str = '{}:{}'.format(k, ':'.join(names))

        self.custom_log_header_tokens = custom_log_header_str.split(':')

        try:
            custom_log_header = copy.deepcopy(logger.info.keywords['extra'])
        except BaseException as e:
            custom_log_header = {}

        custom_log_header.update({key: custom_log_header_str})

        self.info = functools.partial(self._backend.info, extra=custom_log_header)
        self.warning = functools.partial(self._backend.warning, extra=custom_log_header)
        self.debug = functools.partial(self._backend.debug, extra=custom_log_header)
        self.error = functools.partial(self._backend.error, extra=custom_log_header)

        # try:
        #     self.info = functools.partial(self._backend.info.func, extra=custom_log_header)
        # except BaseException as e:
        #     self.info = functools.partial(self._backend.info, extra=custom_log_header)
        #     self.warning = functools.partial(self._backend.warning, extra=custom_log_header)
        #     self.debug = functools.partial(self._backend.debug, extra=custom_log_header)
        #     self.error = functools.partial(self._backend.error, extra=custom_log_header)
        # else:
        #     self.warning = functools.partial(self._backend.warning.func, extra=custom_log_header)
        #     self.debug = functools.partial(self._backend.debug.func, extra=custom_log_header)
        #     self.error = functools.partial(self._backend.error.func, extra=custom_log_header)

    def get_backend(self):
        return self._backend

    @staticmethod
    def add_file_handler(log_dir, _prefix, logger):
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        log_file = linux_path(log_dir, '{}_{}.log'.format(_prefix, time_stamp))
        logging_handler = logging.FileHandler(log_file)
        logger.addHandler(logging_handler)
        logging_fmt = logging.Formatter(
            '%(custom_header)s:%(custom_module)s:%(funcName)s:%(lineno)s :  %(message)s',
        )

        logger.handlers[-1].setFormatter(logging_fmt)
        return log_file, logging_handler

    # @staticmethod
    # def add_string_handler(logger):
    #     log_stream = StringIO()
    #     logging_handler = logging.StreamHandler(log_stream)
    #     # logging_handler.setFormatter(logger.handlers[0].formatter)
    #     logger.addHandler(logging_handler)
    #     # logger.string_stream = log_stream
    #     return logging_handler

    @staticmethod
    def remove_file_handler(logging_handler, logger):
        if logging_handler not in logger.handlers:
            return
        logging_handler.close()
        logger.removeHandler(logging_handler)

    @staticmethod
    def setup(name):
        # PROFILE_LEVEL_NUM = 9
        #
        # def profile(self, message, *args, **kws):
        #     if self.isEnabledFor(PROFILE_LEVEL_NUM):
        #         self._log(PROFILE_LEVEL_NUM, message, args, **kws)

        # logging.addLevelName(PROFILE_LEVEL_NUM, "PROFILE")
        # logging.Logger.profile = profile
        # logging.getLogger().addHandler(ColorHandler())

        # logging_level = logging.DEBUG
        # logging_level = PROFILE_LEVEL_NUM
        # logging.basicConfig(level=logging_level, format=logging_fmt)

        colored_logging_fmt = ColoredFormatter(
            '%(header_log_color)s%(custom_header)s:%(log_color)s%(custom_module)s:%(funcName)s:%(lineno)s : '
            ' %(message)s',
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={
                'header': {
                    'DEBUG': 'white',
                    'INFO': 'white',
                    'WARNING': 'white',
                    'ERROR': 'white',
                    'CRITICAL': 'white',
                }
            },
            style='%'
        )

        nocolor_logging_fmt = logging.Formatter(
            '%(custom_header)s:%(custom_module)s:%(funcName)s:%(lineno)s  :::  %(message)s',
        )
        # logging_fmt = logging.Formatter('%(custom_header)s:%(custom_module)s:%(funcName)s:%(lineno)s :  %(message)s')
        # logging_fmt = logging.Formatter('%(levelname)s::%(module)s::%(funcName)s::%(lineno)s :  %(message)s')

        logging_level = logging.DEBUG
        logging.basicConfig(level=logging_level,
                            # format=colored_logging_fmt
                            )

        _logger = logging.getLogger(name)

        if _logger.hasHandlers():
            _logger.handlers.clear()

        _logger.setLevel(logging_level)

        handler = logging.StreamHandler()
        handler.setFormatter(colored_logging_fmt)
        handler.setLevel(logging_level)
        _logger.addHandler(handler)

        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(nocolor_logging_fmt)
        handler.setLevel(logging_level)
        _logger.addHandler(handler)

        # CustomLogger.add_string_handler(_logger)

        class ContextFilter(logging.Filter):
            def filter(self, record):

                if not hasattr(record, 'custom_module'):
                    record.custom_module = record.module

                if not hasattr(record, 'custom_header'):
                    record.custom_header = record.levelname

                return True

        f = ContextFilter()
        _logger.addFilter(f)

        """avoid duplicate logging when logging used by other libraries"""
        _logger.propagate = False
        return _logger


@contextmanager
def profile(_id, _times, _rel_times, enable=1):
    """

    :param _id:
    :param dict _times:
    :param int enable:
    :return:
    """
    if not enable:
        yield None

    else:
        start_t = time.time()
        yield None
        end_t = time.time()
        _time = end_t - start_t

        print(f'{_id} :: {_time}')

        if _times is not None:

            _times[_id] = _time

            total_time = np.sum(list(_times.values()))

            if _rel_times is not None:

                for __id in _times:
                    rel__time = _times[__id] / total_time
                    _rel_times[__id] = rel__time

                rel_times = [(k, v) for k, v in sorted(_rel_times.items(), key=lambda item: item[1])]

                print(f'rel_times:\n {pformat(rel_times)}')


class VideoWriterGPU:
    def __init__(self, path, fps, size):
        self._path = path
        width, height = size

        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-codec', 'rawvideo',
                   '-s', f'{width}x{height}',  # size of one frame
                   '-pix_fmt', 'rgb24',
                   '-r', f'{fps}',  # frames per second
                   '-i', '-',  # The input comes from a pipe
                   '-an',  # Tells FFMPEG not to expect any audio
                   '-c:v', 'libx265',
                   # '-preset', 'medium',
                   '-preset', 'veryslow',
                   '-x265-params', 'lossless=0',
                   '-hide_banner',
                   '-loglevel', 'panic',

                   f'{self._path}']

        self._pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

        self._frame_id = 0

    def isOpened(self):
        if self._pipe is None:
            return False
        return True

    def write(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = image_rgb.tostring()
        self._pipe.stdin.write(im)
        self._pipe.stdin.flush()

    def release(self):
        self._pipe.stdin.close()
        self._pipe.wait()


class ImageSequenceCapture:
    """
    :param str src_path
    :param int recursive
    """

    def __init__(self, src_path='', recursive=0, img_exts=(), logger=None):
        self.src_path = ''
        self.src_fmt = ''
        self.recursive = 0
        self.img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.webp')
        self.src_files = []
        self.n_src_files = 0
        self.is_open = False
        self.frame_id = 0

        if src_path:
            if self.open(src_path, recursive, img_exts):
                self.is_open = True

    def is_opened(self, cv_prop):
        return self.is_open

    def read(self):
        if self.frame_id >= self.n_src_files:
            raise IOError('Invalid frame_id: {} for sequence with {} frames'.format(
                self.frame_id, self.n_src_files
            ))
        frame = cv2.imread(self.src_files[self.frame_id])
        self.frame_id += 1
        return True, frame

    def set(self, cv_prop, _id):
        if cv_prop == cv2.CAP_PROP_POS_FRAMES:
            print('Setting frame_id to : {}'.format(_id))
            self.frame_id = _id

    def get(self, cv_prop):
        if cv_prop == cv2.CAP_PROP_POS_FRAMES:
            return self.frame_id

    def open(self, src_path='', recursive=0, img_exts=()):
        if src_path:
            img_ext = os.path.splitext(os.path.basename(src_path))[1]
            if img_ext:
                self.src_path = os.path.dirname(src_path)
                self.src_fmt = os.path.basename(src_path)
                self.img_exts = (img_ext,)
            else:
                self.src_path = src_path
                self.src_fmt = ''

            self.recursive = recursive
        if img_exts:
            self.img_exts = img_exts

        if recursive:
            src_file_gen = [[linux_path(dirpath, f) for f in filenames if
                             os.path.splitext(f.lower())[1] in self.img_exts]
                            for (dirpath, dirnames, filenames) in os.walk(self.src_path, followlinks=True)]
            _src_files = [item for sublist in src_file_gen for item in sublist]
        else:
            _src_files = [linux_path(self.src_path, k) for k in os.listdir(self.src_path) if
                          os.path.splitext(k.lower())[1] in self.img_exts]

        if not _src_files:
            print('No images found in {}'.format(self.src_path))
            return False

        _src_files = [os.path.abspath(k) for k in _src_files]
        _src_files.sort(key=sort_key)

        self.src_files = _src_files
        self.n_src_files = len(self.src_files)

        if self.src_fmt:
            matching_files = [self.src_fmt % i for i in range(1, self.n_src_files + 1)]
            self.src_files = [k for k in self.src_files if os.path.basename(k) in matching_files]
            self.n_src_files = len(self.src_files)
        return True


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
        self.cmp_root_dirs = ('../../isl_labelling_tool/tracking_module/log', 'log')
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


def compute_overlaps_multi(iou, ioa_1, ioa_2, objects_1, objects_2, logger=None):
    """

    compute overlap between each pair of objects in two sets of objects
    can be used for computing overlap between all detections and annotations in a frame

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


def compute_overlap(iou, ioa_1, ioa_2, object_1, objects_2, logger=None, debug=False):
    """

    compute overlap of a single object with one or more objects
    specialized version for greater speed

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
def compute_self_overlaps(iou, ioa, boxes):
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


def get_max_overlap_obj(objects, location, _logger):
    """

    :param objects:
    :param location:
    :param _logger:
    :return:
    """
    if objects.shape[0] == 0:
        """no objects"""
        max_iou = 0
        max_iou_idx = None
    elif objects.shape[0] == 1:
        """single object"""
        iou = np.empty((1, 1))
        compute_overlap(iou, None, None, objects[0, 2:6].reshape((1, 4)),
                        location, _logger)
        max_iou_idx = 0
        max_iou = iou
    else:
        """get object with maximum overlap with the location"""
        iou = np.empty((objects.shape[0], 1))
        compute_overlaps_multi(iou, None, None, objects[:, 2:6],
                               location, _logger)
        max_iou_idx = np.argmax(iou, axis=0).item()
        max_iou = iou[max_iou_idx, 0]

    return max_iou, max_iou_idx


def log_debug_multi(logger, vars, names):
    log_str = ''
    log_dict = {}
    for i in range(len(vars)):
        log_str += '{:s}: %%({:d})s'.format(names[i], i + 1)
        log_dict['{:d}'.format(i + 1)] = vars[i]
    logger.debug(log_str, log_dict)


def draw_pts(img, pts):
    for _pt in pts:
        _pt = (int(_pt[0]), int(_pt[1]))
        cv2.circle(img, _pt, 1, (0, 0, 0), 2)


def draw_region(img, corners, color, thickness=1):
    # draw the bounding box specified by the given corners
    for i in range(4):
        p1 = (int(corners[0, i]), int(corners[1, i]))
        p2 = (int(corners[0, (i + 1) % 4]), int(corners[1, (i + 1) % 4]))
        cv2.line(img, p1, p2, color, thickness)


def draw_traj2(frame, _obj_centers_rec, obj_boxes=None, color='blue', thickness=1):
    n_traj = len(_obj_centers_rec)
    for __i in range(1, n_traj):
        pt1 = _obj_centers_rec[__i - 1]
        _obj_cx_rec, _obj_cy_rec = pt1
        pt1 = (int(_obj_cx_rec), int(_obj_cy_rec))

        pt2 = _obj_centers_rec[__i]
        _obj_cx_rec, _obj_cy_rec = pt2
        pt2 = (int(_obj_cx_rec), int(_obj_cy_rec))

        cv2.line(frame, pt1, pt2, col_rgb[color], thickness=thickness)

        if obj_boxes is not None:
            draw_box(frame, obj_boxes[__i], color=color, thickness=thickness)


def draw_boxes(frame, boxes, _id=None, color='black', thickness=2,
               is_dotted=0, transparency=0.):
    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, axis=0)
    for box in boxes:
        draw_box(frame, box, _id, color, thickness,
                 is_dotted, transparency)


def draw_box(frame, box, _id=None, color='black', thickness=2,
             is_dotted=0, transparency=0.):
    """
    :type frame: np.ndarray
    :type box: np.ndarray
    :type _id: int | str | None
    :param color: indexes into col_rgb
    :type color: str
    :type thickness: int
    :type is_dotted: int
    :type transparency: float
    :rtype: None
    """
    if np.any(np.isnan(box)):
        print('invalid location provided: {}'.format(box))
        return

    box = box.squeeze()
    pt1 = (int(box[0]), int(box[1]))
    pt2 = (int(box[0] + box[2]),
           int(box[1] + box[3]))

    if transparency > 0:
        _frame = np.copy(frame)
    else:
        _frame = frame

    if is_dotted:
        draw_dotted_rect(_frame, pt1, pt2, col_rgb[color], thickness=thickness)
    else:
        cv2.rectangle(_frame, pt1, pt2, col_rgb[color], thickness=thickness)

    if transparency > 0:
        frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...] = (
                frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...].astype(np.float32) * (1 - transparency) +
                _frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...].astype(np.float32) * transparency
        ).astype(frame.dtype)

    if _id is not None:
        if cv2.__version__.startswith('2'):
            font_line_type = cv2.CV_AA
        else:
            font_line_type = cv2.LINE_AA

        cv2.putText(frame, str(_id), (int(box[0] - 1), int(box[1] - 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col_rgb[color], 1, font_line_type)


def draw_trajectory(frame, trajectory, color='black', thickness=2, is_dotted=0):
    """
    :type frame: np.ndarray
    :type trajectory: list[np.ndarray]
    :param color: indexes into col_rgb
    :type color: str
    :type thickness: int
    :type is_dotted: int
    :rtype: None
    """

    n_traj = len(trajectory)
    for i in range(1, n_traj):
        pt1 = tuple(trajectory[i - 1].astype(np.int64))
        pt2 = tuple(trajectory[i].astype(np.int64))

        if is_dotted:
            draw_dotted_line(frame, pt1, pt2, col_rgb[color], thickness)
        else:
            try:
                cv2.line(frame, pt1, pt2, col_rgb[color], thickness=thickness)

            except TypeError:
                print('frame.dtype', frame.dtype)
                print('pt1', pt1)
                print('pt2', pt2)


def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=7):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    # if style == 'dotted':
    for p in pts:
        cv2.circle(img, p, thickness, color, -1)
    # else:
    #     s = pts[0]
    #     e = pts[0]
    #     i = 0
    #     for p in pts:
    #         s = e
    #         e = p
    #         if i % 2 == 1:
    #             cv2.line(img, s, e, color, thickness)
    #         i += 1


def draw_dotted_poly(img, pts, color, thickness=1):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_dotted_line(img, s, e, color, thickness)


def draw_dotted_rect(img, pt1, pt2, color, thickness=1):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    draw_dotted_poly(img, pts, color, thickness)


def write_to_files(root_dir, write_to_bin, entries):
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


def prob_to_rgb(value, minimum=0., maximum=1.):
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def prob_to_rgb2(value, minimum=0., maximum=1.):
    ratio = (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(255 - b)
    g = 0
    return r, g, b


def build_targets_3d(frames, annotations, grid_res=(19, 19), frame_gap=1, win_size=100,
                     diff_grid_size=10, one_hot=1):
    """

    :param Annotations annotations:
    :param list[np.ndarray] frames:
    :param tuple(int, int) grid_res:
    :param int frame_gap: frame_gap
    :param int win_size: temporal window size
    :return:

    """
    n_frames = len(frames)
    frame_size = (frames[1].shape[1], frames[1].shape[0])

    diff_grid_res = np.array([frame_size[i] / diff_grid_size for i in range(2)])

    end_frame = n_frames - win_size
    grid_cell_size = np.array([frame_size[i] / grid_res[i] for i in range(2)])

    """grid cell centers
    """
    grid_x, grid_y = [np.arange(grid_cell_size[i] / 2.0, frame_size[i], grid_cell_size[i]) for i in range(2)]
    grid_cx, grid_cy = np.meshgrid(grid_x, grid_y)

    n_grid_cells = grid_cx.size

    ann_sizes = annotations.data[:, 4:6]
    ann_centers = annotations.data[:, 2:4] + annotations.data[:, 4:6] / 2.0

    for frame_id in range(0, end_frame, frame_gap):

        ann_idx = annotations.idx[frame_id]
        obj_centers = ann_centers[ann_idx, :]

        """Map each object to the grid that contains its centre
        """
        obj_grid_ids = (obj_centers.T / grid_cell_size[:, None]).astype(np.int64)
        # arr = np.array([[3, 6, 6], [4, 5, 1]])

        """np.ravel_multi_index takes row, col indices"""
        obj_grid_ids_flat = np.ravel_multi_index(obj_grid_ids[::-1], grid_res)

        u, c = np.unique(obj_grid_ids_flat, return_counts=True)
        dup_grid_ids = u[c > 1]
        active_grid_ids = list(u[c == 1])
        active_obj_ids = [np.nonzero(obj_grid_ids_flat == _id)[0].item() for _id in active_grid_ids]

        obj_cols = (
            'forest_green', 'blue', 'red', 'cyan', 'magenta', 'gold', 'purple', 'peach_puff', 'azure',
            'dark_slate_gray',
            'navy', 'turquoise')

        """Resolve multiple objects mapping to the same grid by choosing the nearest object in each case
        and add it to the list of active objects for this temporal window
        """
        for _id in dup_grid_ids:
            grid_ids = np.nonzero(obj_grid_ids_flat == _id)[0]

            dup_obj_locations = obj_centers[grid_ids, :]

            _id_2d = np.unravel_index(_id, grid_res)

            grid_center = np.array((grid_x[_id_2d[0]], grid_y[_id_2d[1]])).reshape((1, 2))

            dup_obj_distances = np.linalg.norm(dup_obj_locations - grid_center, axis=1)

            nearest_obj_id = grid_ids[np.argmin(dup_obj_distances)]

            active_obj_ids.append(nearest_obj_id)

            active_grid_ids.append(_id)

        active_obj_ids_ann = [int(annotations.data[ann_idx[i], 1]) for i in active_obj_ids]

        end_frame_id = min(frame_id + win_size, n_frames) - 1

        frame_disp = np.copy(frames[frame_id])

        """show grid cells"""
        for grid_id in range(n_grid_cells):
            prev_grid_idy, prev_grid_idx = np.unravel_index(grid_id, grid_res)

            offset_cx, offset_cy = grid_cx[prev_grid_idy, prev_grid_idx], grid_cy[prev_grid_idy, prev_grid_idx]

            grid_box = np.array(
                [offset_cx - grid_cell_size[0] / 2, offset_cy - grid_cell_size[1] / 2, grid_cell_size[0],
                 grid_cell_size[1]])

            draw_box(frame_disp, grid_box, _id=grid_id, color='black')

        """ahow active objects and associated grid cells 
        """
        for _id, obj_id in enumerate(active_obj_ids):
            # traj_idx = annotations.traj_idx[obj_id]
            # curr_ann_data = annotations.data[, :]
            # curr_frame_ann_idx = np.flatnonzero(curr_ann_data[:, 0] == frame_id)
            # ann_idx = traj_idx[curr_frame_ann_idx]

            obj_data = annotations.data[ann_idx[obj_id], :]

            # obj_ann_idx = annotations.traj_idx[obj_id]
            active_grid_id = active_grid_ids[_id]

            grid_idy, grid_idx = np.unravel_index(active_grid_id, grid_res)

            cx, cy = grid_cx[grid_idy, grid_idx], grid_cy[grid_idy, grid_idx]

            grid_box = np.array([cx - grid_cell_size[0] / 2, cy - grid_cell_size[1] / 2,
                                 grid_cell_size[0], grid_cell_size[1]])

            obj_col = obj_cols[_id % len(obj_cols)]

            draw_box(frame_disp, obj_data[2:6], _id=obj_data[1], color=obj_col)
            # show('frame_disp', frame_disp, _pause=0)

            draw_box(frame_disp, grid_box, color=obj_col)

        show('frame_disp', frame_disp, _pause=0)

        """Maximum possible distance between the centre of an object and the centres of 
        all of the neighbouring grid cells
        """
        max_dist = 1.5 * np.sqrt((grid_cell_size[0] ** 2 + grid_cell_size[1] ** 2))

        _pause = 100

        """compute distances and dist_probabilities for each active object wrt each of the 9 neighboring cells 
        in each frame of the current temporal window
        """
        """iterate over active objects from first frame of temporal window"""
        for _id, obj_id in enumerate(active_obj_ids_ann):

            # obj_id2 = active_obj_ids[_id]

            """all annotations for this object in the temporal window"""
            obj_ann_idx = annotations.traj_idx[obj_id]
            obj_ann_idx = [k for k in obj_ann_idx if annotations.data[k, 0] <= end_frame_id]
            obj_ann_data = annotations.data[obj_ann_idx, :]

            curr_obj_sizes = ann_sizes[obj_ann_idx, :]
            curr_obj_centers = ann_centers[obj_ann_idx, :]
            """Map each object to the grid that contains its centre
            """
            curr_obj_grid_ids = (curr_obj_centers.T / grid_cell_size[:, None]).astype(np.int64)
            # arr = np.array([[3, 6, 6], [4, 5, 1]])

            """np.ravel_multi_index takes row, col indices"""

            curr_obj_grid_ids_flat = np.ravel_multi_index(curr_obj_grid_ids[::-1], grid_res)

            obj_col = obj_cols[_id % len(obj_cols)]

            _prev_grid_id = None
            _obj_centers_rec = []
            _obj_centers = []
            _one_hot_obj_centers_rec = []

            """Iterate over objects corresponding to this target in each frame of the temporal window"""
            for temporal_id, curr_grid_id in enumerate(curr_obj_grid_ids_flat):
                if _prev_grid_id is None:
                    _prev_grid_id = curr_grid_id

                prev_grid_idy, prev_grid_idx = np.unravel_index(_prev_grid_id, grid_res)
                curr_grid_idy, curr_grid_idx = np.unravel_index(curr_grid_id, grid_res)

                obj_cx, obj_cy = curr_obj_centers[temporal_id, :]
                obj_w, obj_h = curr_obj_sizes[temporal_id, :]

                prev_cx, prev_cy = grid_cx[prev_grid_idy, prev_grid_idx], grid_cy[prev_grid_idy, prev_grid_idx]
                curr_cx, curr_cy = grid_cx[curr_grid_idy, curr_grid_idx], grid_cy[curr_grid_idy, curr_grid_idx]

                prev_grid_box = np.array([prev_cx - grid_cell_size[0] / 2, prev_cy - grid_cell_size[1] / 2,
                                          grid_cell_size[0], grid_cell_size[1]])
                curr_grid_box = np.array([curr_cx - grid_cell_size[0] / 2, curr_cy - grid_cell_size[1] / 2,
                                          grid_cell_size[0], grid_cell_size[1]])

                diff_cx, diff_cy = curr_cx - prev_cx, curr_cy - prev_cy

                """find quadrant of motion direction
                """
                if diff_cx >= 0:
                    """move right - quadrant 1 or 4"""
                    if diff_cy >= 0:
                        """move up"""
                        quadrant = 1
                    else:
                        """move down"""
                        quadrant = 4
                else:
                    """move left - quadrant 2 or 3"""
                    if diff_cy >= 0:
                        """move up"""
                        quadrant = 2
                    else:
                        """move down"""
                        quadrant = 3

                # dist_probabilities = np.zeros((3, 3))
                distances = []
                distances_inv = []
                dist_ids = []
                grid_centers = []

                neigh_grid_ids = []

                one_hot_probabilities = np.zeros((9,), dtype=np.float32)
                if prev_grid_idy == curr_grid_idy:
                    """middle row"""
                    if prev_grid_idx == curr_grid_idx:
                        class_id = 4
                    elif prev_grid_idx > curr_grid_idx:
                        class_id = 3
                    else:
                        class_id = 5
                elif prev_grid_idy > curr_grid_idy:
                    """bottom row"""
                    if prev_grid_idx == curr_grid_idx:
                        class_id = 7
                    elif prev_grid_idx > curr_grid_idx:
                        class_id = 6
                    else:
                        class_id = 8
                else:
                    """top row"""
                    if prev_grid_idx == curr_grid_idx:
                        class_id = 4
                    elif prev_grid_idx > curr_grid_idx:
                        class_id = 1
                    else:
                        class_id = 7

                one_hot_probabilities[class_id] = 1
                obj_cx_rec, obj_cy_rec = grid_cx[curr_grid_idy, curr_grid_idx], grid_cy[curr_grid_idy, curr_grid_idx]

                _one_hot_obj_centers_rec.append((obj_cx_rec, obj_cy_rec))
                _obj_centers.append((obj_cx, obj_cy))

                """smooth probabilities based on object's distance from grid centers"""

                """Iterate over neighbouring grid cells"""
                for prob_idx, offset_x in enumerate((-1, 0, 1)):
                    offset_idx = prev_grid_idx + offset_x
                    if offset_idx >= grid_res[0]:
                        continue
                    for prob_idy, offset_y in enumerate((-1, 0, 1)):
                        offset_idy = prev_grid_idy + offset_y
                        if offset_idy >= grid_res[1]:
                            continue

                        neigh_grid_ids.append((offset_idy, offset_idx))

                        offset_cx, offset_cy = grid_cx[offset_idy, offset_idx], grid_cy[offset_idy, offset_idx]
                        dist = np.sqrt((obj_cx - offset_cx) ** 2 + (obj_cy - offset_cy) ** 2) / max_dist

                        assert 0 <= dist <= max_dist, f"Invalid distance: {dist}"

                        norm_dist = dist / max_dist

                        """Large distance = small probability"""
                        # inv_dist = 1.0 / (1.0 + dist)
                        inv_dist = 1.0 - norm_dist

                        distances.append(norm_dist)
                        distances_inv.append(inv_dist)
                        dist_ids.append((prob_idy, prob_idx))
                        grid_centers.append((offset_cx, offset_cy))

                distances = np.asarray(distances)
                distances_inv = np.asarray(distances_inv)
                grid_centers = np.asarray(grid_centers)

                """sum to 1"""
                # dist_probabilities = np.exp(distances_inv) / sum(np.exp(distances_inv))
                distances_inv_sum = np.sum(distances_inv)
                dist_inv_probabilities = distances_inv / distances_inv_sum

                distances_sum = np.sum(distances)
                dist_probabilities = distances / distances_sum
                dist_probabilities2 = 1 - dist_probabilities

                eps = np.finfo(np.float32).eps
                dist_probabilities3 = 1.0 / (eps + dist_probabilities)
                dist_probabilities3_sum = np.sum(dist_probabilities3)
                dist_probabilities4 = dist_probabilities3 / dist_probabilities3_sum

                obj_cx_rec = np.average(grid_centers[:, 0], weights=dist_probabilities4)
                obj_cy_rec = np.average(grid_centers[:, 1], weights=dist_probabilities4)

                _obj_centers_rec.append((obj_cx_rec, obj_cy_rec))

                obj_cx_diff = obj_cx_rec - obj_cx
                obj_cy_diff = obj_cy_rec - obj_cy

                obj_box = np.array([obj_cx - obj_w / 2, obj_cy - obj_h / 2,
                                    obj_cx + obj_w / 2, obj_cy + obj_h / 2])

                obj_box_rec = np.array([obj_cx_rec - obj_w / 2, obj_cy_rec - obj_h / 2,
                                        obj_cx_rec + obj_w / 2, obj_cy_rec + obj_h / 2])

                obj_box_rec_iou = np.empty((1,))
                compute_overlap(obj_box_rec_iou, None, None, obj_box.reshape((1, 4)),
                                obj_box_rec.reshape((1, 4)))

                min_probability, max_probability = np.min(dist_probabilities4), np.max(dist_probabilities4)
                dist_probabilities_norm = (dist_probabilities4 - min_probability) / (max_probability - min_probability)

                """Large distance = small probability"""
                # dist_probabilities = 1.0 - dist_probabilities

                """show dist_probabilities as a 2D image with 3x3 grid
                """
                prob_img = np.zeros((300, 300, 3), dtype=np.uint8)
                one_hot_prob_img = np.zeros((300, 300, 3), dtype=np.uint8)

                for _prob_id, _prob in enumerate(dist_probabilities_norm):
                    prob_idy, prob_idx = dist_ids[_prob_id]
                    _prob_col = prob_to_rgb2(_prob)
                    r, g, b = _prob_col
                    start_row = prob_idy * 100
                    start_col = prob_idx * 100

                    end_row = start_row + 100
                    end_col = start_col + 100

                    prob_img[start_row:end_row, start_col:end_col, :] = (b, g, r)

                    one_hot_prob_col = prob_to_rgb2(one_hot_probabilities[_prob_id])
                    r, g, b = one_hot_prob_col
                    one_hot_prob_img[start_row:end_row, start_col:end_col, :] = (b, g, r)

                obj_data = obj_ann_data[temporal_id, :]

                curr_frame_id = int(obj_data[0])
                curr_frame = frames[curr_frame_id]

                curr_frame_disp_grid = np.copy(curr_frame)

                """draw all grid cells"""
                for grid_id in range(n_grid_cells):
                    _grid_idy, _grid_idx = np.unravel_index(grid_id, grid_res)

                    _cx, _cy = grid_cx[_grid_idy, _grid_idx], grid_cy[_grid_idy, _grid_idx]

                    grid_box = np.array(
                        [_cx - grid_cell_size[0] / 2, _cy - grid_cell_size[1] / 2, grid_cell_size[0],
                         grid_cell_size[1]])

                    # col = 'red' if (prev_grid_idy, prev_grid_idx) in neigh_grid_ids else 'black'
                    draw_box(curr_frame_disp_grid, grid_box, color='black')

                """draw neighboring grid cells"""
                col = 'green' if _prev_grid_id == curr_grid_id else 'red'
                for _grid_idy, _grid_idx in neigh_grid_ids:
                    _cx, _cy = grid_cx[_grid_idy, _grid_idx], grid_cy[_grid_idy, _grid_idx]

                    grid_box = np.array(
                        [_cx - grid_cell_size[0] / 2, _cy - grid_cell_size[1] / 2, grid_cell_size[0],
                         grid_cell_size[1]])
                    draw_box(curr_frame_disp_grid, grid_box, color=col)

                draw_box(curr_frame_disp_grid, obj_data[2:6], _id=obj_data[1], color='blue')

                curr_frame_traj_rec = np.copy(curr_frame)
                draw_box(curr_frame_traj_rec, obj_data[2:6], color='blue', thickness=1)
                draw_traj2(curr_frame_traj_rec, _obj_centers_rec, color='red')
                draw_traj2(curr_frame_traj_rec, _obj_centers, color='green')
                # curr_frame_traj_rec = resize_ar(curr_frame_traj_rec, 1920, 1080)

                curr_frame_traj_one_hot_rec = np.copy(curr_frame)
                draw_box(curr_frame_traj_one_hot_rec, obj_data[2:6], color='blue', thickness=1)
                draw_traj2(curr_frame_traj_one_hot_rec, _one_hot_obj_centers_rec, color='red')
                draw_traj2(curr_frame_traj_one_hot_rec, _obj_centers, color='green')
                # curr_frame_traj_one_hot_rec = resize_ar(curr_frame_traj_one_hot_rec, 1920, 1080)

                # curr_frame_traj = np.copy(curr_frame)
                # draw_box(curr_frame_traj, obj_data[2:6], color='blue', thickness=1)
                # draw_traj2(curr_frame_traj, _obj_centers, color='green')
                # curr_frame_traj = resize_ar(curr_frame_traj, 1920, 1080)

                for _obj_cx_rec, _obj_cy_rec in _obj_centers_rec:
                    cv2.circle(curr_frame_disp_grid, (int(_obj_cx_rec), int(_obj_cy_rec)), 1, color=(255, 255, 255),
                               thickness=2)

                curr_frame_disp = np.copy(curr_frame)

                draw_box(curr_frame_disp, prev_grid_box, color='red')
                draw_box(curr_frame_disp, curr_grid_box, color='green')

                draw_box(curr_frame_disp, obj_data[2:6], _id=obj_data[1], color=obj_col)

                show('curr_frame_disp_grid', curr_frame_disp_grid, _pause=0)
                show('curr_frame_disp', curr_frame_disp, _pause=0)
                show('one_hot_prob_img', one_hot_prob_img, _pause=0)
                # show('curr_frame_traj', curr_frame_traj, _pause=0)
                show('curr_frame_traj_rec', curr_frame_traj_rec, _pause=0)
                show('curr_frame_traj_one_hot_rec', curr_frame_traj_one_hot_rec, _pause=0)
                _pause = show('prob_img', prob_img, _pause=_pause)

                _prev_grid_id = curr_grid_id

                # print()

            print()

        print()


def build_targets_seq(frames, annotations, frame_gap=1, win_size=50):
    """

    :param Annotations annotations:
    :param list[np.ndarray] frames:
    :param tuple(int, int) grid_res:
    :param int frame_gap: frame_gap
    :param int win_size: temporal window size
    :return:

    """
    n_frames = len(frames)
    frame_size = (frames[1].shape[1], frames[1].shape[0])

    end_frame = n_frames - win_size

    ann_mins = annotations.data[:, 2:4]
    ann_maxs = annotations.data[:, 2:4] + annotations.data[:, 4:6]
    ann_sizes = annotations.data[:, 4:6]
    ann_obj_ids = annotations.data[:, 1]
    ann_centers = annotations.data[:, 2:4] + annotations.data[:, 4:6] / 2.0

    for frame_id in range(0, end_frame, frame_gap):

        ann_idx = annotations.idx[frame_id]
        obj_centers = ann_centers[ann_idx, :]
        obj_mins = ann_mins[ann_idx, :]
        obj_maxs = ann_maxs[ann_idx, :]

        curr_frame_data = annotations.data[ann_idx, :]

        curr_ann_obj_ids = ann_obj_ids[ann_idx]

        # obj_cols = (
        #     'forest_green', 'blue', 'red', 'cyan', 'magenta', 'gold', 'purple', 'peach_puff', 'azure',
        #     'dark_slate_gray',
        #     'navy', 'turquoise')

        end_frame_id = min(frame_id + win_size, n_frames) - 1

        _pause = 100

        label_txt = 'frames {} --> {}'.format(frame_id + 1, end_frame_id + 1)

        """compute distances and dist_probabilities for each active object wrt each of the 9 neighboring cells 
        in each frame of the current temporal window
        """
        """iterate over active objects from first frame of temporal window"""
        input_attention_map = np.full(frames[1].shape[:2], 255, dtype=np.uint8)
        for _id, obj_id in enumerate(curr_ann_obj_ids):
            traj_id = annotations.obj_to_traj[obj_id]
            # obj_id2 = active_obj_ids[_id]

            """all annotations for this object in the temporal window"""
            obj_ann_idx = annotations.traj_idx[traj_id]
            obj_ann_idx = [k for k in obj_ann_idx if annotations.data[k, 0] <= end_frame_id]
            obj_ann_data = annotations.data[obj_ann_idx, :]

            curr_obj_sizes = ann_sizes[obj_ann_idx, :]
            curr_obj_centers = ann_centers[obj_ann_idx, :]
            curr_obj_mins = ann_mins[obj_ann_idx, :]
            curr_obj_maxs = ann_maxs[obj_ann_idx, :]

            # obj_col = obj_cols[_id % len(obj_cols)]

            _prev_grid_id = None
            _obj_centers_rec = []
            _obj_boxes = []
            _obj_centers = []
            _one_hot_obj_centers_rec = []

            n_obj_ann_data = obj_ann_data.shape[0]

            all_objects = np.copy(frames[frame_id])

            for __id, __obj_id in enumerate(curr_ann_obj_ids):
                if __id == _id:
                    col = 'green'
                elif __id < _id:
                    col = 'red'
                else:
                    col = 'black'

                draw_box(all_objects, curr_frame_data[__id, 2:6], color=col, thickness=2)

            obj_minx, obj_miny = curr_obj_mins[0, :].astype(np.int)
            obj_maxx, obj_maxy = curr_obj_maxs[0, :].astype(np.int)

            output_attention_map = np.full(frames[1].shape[:2], 0, dtype=np.uint8)
            output_attention_map[obj_miny:obj_maxy, obj_minx:obj_maxx] = 255
            _label_txt = label_txt + ' object {}'.format(_id + 1)

            show('input_attention_map', input_attention_map,
                 # text=_label_txt, n_modules=0
                 )
            show('output_attention_map', output_attention_map,
                 # text=_label_txt, n_modules=0
                 )

            """Iterate over objects corresponding to this target in each frame of the temporal window"""
            for temporal_id, curr_grid_id in enumerate(range(n_obj_ann_data)):
                obj_cx, obj_cy = curr_obj_centers[temporal_id, :]
                obj_data = obj_ann_data[temporal_id, :]

                # obj_w, obj_h = curr_obj_sizes[temporal_id, :]
                # obj_minx, obj_miny = curr_obj_mins[temporal_id, :]
                # obj_maxx, obj_maxy = curr_obj_maxs[temporal_id, :]

                _obj_centers.append((obj_cx, obj_cy))

                curr_frame_id = int(obj_data[0])
                curr_frame = frames[curr_frame_id]

                _obj_boxes.append(obj_data[2:6])

                output_boxes = np.copy(curr_frame)
                # draw_box(output_boxes, obj_data[2:6], color='green', thickness=1)
                draw_traj2(output_boxes, _obj_centers, _obj_boxes, color='green')
                # output_boxes = resize_ar(output_boxes, 1600, 900)

                annotate_and_show('output_boxes', output_boxes, text=_label_txt, n_modules=0)

            annotate_and_show('all_objects', all_objects, text=_label_txt, n_modules=0)

            input_attention_map[obj_miny:obj_maxy, obj_minx:obj_maxx] = 0


def compare_files(read_from_bin, files, dirs=None, sync_id=-1, msg=''):
    """
    :type target: target
    :type read_from_bin: bool | int
    :type files: list[(str, Type(np.dtype), tuple)]
    :type dirs: (str, ...) | None
    :type sync_id: int
    :type msg: str
    :rtype: bool
    """

    if dirs is None:
        params = DebugParams()
        dirs = params.cmp_root_dirs

    if read_from_bin:
        file_ext = 'bin'
    else:
        file_ext = 'txt'

    self_dir = os.path.abspath(dirs[1])
    # print('self_dir: {}'.format(self_dir))

    if not dirs[0]:
        if sync_id >= 0:
            sync_fname = '{:s}/write_{:d}.sync'.format(self_dir, sync_id)
            open(sync_fname, 'w').close()

            # sys.stdout.write('{:s} Wrote {:s}...\n'.format(msg, sync_fname))
            # sys.stdout.flush()

            sync_fname = '{:s}/read_{:d}.sync'.format(self_dir, sync_id)
            sys.stdout.write('\n{:s} Waiting for {:s}...'.format(msg, sync_fname))
            sys.stdout.flush()
            iter_id = 0
            while not os.path.isfile(sync_fname):
                time.sleep(0.5)

            sys.stdout.write('\n')
            sys.stdout.flush()

            while True:
                try:
                    os.remove(sync_fname)
                except PermissionError:
                    time.sleep(0.5)
                else:
                    break
        return

    other_dir = os.path.abspath(dirs[0])
    # print('other_dir: {}'.format(other_dir))

    if sync_id >= 0:
        sync_fname = '{:s}/write_{:d}.sync'.format(other_dir, sync_id)
        sys.stdout.write('\n{:s} Waiting for {:s}...'.format(msg, sync_fname))
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
        while True:
            try:
                os.remove(sync_fname)
            except PermissionError:
                time.sleep(0.5)
            else:
                break

    files_are_same = True
    array_1 = {}
    array_2 = {}
    diff_array = {}
    equality_array = {}
    for fname, ftype, fshape in files:
        path_1 = '{:s}/{:s}.{:s}'.format(other_dir, fname, file_ext)
        path_2 = '{:s}/{:s}.{:s}'.format(self_dir, fname, file_ext)

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
            if read_from_bin:
                array_1[fname] = np.fromfile(path_1, dtype=ftype).reshape(fshape)
                array_2[fname] = np.fromfile(path_2, dtype=ftype).reshape(fshape)
                diff_array[fname] = np.abs(array_1[fname] - array_2[fname])
                equality_array[fname] = array_1[fname] == array_2[fname]
            else:
                subprocess.call('diff {:s} {:s} > {:s}/{:s}.diff'.format(
                    path_1, path_2, other_dir, fname), shell=True)
    if not files_are_same:
        print('paused')

    if sync_id >= 0:
        sync_fname = '{:s}/read_{:d}.sync'.format(other_dir, sync_id)
        open(sync_fname, 'w').close()

        # sys.stdout.write('{:s} Wrote {:s}...\n'.format(msg, sync_fname))
        # sys.stdout.flush()

    return files_are_same


class SIIF:
    @staticmethod
    def setup():
        # os.environ["SIIF_DUMP_IMAGES"] = "0"
        procs = []
        for proc in psutil.process_iter():
            try:
                # Get process name & pid from process object.
                process_name = proc.name()
                process_id = proc.pid
                cmdline = proc.cmdline()
                cmdline_txt = ' '.join(cmdline)
                # for _cmd in cmdline:
                #     cmdline_txt += ' ' + _cmd
                procs.append((process_name, cmdline_txt, process_id))
                # print(process_name, ' ::: ', process_id)

                siif_path = [k for k in cmdline if 'show_images_in_folder.py' in k]

                if process_name.startswith('python3') and siif_path:
                    siif_path = os.path.abspath(siif_path[0])
                    siif_dir = os.path.dirname(siif_path)
                    siif_log_path = os.path.join(siif_dir, 'siif_log.txt')
                    if not os.path.isfile(siif_log_path):
                        raise IOError(f'siif_log_path does not exist: {siif_log_path}')
                    with open(siif_log_path, 'r') as fid:
                        siif_src_path = fid.readline()

                    print("SIIF is active at {}".format(siif_src_path))
                    # os.environ["SIIF_DUMP_IMAGES"] = "1"
                    os.environ["SIIF_PATH"] = siif_src_path

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        # procs.sort(key=lambda x: x[0])
        # print(pformat(procs))
        # print(pformat(os.environ))
        # exit()

    @staticmethod
    def imshow(title, img):
        try:
            siif_path = os.environ["SIIF_PATH"]
        except KeyError:
            cv2.imshow(title, img)
            return 0

        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        out_path = os.path.join(siif_path, '{}___{}.bmp'.format(title, time_stamp))

        # cv2.imshow('siif', img)
        cv2.imwrite(out_path, img)

        while os.path.exists(out_path):
            continue

        return 1


class CVText:
    def __init__(self, color='white', bkg_color='black', location=0, font=5,
                 size=0.8, thickness=1, line_type=2, offset=(5, 25)):
        self.color = color
        self.bkg_color = bkg_color
        self.location = location
        self.font = font
        self.size = size
        self.thickness = thickness
        self.line_type = line_type
        self.offset = offset

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


def show(title, frame_disp, _pause=1):
    if SIIF.imshow(title, frame_disp):
        return _pause

    # cv2.imshow(title, frame_disp)

    if _pause > 1:
        wait = _pause
    else:
        wait = 1 - _pause

    k = cv2.waitKey(wait)

    if k == 27:
        sys.exit()
    if k == 32:
        _pause = 1 - _pause

    return _pause


import traceback


def modules_from_trace(call_stack, n_modules, start_module=1):
    """

    :param list[traceback.FrameSummary] call_stack:
    :param int n_modules:
    :param int start_module:
    :return:
    """
    call_stack = call_stack[::-1]

    modules = []

    for module_id in range(start_module, start_module + n_modules):
        module_fs = call_stack[module_id]
        file = os.path.splitext(os.path.basename(module_fs.filename))[0]
        line = module_fs.lineno
        func = module_fs.name

        modules.append('{}:{}:{}'.format(file, func, line))

    modules_str = '<'.join(modules)
    return modules_str


def annotate_and_show(title, img_list, text=None, pause=1,
                      fmt=CVText(), no_resize=1, grid_size=(-1, 1), n_modules=3,
                      use_plt=0, max_width=0, max_height=0, only_annotate=0):
    """

    :param str title:
    :param np.ndarray | list | tuple img_list:
    :param str | logging.RootLogger | CustomLogger text:
    :param int pause:
    :param CVText fmt:
    :param int no_resize:
    :param int n_modules:
    :param int use_plt:
    :param tuple(int) grid_size:
    :return:
    """

    # call_stack = traceback.format_stack()
    # print(pformat(call_stack))
    # for line in traceback.format_stack():
    #     print(line.strip())

    if isinstance(text, (logging.RootLogger, CustomLogger)):
        string_stream = [k.stream for k in text.handlers if isinstance(k.stream, StringIO)]
        assert string_stream, "No string streams in logger"
        _str = string_stream[0].getvalue()
        _str_list = _str.split('\n')[-2].split('  :::  ')

        if n_modules:
            modules_str = modules_from_trace(traceback.extract_stack(), n_modules - 1, start_module=2)
            _str_list[0] = '{} ({})'.format(_str_list[0], modules_str)
        text = '\n'.join(_str_list)
    else:
        if n_modules:
            modules_str = modules_from_trace(traceback.extract_stack(), n_modules, start_module=1)
            if text is None:
                text = modules_str
            else:
                text = '{}\n({})'.format(text, modules_str)
        else:
            if text is None:
                text = title

    if not isinstance(img_list, (list, tuple)):
        img_list = [img_list, ]

    size = fmt.size

    # print('self.size: {}'.format(self.size))

    color = col_rgb[fmt.color]
    font = CVConstants.fonts[fmt.font]
    line_type = CVConstants.line_types[fmt.line_type]

    location = list(fmt.offset)

    if '\n' in text:
        text_list = text.split('\n')
    else:
        text_list = [text, ]

    max_text_width = 0
    text_height = 0
    text_heights = []

    for _text in text_list:
        (_text_width, _text_height) = cv2.getTextSize(_text, font, fontScale=fmt.size, thickness=fmt.thickness)[0]
        if _text_width > max_text_width:
            max_text_width = _text_width
        text_height += _text_height + 5
        text_heights.append(_text_height)

    text_width = max_text_width + 10
    text_height += 30

    text_img = np.zeros((text_height, text_width), dtype=np.uint8)
    for _id, _text in enumerate(text_list):
        cv2.putText(text_img, _text, tuple(location), font, size, color, fmt.thickness, line_type)
        location[1] += text_heights[_id] + 5

    text_img = text_img.astype(np.float32) / 255.0

    text_img = np.stack([text_img, ] * 3, axis=2)

    for _id, _img in enumerate(img_list):
        if len(_img.shape) == 2:
            _img = np.stack([_img, ] * 3, axis=2)
        if _img.dtype == np.uint8:
            _img = _img.astype(np.float32) / 255.0
        img_list[_id] = _img

    img_stacked = stack_images_with_resize(img_list, grid_size=grid_size, preserve_order=1,
                                           only_border=no_resize)
    img_list_txt = [text_img, img_stacked]

    img_stacked_txt = stack_images_with_resize(img_list_txt, grid_size=(2, 1), preserve_order=1,
                                               only_border=no_resize)
    # img_stacked_txt_res = cv2.resize(img_stacked_txt, (300, 300), fx=0, fy=0)
    # img_stacked_txt_res_gs = cv2.cvtColor(img_stacked_txt_res, cv2.COLOR_BGR2GRAY)

    img_stacked_txt = (img_stacked_txt * 255).astype(np.uint8)

    if img_stacked_txt.shape[0] > max_height > 0:
        img_stacked_txt = resize_ar(img_stacked_txt, height=max_height)

    if img_stacked_txt.shape[1] > max_width > 0:
        img_stacked_txt = resize_ar(img_stacked_txt, width=max_width)

    if only_annotate:
        return img_stacked_txt

    if use_plt:
        img_stacked_txt = cv2.resize(img_stacked_txt, (300, 300), fx=0, fy=0)
        plt.imshow(img_stacked_txt)
        plt.pause(0.0001)
    else:
        _siif = SIIF.imshow(title, img_stacked_txt)
        if _siif:
            return pause

        if pause == 0:
            _pause_time = 100
        else:
            _pause_time = 0

        k = cv2.waitKey(_pause_time)
        if k == 27:
            cv2.destroyWindow(title)
            exit()
        if k == 32:
            pause = 1 - pause

    return pause


def stack_images(img_list, stack_order=0, grid_size=None):
    """

    :param img_list:
    :param int stack_order:
    :param list | None | tuple grid_size:
    :return:
    """
    if isinstance(img_list, (tuple, list)):
        n_images = len(img_list)
        img_shape = img_list[0].shape
        is_list = 1
    else:
        n_images = img_list.shape[0]
        img_shape = img_list.shape[1:]
        is_list = 0

    if grid_size is None:
        grid_size = [int(np.ceil(np.sqrt(n_images))), ] * 2
    else:
        if len(grid_size) == 1:
            grid_size = [grid_size[0], grid_size[0]]
        elif grid_size[0] == -1:
            grid_size = [int(math.ceil(n_images / grid_size[1])), grid_size[1]]
        elif grid_size[1] == -1:
            grid_size = [grid_size[0], int(math.ceil(n_images / grid_size[0]))]

    stacked_img = None
    list_ended = False
    inner_axis = 1 - stack_order
    for row_id in range(grid_size[0]):
        start_id = grid_size[1] * row_id
        curr_row = None
        for col_id in range(grid_size[1]):
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
                curr_row = np.concatenate((curr_row, curr_img), axis=inner_axis)
        if stacked_img is None:
            stacked_img = curr_row
        else:
            stacked_img = np.concatenate((stacked_img, curr_row), axis=stack_order)
        if list_ended:
            break
    return stacked_img



def stack_images_1D(img_list, stack_order=0):
    # stack into a single row or column
    stacked_img = None
    inner_axis = 1 - stack_order
    for img in img_list:
        if stacked_img is None:
            stacked_img = img
        else:
            stacked_img = np.concatenate((stacked_img, img), axis=inner_axis)
    return stacked_img


def remove_sub_folders(dir_name, sub_dir_prefix):
    folders = [linux_path(dir_name, name) for name in os.listdir(dir_name) if
               name.startswith(sub_dir_prefix) and
               os.path.isdir(linux_path(dir_name, name))]
    for folder in folders:
        shutil.rmtree(folder)


def write(str):
    sys.stdout.write(str)
    sys.stdout.flush()


def get_date_time():
    return time.strftime("%y%m%d_%H%M", time.localtime())


def parse_seq_IDs(ids):
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


def help_from_docs(obj, member):
    _help = ''
    doc = inspect.getdoc(obj)
    if doc is None:
        return _help

    doc_lines = doc.splitlines()
    if not doc_lines:
        return _help

    templ = ':param {} {}: '.format(type(getattr(obj, member)).__name__, member)
    relevant_line = [k for k in doc_lines if k.startswith(templ)]

    if relevant_line:
        _help = relevant_line[0].replace(templ, '')

    return _help


def str_to_tuple(val):
    if val.startswith('range('):
        val_list = val[6:].replace(')', '').split(',')
        val_list = [int(x) for x in val_list]
        val_list = tuple(range(*val_list))
        return val_list
    elif ',' not in val:
        val = '{},'.format(val)
    return literal_eval(val)


def add_params_to_parser(parser, obj, root_name='', obj_name=''):
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
        if isinstance(default_val, (int, bool, float, str, tuple, dict)):
            if root_name:
                member_param_name = '{:s}.{:s}'.format(root_name, member)
            else:
                member_param_name = '{:s}'.format(member)
            if member in obj.help:
                _help = obj.help[member]
            else:
                _help = help_from_docs(obj, member)

            if isinstance(default_val, tuple):
                parser.add_argument('--{:s}'.format(member_param_name), type=str_to_tuple,
                                    default=default_val, help=_help, metavar='')
            elif isinstance(default_val, dict):
                parser.add_argument('--{:s}'.format(member_param_name), type=json.loads, default=default_val,
                                    help=_help, metavar='')
            else:
                parser.add_argument('--{:s}'.format(member_param_name), type=type(default_val), default=default_val,
                                    help=_help, metavar='')
        else:
            # parameter is itself an instance of some other parameter class so its members must
            # be processed recursively
            add_params_to_parser(parser, getattr(obj, member), root_name, member)


def assign_arg(obj, arg, id, val):
    if id >= len(arg):
        print('Invalid arg: ', arg)
        return
    _arg = arg[id]
    obj_attr = getattr(obj, _arg)
    if isinstance(obj_attr, (int, bool, float, str, list, tuple, dict)):
        if val == '#' or val == '__n__':
            if isinstance(obj_attr, str):
                # empty string
                val = ''
            elif isinstance(obj_attr, tuple):
                # empty tuple
                val = ()
            elif isinstance(obj_attr, list):
                # empty list
                val = []
            elif isinstance(obj_attr, dict):
                # empty dict
                val = {}
        setattr(obj, _arg, val)
    else:
        # parameter is itself an instance of some other parameter class so its members must
        # be processed recursively
        assign_arg(obj_attr, arg, id + 1, val)


def process_args_from_parser(obj, args):
    # arg_prefix = ''
    # if hasattr(obj, 'arg_prefix'):
    #     arg_prefix = obj.arg_prefix
    members = vars(args)
    for key in members.keys():
        val = members[key]
        key_parts = key.split('.')
        assign_arg(obj, key_parts, 0, val)


def get_intersection_area(boxA, boxB):
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


def processArguments(args, params):
    # arguments specified as 'arg_name=argv_val'
    no_of_args = len(args)
    for arg_id in range(no_of_args):
        arg_str = args[arg_id]
        if arg_str.startswith('--'):
            arg_str = arg_str[2:]
        arg = arg_str.split('=')
        if len(arg) != 2 or not arg[0] in params.keys():
            raise IOError('Invalid argument provided: {:s}'.format(args[arg_id]))

        if not arg[1] or not arg[0] or arg[1] == '#':
            continue

        if isinstance(params[arg[0]], (list, tuple)):

            if ':' in arg[1]:
                inclusive_start = inclusive_end = 1
                if arg[1].endswith(')'):
                    arg[1] = arg[1][:-1]
                    inclusive_end = 0
                if arg[1].startswith(')'):
                    arg[1] = arg[1][1:]
                    inclusive_start = 0

                _temp = [float(k) for k in arg[1].split(':')]
                if len(_temp) == 3:
                    _step = _temp[2]
                else:
                    _step = 1.0
                if inclusive_end:
                    _temp[1] += _step
                if not inclusive_start:
                    _temp[0] += _step
                arg_vals_parsed = list(np.arange(*_temp))
            else:
                if arg[1] and ',' not in arg[1]:
                    arg[1] = '{},'.format(arg[1])

                arg_vals = [x for x in arg[1].split(',') if x]
                arg_vals_parsed = []
                for _val in arg_vals:
                    if _val == '__n__':
                        _val = ''
                    try:
                        _val_parsed = int(_val)
                    except ValueError:
                        try:
                            _val_parsed = float(_val)
                        except ValueError:
                            _val_parsed = _val
                    arg_vals_parsed.append(_val_parsed)

            params[arg[0]] = type(params[arg[0]])(arg_vals_parsed)
        else:
            params[arg[0]] = type(params[arg[0]])(arg[1])


def sort_key(fname):
    fname = os.path.splitext(os.path.basename(fname))[0]
    # print('fname: ', fname)
    # split_fname = fname.split('_')
    # print('split_fname: ', split_fname)

    # nums = [int(s) for s in fname.split('_') if s.isdigit()]
    # non_nums = [s for s in fname.split('_') if not s.isdigit()]

    split_list = fname.split('_')
    key = ''

    for s in split_list:
        if s.isdigit():
            if not key:
                key = '{:08d}'.format(int(s))
            else:
                key = '{}_{:08d}'.format(key, int(s))
        else:
            if not key:
                key = s
            else:
                key = '{}_{}'.format(key, s)

    # for non_num in non_nums:
    #     if not key:
    #         key = non_num
    #     else:
    #         key = '{}_{}'.format(key, non_num)
    # for num in nums:
    #     if not key:
    #         key = '{:08d}'.format(num)
    #     else:
    #         key = '{}_{:08d}'.format(key, num)

    # try:
    #     key = nums[-1]
    # except IndexError:
    #     return fname

    # print('fname: {}, key: {}'.format(fname, key))
    return key


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def nms(score_maxima_loc, dist_sqr_thresh):
    valid_score_maxima_x = []
    valid_score_maxima_y = []

    x, y = np.copy(score_maxima_loc[0]), np.copy(score_maxima_loc[1])

    while True:
        n_score_maxima_loc = len(x)
        if n_score_maxima_loc == 0:
            break

        curr_x, curr_y = x[0], y[0]
        valid_score_maxima_x.append(curr_x)
        valid_score_maxima_y.append(curr_y)

        # for j in range(1, n_score_maxima_loc):
        #     _x, _y = score_maxima_loc[0][j], score_maxima_loc[1][j]

        removed_idx = [0, ]

        removed_idx += [i for i in range(1, n_score_maxima_loc) if
                        (curr_x - x[i]) ** 2 + (curr_y - y[i]) ** 2 < dist_sqr_thresh]

        x, y = np.delete(x, removed_idx), np.delete(y, removed_idx)

    return [valid_score_maxima_x, valid_score_maxima_y]


#
# def df_test():
#     import pandas as pd
#     from policy import PolicyDecision
#
#     _stats_df = pd.DataFrame(
#         np.zeros((len(PolicyDecision.types), len(AnnotationStatus.types))),
#         columns=AnnotationStatus.types,
#         index=PolicyDecision.types,
#     )
#     _stats_df2 = pd.DataFrame(
#         np.zeros((len(PolicyDecision.types),)),
#         index=PolicyDecision.types,
#     )
#
#     _stats_df20 = _stats_df2[0]
#
#     _stats_df11 = _stats_df['fp_background']['unknown_neg']
#     _stats_df21 = _stats_df20['correct']
#
#     _stats_df20['correct'] = 67
#     _stats_df['fp_background']['unknown_neg'] = 92
#
#     _stats_df3 = pd.DataFrame(
#         np.full((len(PolicyDecision.types),), 3),
#         index=PolicyDecision.types,
#     )
#     _stats_df4 = pd.DataFrame(
#         np.full((len(PolicyDecision.types), len(AnnotationStatus.types)), 5),
#         columns=AnnotationStatus.types,
#         index=PolicyDecision.types,
#     )
#     _stats_df4['fp_background'] += _stats_df3[0]


def combined_motmetrics(acc_dict, logger):
    # logger.info(f'Computing overall MOT metrics over {len(acc_dict)} sequences...')
    # start_t = time.time()
    try:
        import evaluation.motmetrics as mm
    except ImportError as excp:
        logger.error('MOT evaluator is not available: {}'.format(excp))
        return False
    seq_names, accs = map(list, zip(*acc_dict.items()))

    # logger.info(f'Merging accumulators...')
    accs = mm.MOTAccumulator.merge_event_dataframes(accs)

    # logger.info(f'Computing metrics...')
    mh = mm.metrics.create()
    summary = mh.compute(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        name='OVERALL',
    )
    # end_t = time.time()
    # logger.info('Time taken: {:.3f}'.format(end_t - start_t))

    summary = summary.rename(columns=mm.io.motchallenge_metric_names)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters
    )
    print(strsummary)

    return summary, strsummary




def write_stats(stats, out_path, index_label, title):
    with open(out_path, 'a') as fid:
        fid.write(title + '\n')
    stats.to_csv(out_path, sep='\t', index_label=index_label, line_terminator='\n', mode='a')


def motmetrics_to_file(eval_paths, summary, load_fname, seq_name,
                       mode='a', time_stamp='', verbose=1, devkit=0):
    """

    :param eval_paths:
    :param summary:
    :param load_fname:
    :param seq_name:
    :param mode:
    :param time_stamp:
    :param verbose:
    :return:
    """

    if not time_stamp:
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    for eval_path in eval_paths:
        if verbose:
            print(f'{eval_path}')

        write_header = False
        if not os.path.isfile(eval_path):
            write_header = True

        with open(eval_path, mode) as eval_fid:
            if write_header:
                eval_fid.write('{:<50}'.format('timestamp'))
                eval_fid.write('\t{:<50}'.format('file'))
                for _metric, _type in zip(summary.columns.values, summary.dtypes):
                    if _type == np.int64:
                        eval_fid.write('\t{:>6}'.format(_metric))
                    else:
                        eval_fid.write('\t{:>8}'.format(_metric))
                if not devkit:
                    eval_fid.write('\t{:>10}'.format('MT(%)'))
                    eval_fid.write('\t{:>10}'.format('ML(%)'))
                    eval_fid.write('\t{:>10}'.format('PT(%)'))

                eval_fid.write('\n')
            eval_fid.write('{:13s}'.format(time_stamp))
            eval_fid.write('\t{:50s}'.format(load_fname))
            _values = summary.loc[seq_name].values
            # if seq_name == 'OVERALL':
            #     if verbose:
            #         print()

            for _val, _type in zip(_values, summary.dtypes):
                if _type == np.int64:
                    eval_fid.write('\t{:6d}'.format(int(_val)))
                else:
                    eval_fid.write('\t{:.6f}'.format(_val))
            if not devkit:
                try:
                    _gt = float(summary['GT'][seq_name])
                except KeyError:
                    pass
                else:
                    mt_percent = float(summary['MT'][seq_name]) / _gt * 100.0
                    ml_percent = float(summary['ML'][seq_name]) / _gt * 100.0
                    pt_percent = float(summary['PT'][seq_name]) / _gt * 100.0
                    eval_fid.write('\t{:3.6f}\t{:3.6f}\t{:3.6f}'.format(
                        mt_percent, ml_percent, pt_percent))

            eval_fid.write('\n')


def add_suffix(src_path, suffix):
    # abs_src_path = os.path.abspath(src_path)
    src_dir = os.path.dirname(src_path)
    src_name, src_ext = os.path.splitext(os.path.basename(src_path))
    dst_path = os.path.join(src_dir, src_name + '_' + suffix + src_ext)
    return dst_path


def most_recently_modified_dir(prev_results_dir, excluded=()):
    if isinstance(excluded, str):
        excluded = (excluded,)

    subdirs = [linux_path(prev_results_dir, k) for k in os.listdir(prev_results_dir)
               if os.path.isdir(linux_path(prev_results_dir, k)) and k not in excluded]

    subdirs_mtime = [os.path.getmtime(k) for k in subdirs]

    subdirs_sorted = sorted(zip(subdirs_mtime, subdirs))
    load_dir = subdirs_sorted[-1][1]

    return load_dir


def get_neighborhood(_score_map, cx, cy, r, _score_sz, type, thickness=1):
    if type == 0:

        # _max = -np.inf
        neighborhood = []

        x1, y1 = int(cx - r), int(cy - r)
        x2, y2 = int(cx + r), int(cy + r)

        # max_x1 = max_x2 = max_y1 = max_y2 = -np.inf
        incl_x1 = incl_y1 = incl_x2 = incl_y2 = 1

        if x1 < 0:
            x1 = 0
            incl_x1 = 0
        if y1 < 0:
            y1 = 0
            incl_y1 = 0
        if x2 >= _score_sz:
            x2 = _score_sz - 1
            incl_x2 = 0
        if y2 >= _score_sz:
            y2 = _score_sz - 1
            incl_y2 = 0

        if y2 >= y1:
            if incl_x1:
                # max_x1 = np.amax(_score_map[y1:y2 + 1, x1])
                # _max = max(_max, max_x1)
                x1 += 1
                neighborhood += list(_score_map[y1:y2 + 1, x1].flat)

            if incl_x2:
                # max_x2 = np.amax(_score_map[y1:y2 + 1, x2])
                # _max = max(_max, max_x2)
                x2 -= 1
                neighborhood += list(_score_map[y1:y2 + 1, x2].flat)

        if x2 >= x1:
            if incl_y1:
                # max_y1 = np.amax(_score_map[y1, x1:x2 + 1])
                # _max = max(_max, max_y1)
                neighborhood += list(_score_map[y1, x1:x2 + 1].flat)
            if incl_y2:
                # max_y2 = np.amax(_score_map[y2, x1:x2 + 1])
                # _max = max(_max, max_y2)
                neighborhood += list(_score_map[y2, x1:x2 + 1].flat)

        return np.asarray(neighborhood)
    elif type == 1:
        x = np.arange(0, _score_sz)
        y = np.arange(0, _score_sz)
        mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 >= r ** 2
        # _max = np.amax(_score_map[mask])

        neighborhood = _score_map[mask].flatten()
        return np.asarray(neighborhood)
        # return _max
    elif type == 2:
        mask = np.zeros_like(_score_map)
        cv2.circle(mask, (cx, cy), int(r), color=1, thickness=thickness)
        # _max = np.amax(_score_map[mask.astype(np.bool)])
        neighborhood = _score_map[mask.astype(np.bool)].flatten()

        return neighborhood
    else:
        raise AssertionError('Invalid neighborhood type: {}'.format(type))

    # if _max > 0:
    #     _conf = 1 - math.exp(-1.0 / _max)
    # else:
    #     _conf = 1

    # return _max


def clamp(list_x, minx, maxx):
    return [max(min(x, maxx), minx) for x in list_x]


def get_patch(img, bbox, to_gs, out_size):
    min_x, min_y, w, h = np.asarray(bbox).squeeze()
    max_x, max_y = min_x + w, min_y + h

    img_h, img_w = img.shape[:2]
    min_x, max_x = clamp([min_x, max_x], 0, img_w)
    min_y, max_y = clamp([min_y, max_y], 0, img_h)

    if max_x <= min_x or max_y <= min_y:
        assert out_size is not None, f"out_size must be provided to handle annoying invalid boxes like: {bbox}"

        patch = np.zeros(out_size, dtype=img.dtype)
        resized_patch = patch
    else:
        patch = img[int(min_y):int(max_y), int(min_x):int(max_x), ...]
        if to_gs:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        if out_size is not None:
            resized_patch = cv2.resize(patch, dsize=out_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized_patch = patch

    return patch, resized_patch


def spawn(dst, src):
    src_members = [a for a in dir(src) if not a.startswith('__') and not callable(getattr(src, a))]
    dst_members = [a for a in dir(dst) if not a.startswith('__') and not callable(getattr(dst, a))]

    members_to_spawn = list(filter(lambda a: a not in dst_members, src_members))
    # print(f'members_to_spawn:\n{pformat(members_to_spawn)}')
    for _member in members_to_spawn:
        setattr(dst, _member, getattr(src, _member))


def load_samples_from_file(db_path, load_prev_paths):
    if os.path.isdir(db_path):
        db_path = linux_path(db_path, 'model.bin.npz')

    db_dict = np.load(db_path)
    features = db_dict['features']  # type: np.ndarray
    labels = db_dict['labels']  # type: np.ndarray

    n_samples = features.shape[0]
    assert labels.shape[0] == n_samples, f"Mismatch between n_samples in " \
        f"labels: {labels.shape[0]} and features: {n_samples}"

    if not n_samples:
        print(f'no samples found in {db_path}')
    else:
        print(f'Loaded {n_samples} samples from  {db_path}')

    if not load_prev_paths:
        return features, labels

    try:
        prev_paths = list(db_dict['prev_db_paths'])
        prev_paths = [k for k in prev_paths if k != db_path]
    except KeyError:
        print(f'No prev_db_paths found')
        prev_paths = []

    return features, labels, prev_paths
