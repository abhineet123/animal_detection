import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import tensorflow as tf

print('Using Tensorflow ' + tf.__version__)
# import matplotlib.pyplot as plt

# import csv
import numpy as np
# from PIL import Image
import cv2

# try:

from .src import siamese as siam
from .src.parse_arguments import parse_arguments
from .src.region_to_bbox import region_to_bbox


# except ImportError:
#     import siamfc.src.siamese as siam
#     from siamfc.src.parse_arguments import parse_arguments
#     from siamfc.src.region_to_bbox import region_to_bbox


# from src.visualization import show_frame, show_crops, show_scores


# gpu_device = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)


class DesignParams:
    def __init__(self):
        self.join_method = "xcorr"
        self.net = "baseline-conv5_e55.mat"

        self.net_gray = ''
        self.windowing = 'cosine_sum'

        self.exemplar_sz = 127
        self.search_sz = 255
        self.score_sz = 33
        self.tot_stride = 4

        self.context = 0.5
        self.pad_with_image_mean = 1

        self.help = {
        }


class EnvironmentParams:
    def __init__(self):
        # self.root_dataset = "data"
        self.root_pretrained = "siamfc/pretrained"

        self.root_parameters = 'parameters'
        self.help = {
        }


class HyperParams:
    def __init__(self):
        self.response_up = 8
        self.window_influence = 0.25
        self.z_lr = 0.01
        self.scale_num = 3

        self.scale_step = 1.04
        self.scale_penalty = 0.97
        self.scale_lr = 0.59
        self.scale_min = 0.2
        self.scale_max = 5
        self.help = {
        }


class SiamFCParams:
    def __init__(self):
        self.update_location = 0

        self.allow_gpu_memory_growth = 1
        self.per_process_gpu_memory_fraction = 1.0

        self.gpu = -1
        self.visualize = 0

        self.design = DesignParams()
        self.env = EnvironmentParams()
        self.hp = HyperParams()

        self.help = {
            'design': 'DesignParams',
            'env': 'EnvironmentParams',
            'hp': 'HyperParams',
        }


class SiamFC:
    """
    :param SiamFCParams params:
    """

    def __init__(self, params, target_id=0,
                 label='generic', confidence=1.0):
        """
        :param SiamFCParams params:
        """

        self.params = params

        self.target_id = target_id
        self.label = label
        self.confidence = confidence
        self.cumulative_confidence = confidence

        # self.tf_graph = tf.Graph()
        # avoid printing TF debugging information
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # TODO: allow parameters from command line or leave everything in json files?
        hp, _, _, env, design = parse_arguments()

        design = self.params.design
        env = self.params.env
        hp = self.params.hp

        # Set size for use with tf.image.resize_images with align_corners=True.
        # For example,
        #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
        # instead of
        # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
        final_score_sz = hp.response_up * (design.score_sz - 1) + 1

        self.tf_graph = tf.Graph()
        self.tf_sess = tf.Session(graph=self.tf_graph)

        # self.tf_sess = tf.Session()

        # with self.tf_graph.as_default():

        # build TF graph once for all
        with self.tf_graph.as_default():
            image, templates_z, scores, pos, sizes = siam.build_tracking_graph2(final_score_sz, design, env)

        pos_x_ph, pos_y_ph = pos
        z_sz_ph, x_sz0_ph, x_sz1_ph, x_sz2_ph = sizes

        self.hp = hp
        # self.run = run
        self.design = design
        self.final_score_sz = final_score_sz

        self.image = image
        self.templates_z = templates_z
        self.scores = scores

        self.pos_x_ph = pos_x_ph
        self.pos_y_ph = pos_y_ph
        self.z_sz_ph = z_sz_ph
        self.x_sz0_ph = x_sz0_ph
        self.x_sz1_ph = x_sz1_ph
        self.x_sz2_ph = x_sz2_ph

        # with self.tf_graph.as_default():
        #     tf.variables_initializer(tf.get_collection(tf.GraphKeys.VARIABLES)).run(session=self.tf_sess)

        # self.start_frame = start_frame

        self.coord = None
        self.threads = None

        self.scale_factors = self.hp.scale_step ** np.linspace(-np.ceil(self.hp.scale_num / 2),
                                                               np.ceil(self.hp.scale_num / 2),
                                                               self.hp.scale_num)
        # cosine window to penalize large displacements
        self.hann_1d = np.expand_dims(np.hanning(self.final_score_sz), axis=0)
        self.penalty = np.transpose(self.hann_1d) * self.hann_1d
        self.penalty = self.penalty / np.sum(self.penalty)

        # run_metadata = tf.RunMetadata()
        # run_opts = {
        #     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        #     'run_metadata': run_metadata,
        # }

        self.run_opts = {}

    def initialize(self, init_frame, init_bbox, init_file_path=None):
        # if init_file_path == None:
        #     init_file_path = 'image_0.jpg'
        #     cv2.imwrite(init_file_path, init_frame)

        init_frame = init_frame.astype(np.float32)

        pos_x, pos_y, target_w, target_h = init_bbox

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.target_w = target_w
        self.target_h = target_h

        context = self.design.context * (target_w + target_h)
        self.z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
        self.x_sz = float(self.design.search_sz) / self.design.exemplar_sz * self.z_sz

        # thresholds to saturate patches shrinking/growing
        min_z = self.hp.scale_min * self.z_sz
        max_z = self.hp.scale_max * self.z_sz
        min_x = self.hp.scale_min * self.x_sz
        max_x = self.hp.scale_max * self.x_sz

        # bbox = np.zeros((1, 4))

        # with self.tf_graph.as_default():
        # with self.tf_sess as sess:
        with self.tf_graph.as_default():
            tf.global_variables_initializer().run(session=self.tf_sess)

        # Coordinate the loading of image files.
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.tf_sess)

        # save first frame position (from ground-truth)
        # bbox[0, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h

        templates_z_ = self.tf_sess.run([self.templates_z], feed_dict={
            self.pos_x_ph: pos_x,
            self.pos_y_ph: pos_y,
            self.z_sz_ph: self.z_sz,
            self.image: init_frame})

        self.templates_z_ = templates_z_
        self.new_templates_z_ = templates_z_

    def update(self, frame, frame_id, file_path=None):
        # if file_path == None:
        #     file_path = 'image_{}.jpg'.format(frame_id)
        #     cv2.imwrite(file_path, frame)

        frame = frame.astype(np.float32)

        # with self.tf_sess as sess:
        scaled_exemplar = self.z_sz * self.scale_factors
        scaled_search_area = self.x_sz * self.scale_factors
        scaled_target_w = self.target_w * self.scale_factors
        scaled_target_h = self.target_h * self.scale_factors
        scores_ = self.tf_sess.run(self.scores,
                                   feed_dict={
                                       self.pos_x_ph: self.pos_x,
                                       self.pos_y_ph: self.pos_y,
                                       self.x_sz0_ph: scaled_search_area[0],
                                       self.x_sz1_ph: scaled_search_area[1],
                                       self.x_sz2_ph: scaled_search_area[2],
                                       self.templates_z: np.squeeze(self.templates_z_),
                                       self.image: frame,
                                   }, **self.run_opts)
        scores_ = np.squeeze(scores_)
        # penalize change of scale
        scores_[0, :, :] = self.hp.scale_penalty * scores_[0, :, :]
        scores_[2, :, :] = self.hp.scale_penalty * scores_[2, :, :]
        # find scale with highest peak (after penalty)
        new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))
        # update scaled sizes
        self.x_sz = (1 - self.hp.scale_lr) * self.x_sz + self.hp.scale_lr * scaled_search_area[new_scale_id]
        self.target_w = (1 - self.hp.scale_lr) * self.target_w + self.hp.scale_lr * scaled_target_w[new_scale_id]
        self.target_h = (1 - self.hp.scale_lr) * self.target_h + self.hp.scale_lr * scaled_target_h[new_scale_id]
        # select response with new_scale_id
        score_ = scores_[new_scale_id, :, :]
        score_ = score_ - np.min(score_)
        score_ = score_ / np.sum(score_)
        # apply displacement penalty
        score_ = (1 - self.hp.window_influence) * score_ + self.hp.window_influence * self.penalty
        self.pos_x, self.pos_y, max_score = _update_target_position(self.pos_x, self.pos_y, score_, self.final_score_sz,
                                                         self.design.tot_stride,
                                                         self.design.search_sz, self.hp.response_up, self.x_sz)

        # update the target representation with a rolling average
        if self.hp.z_lr > 0:
            self.new_templates_z_ = self.tf_sess.run([self.templates_z], feed_dict={
                self.pos_x_ph: self.pos_x,
                self.pos_y_ph: self.pos_y,
                self.z_sz_ph: self.z_sz,
                self.image: frame
            })

            self.templates_z_ = (1 - self.hp.z_lr) * np.asarray(self.templates_z_) + self.hp.z_lr * np.asarray(
                self.new_templates_z_)

        # update template patch size
        self.z_sz = (1 - self.hp.scale_lr) * self.z_sz + self.hp.scale_lr * scaled_exemplar[new_scale_id]

        # convert <cx,cy,w,h> to <x,y,w,h> and save output
        bbox = [self.pos_x - self.target_w / 2, self.pos_y - self.target_h / 2, self.target_w, self.target_h]

        # if self.run.visualization:
        #     show_frame(image_, bbox[0, :], 1)

        self.confidence = max_score
        self.cumulative_confidence *= max_score
        return bbox

    def close(self):
        # tf.reset_default_graph()
        self.tf_sess.close()


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    max_score_id = np.argmax(score)
    max_score = score[np.unravel_index(max_score_id, score.shape)]
    p = np.asarray(np.unravel_index(max_score_id, np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop * x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y, max_score
