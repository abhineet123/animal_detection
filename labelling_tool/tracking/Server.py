import numpy as np
import argparse
import os
import socket, paramiko
import time
import cv2
import threading
import sys
import multiprocessing
import pandas as pd

import logging


def profile(self, message, *args, **kws):
    if self.isEnabledFor(PROFILE_LEVEL_NUM):
        self._log(PROFILE_LEVEL_NUM, message, args, **kws)


from PatchTracker import PatchTracker, PatchTrackerParams
from Visualizer import Visualizer, VisualizerParams
from Utilities import processArguments, addParamsToParser, processArgsFromParser, \
    str2list, list2str, drawRegion
from utils.netio import send_msg_to_connection, recv_from_connection

from libs.frames_readers import get_frames_reader
from libs.netio import bindToPort

sys.path.append('../..')
from tf_api.utilities import sortKey


# from utils.frames_readers import get_frames_reader


# from functools import partial

class ServerParams:
    """
    :type mode: int
    :type load_path: str
    :type continue_training: int | bool
    :type gate: GateParams
    :type patch_tracker: PatchTrackerParams
    :type visualizer: VisualizerParams
    """

    def __init__(self):
        self.cfg = 'cfg/params.cfg'
        self.mode = 0
        self.wait_timeout = 3
        self.port = 3002
        self.verbose = 0
        self.save_as_bin = 0

        self.remote_path = '/home/abhineet/acamp_code_non_root/labelling_tool/tracking'
        self.remote_cfg = 'params.cfg'
        self.remote_img_root_path = '/home/abhineet/acamp/object_detection/videos'
        self.hostname = ''
        self.username = ''
        self.password = ''

        self.img_path = ''
        self.img_paths = ''
        self.root_dir = ''
        self.save_dir = 'log'
        self.save_csv = 0
        self.track_init_frame = 1

        self.roi = ''
        self.id_number = 0
        self.init_frame_id = 0
        self.end_frame_id = -1
        self.init_bbox = ''

        self.patch_tracker = PatchTrackerParams()
        self.visualizer = VisualizerParams()
        self.help = {
            'cfg': 'optional ASCII text file from where parameter values can be read;'
                   'command line parameter values will override the values in this file',
            'mode': 'mode in which to run the server:'
                    ' 0: local execution'
                    ' 1: remote execution'
                    ' 2: output to terminal / GUI in local execution mode (non-server)',
            'port': 'port on which the server listens for requests',
            'save_as_bin': 'save images as binary files for faster reloading (may take a lot of disk space)',
            'img_path': 'single sequence on which patch tracker is to be run (mode=2); overriden by img_path',
            'img_paths': 'list of sequences on which patch tracker is to be run (mode=2); overrides img_path',
            'root_dir': 'optional root directory containing sequences on which patch tracker is to be run (mode=2)',

            'verbose': 'show detailed diagnostic messages',
            'patch_tracker': 'parameters for the patch tracker module',
            'visualizer': 'parameters for the visualizer module',
        }


class Server:
    """
    :type params: ServerParams
    :type logger: logging.RootLogger
    """

    def __init__(self, params, _logger):
        """
        :type params: ServerParams
        :type _logger: logging.RootLogger
        :rtype: None
        """

        self.params = params
        self.logger = _logger

        self.request_dict = {}
        self.request_list = []

        self.current_path = None
        self.frames_reader = None
        self.trainer = None
        self.tester = None
        self.visualizer = None
        self.enable_visualization = False
        self.traj_data = []

        self.trained_target = None
        self.tracking_res = None
        self.index_to_name_map = None

        self.max_frame_id = -1
        self.frame_id = -1

        self.pid = os.getpid()

        self.request_lock = threading.Lock()

        # create parsers for real time parameter manipulation
        self.parser = argparse.ArgumentParser()
        addParamsToParser(self.parser, self.params)

        self.client = None
        self.channel = None
        self._stdout = None
        self.remote_output = None

        if self.params.mode == 0:
            self.logger.info('Running in local execution mode')
        elif self.params.mode == 1:
            self.logger.info('Running in remote execution mode')
            self.connectToExecutionServer()
        elif self.params.mode == 2:
            self.logger.info('Running patch tracker directly')

        # self.patch_tracking_results = []

    def connectToExecutionServer(self):
        self.logger.info('Executing on {}@{}'.format(self.params.username, self.params.hostname))

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self.client.connect(
            self.params.hostname,
            username=self.params.username,
            password=self.params.password
        )
        self.channel = self.client.invoke_shell(width=1000, height=3000)
        self._stdout = self.channel.makefile()
        # self.flushChannel()

    def parseParams(self, parser, cmd_args):
        args_in = []
        # check for a custom cfg file specified at command line
        # prefix = '{:s}.'.format(name)
        if len(cmd_args) > 0 and '--cfg' in cmd_args[0]:
            _, arg_val = cmd_args[0].split('=')
            self.params.cfg = arg_val
            # print('Reading {:s} parameters from {:s}'.format(name, cfg))
        if os.path.isfile(self.params.cfg):
            file_args = open(self.params.cfg, 'r').readlines()
            # lines starting with # in the cfg file are regarded as comments and thus ignored
            file_args = ['--{:s}'.format(arg.strip()) for arg in file_args if arg.strip() and not arg.startswith('#')]
            # print('file_args', file_args)
            args_in += file_args
        # command line arguments override those in the cfg file
        args_in += ['--{:s}'.format(arg[2:]) for arg in cmd_args]
        # args_in = [arg[len(prefix):] for arg in args_in if prefix in arg]
        # print('args_in', args_in)
        args = parser.parse_args(args_in)
        processArgsFromParser(self.params, args)

    def getRemoteOutput(self):
        self.remote_output = self._stdout.readline().replace("^C", "")

    def flushChannel(self):
        # while not self.channel.exit_status_ready():
        while True:
            # if not self.channel.recv_ready():
            #     continue

            # remote_output = self._stdout.readline().replace("^C", "")

            self.remote_output = None

            p = multiprocessing.Process(target=self.getRemoteOutput)
            p.start()
            # Wait for 1 second or until process finishes
            p.join(self.params.wait_timeout)

            if p.is_alive():
                p.terminate()
                p.join()

            if not self.remote_output:
                break

            # print('remote_output: ', remote_output)
            if not self.remote_output.startswith('###'):
                sys.stdout.write(self.remote_output)
                sys.stdout.flush()

    def visualize(self, request):
        request_path = request["path"]
        csv_path = request["csv_path"]
        class_dict = request["class_dict"]
        request_roi = request["roi"]
        init_frame_id = request["frame_number"]

        save_fname_templ = os.path.splitext(os.path.basename(request_path))[0]

        df = pd.read_csv(csv_path)

        if request_path != self.current_path:
            self.frames_reader = get_frames_reader(request_path, save_as_bin=self.params.save_as_bin)
            if request_roi is not None:
                self.frames_reader.setROI(request_roi)
            self.current_path = request_path
        class_labels = dict((v, k) for k, v in class_dict.items())

        # print('self.params.visualizer.save: ', self.params.visualizer.save)
        visualizer = Visualizer(self.params.visualizer, self.logger, class_labels)
        init_frame = self.frames_reader.get_frame(init_frame_id)

        height, width, _ = init_frame.shape
        frame_size = width, height
        visualizer.initialize(save_fname_templ, frame_size)

        n_frames = self.frames_reader.num_frames
        for frame_id in range(init_frame_id, n_frames):
            try:
                curr_frame = self.frames_reader.get_frame(frame_id)
            except IOError as e:
                print('{}'.format(e))
                break

            file_path = self.frames_reader.get_file_path()
            if file_path is None:
                print('Visualization is only supported on image sequence data')
                return

            filename = os.path.basename(file_path)

            multiple_instance = df.loc[df['filename'] == filename]
            # Total # of object instances in a file
            no_instances = len(multiple_instance.index)
            # Remove from df (avoids duplication)
            df = df.drop(multiple_instance.index[:no_instances])

            frame_data = []

            for instance in range(0, len(multiple_instance.index)):
                target_id = multiple_instance.iloc[instance].loc['target_id']
                xmin = multiple_instance.iloc[instance].loc['xmin']
                ymin = multiple_instance.iloc[instance].loc['ymin']
                xmax = multiple_instance.iloc[instance].loc['xmax']
                ymax = multiple_instance.iloc[instance].loc['ymax']
                class_name = multiple_instance.iloc[instance].loc['class']
                class_id = class_dict[class_name]

                width = xmax - xmin
                height = ymax - ymin

                frame_data.append([frame_id, target_id, xmin, ymin, width, height, class_id])

            frame_data = np.asarray(frame_data)
            if not visualizer.update(frame_id, curr_frame, frame_data):
                break

        visualizer.close()

    def patchTracking(self, request=None, img_path=''):
        if self.params.mode == 2:
            sys.stdout.write('@@@ Starting tracker\n')
            sys.stdout.flush()
            cmd_args = sys.argv[1:]
        else:
            if request is not None:
                cmd_args = request['cmd_args']
            else:
                cmd_args = ''

        self.parseParams(self.parser, cmd_args)

        if request is not None:
            request_path = request["path"]
            request_roi = request["roi"]
            id_number = request['id_number']
            init_frame_id = request["frame_number"]
            init_bbox = request["bbox"]
            init_bbox_list = [
                int(init_bbox['xmin']),
                int(init_bbox['ymin']),
                int(init_bbox['xmax']),
                int(init_bbox['ymax']),
            ]
            label = request['label']
            request_port = request["port"]
        else:
            request_path = img_path if img_path else self.params.img_path
            request_roi = str2list(self.params.roi)
            id_number = self.params.id_number
            init_frame_id = self.params.init_frame_id

            if self.params.init_bbox:
                init_bbox_list = str2list(self.params.init_bbox)
                init_bbox = {
                    'xmin': init_bbox_list[0],
                    'ymin': init_bbox_list[1],
                    'xmax': init_bbox_list[2],
                    'ymax': init_bbox_list[3],
                }
            else:
                init_bbox = {}
            label = request_port = None

        gt_available = 0
        if request_path != self.current_path:
            self.frames_reader = get_frames_reader(request_path, save_as_bin=self.params.save_as_bin)
            if request_roi is not None:
                self.frames_reader.setROI(request_roi)
            self.current_path = request_path
            if not init_bbox:
                csv_path = os.path.join(request_path, 'annotations.csv')
                print('Reading annotations from {}'.format(csv_path))
                import pandas as pd
                df_gt = pd.read_csv(csv_path)
                _ = self.frames_reader.get_frame(init_frame_id)

                file_path = self.frames_reader.get_file_path()
                filename = os.path.basename(file_path)
                multiple_instance = df_gt.loc[df_gt['filename'] == filename]
                bbox = multiple_instance.iloc[0]
                xmin = bbox.loc['xmin']
                ymin = bbox.loc['ymin']
                xmax = bbox.loc['xmax']
                ymax = bbox.loc['ymax']
                init_bbox = {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                }
                gt_available = 1

        show_only = (self.params.mode == 1)
        tracker = PatchTracker(self.params.patch_tracker, self.logger, id_number, label, show_only=show_only)
        if not tracker.is_created:
            return

        init_frame = self.frames_reader.get_frame(init_frame_id)
        tracker.initialize(init_frame, init_bbox)
        if not tracker.is_initialized:
            self.logger.error('Tracker initialization was unsuccessful')
            return

        n_frames = self.frames_reader.num_frames

        if self.params.end_frame_id >= init_frame_id:
            end_frame_id = self.params.end_frame_id
        else:
            end_frame_id = n_frames - 1

        if self.params.mode == 1:

            if self.client is None:
                self.connectToExecutionServer()

            # print('init_bbox_list: ', init_bbox_list)
            remote_bbox = list2str(init_bbox_list)
            remote_img_path = os.path.join(self.params.remote_img_root_path, os.path.basename(request_path))
            cd_command = 'cd {:s}'.format(self.params.remote_path)
            exec_command = 'python3 Server.py --mode=2 ' \
                           '--cfg={:s} --img_path={:s} --id_number={:d} ' \
                           '--init_frame_id={:d} --init_bbox={:s}' \
                           ' --patch_tracker.tracker_type={:d}' \
                           ' --patch_tracker.cv_tracker_type={:d}' \
                           ' --patch_tracker.show=0' \
                           '' \
                           '\n'.format(
                self.params.remote_cfg,
                remote_img_path,
                id_number,
                init_frame_id,
                remote_bbox,
                self.params.patch_tracker.tracker_type,
                self.params.patch_tracker.cv_tracker_type,
            )
            # command = '{} && {}'.format(cd_command, exec_command)
            # if request_roi is not None:
            #     remote_roi = list2str(request_roi)
            #     command = '{:s} {:s}'.format(command, remote_roi)
            #
            print('Running:\n{:s}'.format(exec_command))
            curr_corners = np.zeros((2, 4), dtype=np.float64)

            # channel = self.client.invoke_shell(width=1000, height=3000)
            # _stdout = channel.makefile()

            self.channel.send(cd_command + '\n')

            # channel.send("sudo -s\n'''\n")
            # channel.send("'''" + '\n')

            self.channel.send(exec_command + '\n')
            # channel.send('exit' + '\n')

            # s = channel.recv(4096)
            # print(s)
            # client.close()
            # sys.exit()

            # _stdin, _stdout, _stderr = client.exec_command(command)
            # channel = _stdout.channel

            # print(_stdout.readlines())
            # while not _stderr.channel.recv_exit_status():
            #     if _stderr.channel.recv_ready():
            #         print(_stderr.read())

            pid = None
            tracking_started = 0
            wait_start_time = tracking_start_time = time.clock()
            while not self.channel.exit_status_ready():
                if not self.channel.recv_ready():
                    wait_time = time.clock() - wait_start_time
                    # sys.stdout.write('Waiting for stdout for {:f} secs\n'.format(wait_time))
                    # sys.stdout.flush()
                    if wait_time > self.params.wait_timeout:
                        print('Waiting time threshold exceeded')
                        break
                    continue

                wait_start_time = time.clock()
                # sys.stdout.write('\n')
                # sys.stdout.flush()
                # remote_output = channel.recv(45).decode("utf-8")
                remote_output = self._stdout.readline().replace("^C", "")
                # print('remote_output: ', remote_output)

                if remote_output.startswith('@@@'):
                    tracking_started = 1
                    continue

                if not remote_output.startswith('###'):
                    sys.stdout.write(remote_output)
                    sys.stdout.flush()
                    continue

                if not tracking_started:
                    continue

                # sys.stdout.write(remote_output)
                # sys.stdout.flush()

                result_list = remote_output.strip().split()
                # print('remote_output: ', remote_output)
                # print('result_list: ', result_list)

                if len(result_list) != 8:
                    print('remote_output: ', remote_output)
                    print('result_list: ', result_list)
                    raise SystemError('Invalid output from the remote server')

                pid = int(result_list[1])
                frame_id = int(result_list[2])
                xmin = int(result_list[3])
                ymin = int(result_list[4])
                xmax = int(result_list[5])
                ymax = int(result_list[6])
                remote_fps = float(result_list[7])

                curr_corners[:, 0] = (xmin, ymin)
                curr_corners[:, 1] = (xmax, ymin)
                curr_corners[:, 2] = (xmax, ymax)
                curr_corners[:, 3] = (xmin, ymax)

                try:
                    curr_frame = self.frames_reader.get_frame(frame_id)
                except IOError as e:
                    print('{}'.format(e))
                    break

                end_time = time.clock()
                fps = 1.0 / (end_time - tracking_start_time)

                tracker.show(curr_frame, curr_corners, frame_id, fps, remote_fps)

                out_bbox = dict(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                )
                self.send(curr_frame, out_bbox, label, request_path, frame_id, id_number, request_port)

                tracking_start_time = end_time
                if frame_id > end_frame_id or tracker.is_terminated:
                    # exit_client = paramiko.SSHClient()
                    # exit_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
                    # exit_client.connect(self.params.hostname, username=self.params.username,
                    #                     password=self.params.password)
                    # _stdin, _stdout, _stderr = exit_client.exec_command()
                    # exit_client.close()
                    # self.channel.send(chr(3))
                    # self.channel.send(chr(3))
                    # self.channel.send(chr(3))

                    # self.channel.send('exit' + '\n')

                    # channel_closed = 1

                    # channel.close()
                    # self.channel.send('pkill -P {:d}\n'.format(pid))
                    break

            # while not channel.recv_ready():
            #     sys.stdout.write('\rwaiting')
            #     sys.stdout.flush()
            #     continue

            tracker.close()

            self.channel.send(chr(3))
            if pid is not None:
                self.channel.send('pkill -P {:d}\n'.format(pid))

            # self.flushChannel()

            # p = multiprocessing.Process(target=self.flushChannel)
            # p.start()
            # # Wait for 1 second or until process finishes
            # p.join(self.params.flush_timeout)
            # if p.is_alive():
            #     p.terminate()
            #     p.join()

            print('\ndone execution')

            # _stdout.close()
            # _stderr.close()

            # channel.send('exit' + '\n')
            # self.channel.close()
            # client.close()
            return

        # init_frame = self.frames_reader.get_frame(init_frame_id)
        # init_file_path = self.frames_reader.get_file_path()

        # try:
        #     self.runTracker(tracker, init_frame_id)
        # except KeyboardInterrupt:
        #     pass

        # self.logger.info('Tracking target {:d} in sequence with {:d} frames '
        #                  'starting from frame {:d}'.format(
        #     id_numbers[0], n_frames, init_frame_id + 1))

        save_path = ''
        if self.params.save_dir:
            file_path = self.frames_reader.get_file_path()
            save_path = os.path.join('log', self.params.save_dir, os.path.basename(os.path.dirname(file_path)))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            save_csv_path = os.path.join(save_path, 'annotations.csv')
            print('Saving results csv to {}'.format(save_csv_path))

        if self.params.track_init_frame:
            start_frame_id = init_frame_id
        else:
            start_frame_id = init_frame_id + 1

        csv_raw = []
        if label is None:
            label = 'generic'
        for frame_id in range(start_frame_id, end_frame_id + 1):
            try:
                curr_frame = self.frames_reader.get_frame(frame_id)
            except IOError as e:
                print('{}'.format(e))
                break

            file_path = self.frames_reader.get_file_path()
            filename = os.path.basename(file_path)

            gt_bbox = None
            if gt_available:
                multiple_instance = df_gt.loc[df_gt['filename'] == filename]
                bbox = multiple_instance.iloc[0]
                xmin = bbox.loc['xmin']
                ymin = bbox.loc['ymin']
                xmax = bbox.loc['xmax']
                ymax = bbox.loc['ymax']
                label = bbox.loc['class']
                gt_bbox = {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                }
            try:
                fps = tracker.update(curr_frame, frame_id, gt_bbox=gt_bbox)
            except KeyboardInterrupt:
                break

            if tracker.out_bbox is None:
                # self.logger.error('Tracker update was unsuccessful')
                break

            if save_path:

                if tracker.curr_mask_cropped is not None:
                    mask_filename = os.path.splitext(filename)[0] + '.png'
                    save_mask_path = os.path.join(save_path, mask_filename)
                    curr_mask_norm = (tracker.curr_mask_cropped * 255.0).astype(np.uint8)
                    cv2.imwrite(save_mask_path, curr_mask_norm)

                    mask_filename_bin = os.path.splitext(filename)[0] + '.npy'
                    save_mask_path_bin = os.path.join(save_path, mask_filename_bin)
                    np.save(save_mask_path_bin, tracker.curr_mask_cropped)


                if self.params.save_csv:
                    orig_height, orig_width = curr_frame.shape[:2]
                    xmin = tracker.out_bbox['xmin']
                    xmax = tracker.out_bbox['xmax']
                    ymin = tracker.out_bbox['ymin']
                    ymax = tracker.out_bbox['ymax']

                    raw_data = {
                        'filename': filename,
                        'width': orig_width,
                        'height': orig_height,
                        'class': label,
                        'xmin': int(xmin),
                        'ymin': int(ymin),
                        'xmax': int(xmax),
                        'ymax': int(ymax),
                        'confidence': tracker.score
                    }
                    csv_raw.append(raw_data)

            if self.params.mode == 2 and not self.params.patch_tracker.show:
                sys.stdout.write('### {:d} {:d} {:d} {:d} {:d} {:d} {:5.2f}\n'.format(
                    self.pid, frame_id, tracker.out_bbox['xmin'], tracker.out_bbox['ymin'],
                    tracker.out_bbox['xmax'], tracker.out_bbox['ymax'], fps
                ))
                sys.stdout.flush()
                continue

            if request_port is not None:
                mask = os.path.abspath(save_mask_path_bin)
                # if tracker.curr_mask_cropped is not None:
                #     # mask = np.expand_dims(tracker.curr_mask, axis=0).tolist()
                #     mask = tracker.curr_mask_cropped.tolist()
                # else:
                #     mask = None

                self.send(curr_frame, tracker.out_bbox, label, request_path, frame_id,
                          id_number, request_port, masks=mask)
            # self.single_object_tracking_results.append(tracking_result)

            if tracker.is_terminated:
                break

        sys.stdout.write('Closing tracker...\n')
        sys.stdout.flush()

        tracker.close()

        if save_path and self.params.save_csv:
            df = pd.DataFrame(csv_raw)
            df.to_csv(save_csv_path)

    def send(self, curr_frame, out_bbox, label, request_path, frame_id, id_number,
             request_port, masks=None):

        # print('frame_id: {}, out_bbox: {}'.format(frame_id, out_bbox))

        if len(curr_frame.shape) == 3:
            height, width, channels = curr_frame.shape
        else:
            height, width = curr_frame.shape
            channels = 1

        tracking_result = dict(
            action="add_bboxes",
            path=request_path,
            frame_number=frame_id,
            width=width,
            height=height,
            channel=channels,
            bboxes=[out_bbox],
            scores=[0],
            labels=[label],
            id_numbers=[id_number],
            bbox_source="single_object_tracker",
            last_frame_number=frame_id - 1,
            trigger_tracking_request=False,
            num_frames=1,
            # port=request_port,
        )
        if masks is not None:
            tracking_result['masks'] = [masks, ]

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', request_port))
        send_msg_to_connection(tracking_result, sock)
        sock.close()

    def run(self):
        if self.params.mode == 2:
            img_paths = self.params.img_paths
            root_dir = self.params.root_dir

            if img_paths:
                if os.path.isfile(img_paths):
                    img_paths = [x.strip() for x in open(img_paths).readlines() if x.strip()]
                else:
                    img_paths = img_paths.split(',')
                if root_dir:
                    img_paths = [os.path.join(root_dir, name) for name in img_paths]

            elif root_dir:
                img_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                             os.path.isdir(os.path.join(root_dir, name))]
                img_paths.sort(key=sortKey)

            else:
                img_paths = (self.params.img_path,)

            print('Running patch tracker on {} sequences'.format(len(img_paths)))
            for img_path in img_paths:
                self.patchTracking(img_path=img_path)
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bindToPort(sock, self.params.port, 'tracking')
        sock.listen(1)
        self.logger.info('Tracking server started')
        # if self.params.mode == 0:
        #     self.logger.info('Started tracking server in local execution mode')
        # else:
        #     self.logger.info('Started tracking server in remote execution mode')
        while True:
            try:
                connection, addr = sock.accept()
                connection.settimeout(None)
                msg = recv_from_connection(connection)
                connection.close()
                if isinstance(msg, list):
                    raw_requests = msg
                else:
                    raw_requests = [msg]
                for request in raw_requests:
                    # print('request: ', request)
                    request_type = request['request_type']
                    if request_type == 'patch_tracking':
                        # self.params.processArguments()
                        try:
                            self.patchTracking(request)
                        except KeyboardInterrupt:
                            continue
                    # elif request_type == 'stop':
                    #     break
                    elif request_type == 'visualize':
                        self.visualize(request)
                    else:
                        self.logger.error('Invalid request type: {}'.format(request_type))
            except KeyboardInterrupt:
                print('Exiting due to KeyboardInterrupt')
                if self.client is not None:
                    self.client.close()
                return
            except SystemExit:
                if self.client is not None:
                    self.client.close()
                return
        # self.logger.info('Stopped tracking server')

    # def run(self):
    # threading.Thread(target=self.request_loop).start()


if __name__ == '__main__':
    # get parameters
    _params = ServerParams()
    processArguments(_params, description='Tracking Server')

    # setup logger
    PROFILE_LEVEL_NUM = 9
    logging.addLevelName(PROFILE_LEVEL_NUM, "PROFILE")
    logging.Logger.profile = profile

    logging_fmt = '%(levelname)s::%(module)s::%(funcName)s::%(lineno)s :  %(message)s'
    logging_level = logging.INFO
    # logging_level = logging.DEBUG
    # logging_level = PROFILE_LEVEL_NUM
    logging.basicConfig(level=logging_level, format=logging_fmt)
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)

    server = Server(_params, _logger)
    server.run()
