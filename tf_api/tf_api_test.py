import cv2
import numpy as np
import pandas as pd
import time
import sys, os
from datetime import datetime
from pprint import pprint
import imageio

# import threading
# import queue as queue

from threading import Thread as Process
from threading import Lock

# from multiprocessing import Process, Lock

try:
    sys.path.remove('/home/abhineet/labelling_tool/object_detection_module')
    sys.path.remove('/home/abhineet/labelling_tool/object_detection_module/object_detection')
except:
    pass

try:
    sys.path.remove('/home/abhineet/617_w18/Assignment2/models/research/object_detection')
except:
    pass
    # print('could not remove /home/abhineet/617_w18/Assignment2/models/research/object_detection')

try:
    sys.path.remove('/home/abhineet/617_w18/Assignment2/models/research')
except:
    pass
    # print('could not remove /home/abhineet/617_w18/Assignment2/models/research')
sys.path.append("..")

import tensorflow as tf

# This is needed since the notebook is stored in the object_detection folder.

# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util
from utils import visualization_utils as vis_util
from utilities import processArguments, sortKey, resizeAR

params = {
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    'ckpt_path': 'evaluation_frozen_graphs/F-RCNN_Inceptionv2/frozen_inference_graph.pb',
    # List of the strings that is used to add correct label for each box.
    'labels_path': 'data/wildlife_label_map.pbtxt',
    'file_name': '',
    'list_file_name': '',
    'model_list': '',
    'model_id': 0,
    'root_dir': '',
    'save_dir': '',
    'save_file_name': '',
    'csv_file_name': '',
    # 'map_folder': '',
    'load_path': '',
    'n_classes': 4,
    'input_type': '',
    'batch_size': 1,
    'show_img': 0,
    'save_video': 0,
    'vis_width': 0,
    'vis_height': 0,
    'n_frames': 0,
    'codec': 'H264',
    'fps': 20,
    'allow_memory_growth': 1,
    'gpu_memory_fraction': 1.0,
    'write_det': 1,
    'write_data': 0,
    'classes_to_include': [],
    'use_ptgrey': 0,
    'threaded_mode': 1,
    'rgb_mode': 1,
    'video_mode': 1,
    'fullscreen': 0,
    'use_mtf': 0,
    'mtf_args': '',
    'use_ffmpeg': 0,
    'ffmpeg_size': '1280x720',
    'ffmpeg_url': 'udp://127.0.0.1:8888',
}

processArguments(sys.argv[1:], params)
_ckpt_path = params['ckpt_path']
_labels_path = params['labels_path']
_n_classes = params['n_classes']
model_list = params['model_list']
model_id = params['model_id']
file_name = params['file_name']
list_file_name = params['list_file_name']
root_dir = params['root_dir']
save_dir = params['save_dir']
save_file_name = params['save_file_name']
csv_file_name = params['csv_file_name']
load_path = params['load_path']
input_type = params['input_type']
batch_size = params['batch_size']
show_img = params['show_img']
save_video = params['save_video']
_vis_width = params['vis_width']
_vis_height = params['vis_height']
n_frames = params['n_frames']
codec = params['codec']
fps = params['fps']
allow_memory_growth = params['allow_memory_growth']
gpu_memory_fraction = params['gpu_memory_fraction']
write_det = params['write_det']
write_data = params['write_data']
classes_to_include = params['classes_to_include']
use_ptgrey = params['use_ptgrey']
rgb_mode = params['rgb_mode']
video_mode = params['video_mode']
threaded_mode = params['threaded_mode']
fullscreen = params['fullscreen']
use_mtf = params['use_mtf']
mtf_args = params['mtf_args']
use_ffmpeg = params['use_ffmpeg']
_ffmpeg_size = params['ffmpeg_size']
ffmpeg_url = params['ffmpeg_url']

if mtf_args:
    mtf_args = mtf_args.replace(',', ' ')

if show_img and fullscreen:
    from screeninfo import get_monitors

    monitors = get_monitors()
    curr_monitor = str(monitors[0])
    resolution = curr_monitor.split('(')[1].split('+')[0].split('x')

    _vis_width, _vis_height = [int(x) for x in resolution]

if use_mtf:
    print('Using MTF pipeline')
    try:
        from mtf import mtf
    except ImportError as e:
        raise IOError('MTF import failed: {}'.format(e))
    use_ptgrey = 0

if use_ptgrey == 1:
    try:
        import PySpin
    except ImportError as e:
        print('PySpin import failed: {}'.format(e))
        raise IOError('Spinnsker based PtGrey cameras cannot be used without PySpin')
elif use_ptgrey == 2:
    try:
        import PyCapture2
    except ImportError as e:
        print('PySpin import failed: {}'.format(e))
        raise IOError('FlyCapture based PtGrey cameras cannot be used without PyCapture2')

if not show_img:
    write_data = 0

if model_list:
    if not os.path.exists(model_list):
        raise IOError('Checkpoint list file: {} does not exist'.format(model_list))
    model_list = [x.strip().split(',') for x in open(model_list).readlines() if x.strip()]
    model_list = [[ckpt_path, labels_path, int(n_classes), ckpt_label]
                      for ckpt_path, labels_path, n_classes, ckpt_label in model_list]
else:
    model_list = [[_ckpt_path, _labels_path, _n_classes, ''], ]
    model_id = 0

ckpt_path, labels_path, n_classes, ckpt_label = model_list[model_id]
if not os.path.exists(ckpt_path):
    raise IOError('Checkpoint file: {} does not exist'.format(ckpt_path))

if os.path.isdir(ckpt_path):
    if 'inference' in os.path.basename(ckpt_path) and\
            os.path.isfile(os.path.join(ckpt_path, 'frozen_inference_graph.pb')):
        ckpt_path = os.path.join(ckpt_path, 'frozen_inference_graph.pb')
    else:
        inference_dirs = [f for f in os.listdir(ckpt_path) if
                          os.path.isdir(os.path.join(ckpt_path, f)) and f.startswith('inference')]
        max_num_dir = None
        max_num = -1
        for inference_dir in inference_dirs:
            try:
                _num = int(inference_dir.split('_')[-1])
            except:
                _num = 0
            if _num > max_num:
                max_num = _num
                max_num_dir = inference_dir

        inference_dir = max_num_dir
        print('inference_dirs: ', inference_dirs)
        print('max_num_dir: ', max_num_dir)
        print('ckpt_path: ', ckpt_path)
        ckpt_path = os.path.join(ckpt_path, max_num_dir, 'frozen_inference_graph.pb')
else:
    inference_dir = os.path.basename(os.path.dirname(ckpt_path))

print('\nLoading inference graph from: {}'.format(ckpt_path))
if ckpt_label:
    print('\nCheckpoint label: {}\n'.format(ckpt_label))

label_map = label_map_util.load_labelmap(labels_path)

# if n_classes <= 0:
#     n_classes = len(label_map)s
# print('label_map: ', label_map)
# print('len(label_map): ', len(label_map))

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=n_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
category_index_dict = {k['id']: k['name'] for k in category_index.values()}

# print('categories: ', categories)
# print('category_index: ', category_index)
print('category_index_dict:')
pprint(category_index_dict)

if write_data == 1:
    data_out_fmt = 'jpg'
elif write_data > 1:
    data_out_fmt = 'gif'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ckpt_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

video_exts = ['mp4', 'mkv', 'avi', 'mpg', 'mpeg', 'mjpg']


# ## Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


using_camera = 0
out_postfix = ''
if list_file_name:
    if not os.path.exists(list_file_name):
        raise IOError('List file: {} does not exist'.format(list_file_name))
    file_list = [x.strip() for x in open(list_file_name).readlines() if x.strip()]
    if root_dir:
        file_list = [os.path.join(root_dir, x) for x in file_list]
    out_postfix = os.path.splitext(os.path.basename(list_file_name))[0]
elif root_dir:
    if root_dir.startswith('camera'):
        file_list = [root_dir]
        using_camera = 1
    else:
        file_list = []
        if input_type:
            if input_type == 'videos':
                for ext in video_exts:
                    file_list += [os.path.join(root_dir, k) for k in os.listdir(root_dir) if
                                  not os.path.isdir(os.path.join(root_dir, k)) and k.endswith('.{:s}'.format(ext))]
                if len(file_list) == 0:
                    file_gen = [[os.path.join(dirpath, f) for f in filenames if
                                 os.path.splitext(f.lower())[1][1:] in video_exts]
                                for (dirpath, dirnames, filenames) in os.walk(root_dir)]
                    file_list = [item for sublist in file_gen for item in sublist]
                    print('Here we are')
            else:
                file_list += [os.path.join(root_dir, k) for k in os.listdir(root_dir) if
                              not os.path.isdir(os.path.join(root_dir, k)) and k.endswith('.{:s}'.format(input_type))]
        else:
            file_list = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                         os.path.isdir(os.path.join(root_dir, name))]
        file_list.sort(key=sortKey)
else:
    if not file_name:
        raise IOError('Either list file or a single sequence file must be provided')
    file_list = [file_name]

print('file_list: ', file_list)

n_seq = len(file_list)
if using_camera:
    print('Running over live camera sequence')
else:
    print('Running over {} sequence(s)'.format(n_seq))


if not save_dir:
    save_dir = 'results'

if ckpt_label:
    out_dir = ckpt_label
    if out_postfix:
        out_dir = '{}_on_{}'.format(out_dir, out_postfix)
    save_dir = os.path.join(save_dir, out_dir)

avg_fps_list = np.zeros((n_seq,))
session_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
session_config.gpu_options.allow_growth = allow_memory_growth
session_config.gpu_options.per_process_gpu_memory_fraction=gpu_memory_fraction

n_gt = n_tp = n_fn = n_fp = 0
_recall = _precision = 0.0

data_mutex = Lock()

save_fn = save_tp = save_fp = None


def save_data(time_stamp, out_dir):
    out_fname = '{}.{}'.format(time_stamp, data_out_fmt)
    # out_fname = '{}.jpg'.format(time.strftime("%y%m%d%H%M%S", time.localtime()))
    out_path = os.path.join(out_dir, out_fname)
    with data_mutex:
        if write_data == 1:
            imageio.imsave(out_path, data_buffer[0])
        else:
            imageio.mimsave(out_path, data_buffer)


def registerTruePositive():
    global n_gt, n_tp, save_tp, _recall, _precision
    n_gt += 1
    n_tp += 1

    _recall = float(n_tp) / float(n_gt) if n_gt > 0 else 1.0
    _recall *= 100.0

    n_det = n_tp + n_fp
    _precision = float(n_tp) / float(n_det) if n_det > 0 else 1.0
    _precision *= 100.0

    time_stamp = datetime.now().strftime("%y%m%d%H%M%S%f")
    print(
        '{} :: True positive registered (gt: {:d} tp: {:d} fn: {:d} fp: {:d} recall: {:.2f} precision: {:.2f})'.format(
            time_stamp, n_gt, n_tp, n_fn, n_fp, _recall, _precision))
    if write_data:
        data_thread = Process(target=save_data, args=(time_stamp, save_tp))
        data_thread.start()


def registerFalseNegative():
    global n_gt, n_fn, save_fn, _recall, _precision
    n_gt += 1
    n_fn += 1

    _recall = float(n_tp) / float(n_gt) if n_gt > 0 else 1.0
    _recall *= 100.0

    n_det = n_tp + n_fp
    _precision = float(n_tp) / float(n_det) if n_det > 0 else 1.0
    _precision *= 100.0

    time_stamp = datetime.now().strftime("%y%m%d%H%M%S%f")
    print(
        '{} :: False negative registered (gt: {:d} tp: {:d} fn: {:d} fp: {:d} recall: {:.2f} precision: {:.2f})'.format(
            time_stamp, n_gt, n_tp, n_fn, n_fp, _recall, _precision))
    if write_data:
        data_thread = Process(target=save_data, args=(time_stamp, save_fn))
        data_thread.start()


def registerFalsePositive():
    global n_fp, save_fp, _recall, _precision
    n_fp += 1

    n_det = n_tp + n_fp
    _precision = float(n_tp) / float(n_det) if n_det > 0 else 1.0
    _precision *= 100.0

    time_stamp = datetime.now().strftime("%y%m%d%H%M%S%f")
    print(
        '{} :: False positive registered (gt: {:d} tp: {:d} fn: {:d} fp: {:d} recall: {:.2f} precision: {:.2f})'.format(
            time_stamp, n_gt, n_tp, n_fn, n_fp, _recall, _precision))
    if write_data:
        data_thread = Process(target=save_data, args=(time_stamp, save_fp))
        data_thread.start()


if show_img:
    def mouseHandler(event, x, y, flags=None, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            registerTruePositive()
        elif event == cv2.EVENT_LBUTTONUP:
            pass
        elif event == cv2.EVENT_RBUTTONDOWN:
            registerFalseNegative()
        elif event == cv2.EVENT_RBUTTONUP:
            pass
        elif event == cv2.EVENT_MBUTTONDOWN:
            registerFalsePositive()
        elif event == cv2.EVENT_MOUSEMOVE:
            pass


    win_title = 'Detections'

    if fullscreen:
        cv2.namedWindow(win_title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        # cv2.namedWindow(win_title, cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow(win_title)
    cv2.setMouseCallback(win_title, mouseHandler)

_pause = 0
image_converted = None
stop_pt_grey_cam = stop_ffmpeg_stream = 0
height = width = None
cam = None
ffmpeg_mutex = Lock()
ptgrey_mutex = Lock()
capture_residual = 0.0
cap_fps = None
enable_recording = 1

if use_ffmpeg:
    print('Using ffmpeg stream capture')
    import subprocess as sp

    command = ['ffmpeg',
               '-i', ffmpeg_url,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)

    ffmpeg_w, ffmpeg_h = [int(x) for x in _ffmpeg_size.split('x')]
    ffmpeg_size = ffmpeg_w * ffmpeg_h * 3


    def runFFMPEGCapture():
        global image_converted
        while True:
            if stop_ffmpeg_stream:
                break
            raw_image = pipe.stdout.read(ffmpeg_size)
            with ffmpeg_mutex:
                image_converted = np.fromstring(raw_image, dtype=np.uint8).reshape((ffmpeg_h, ffmpeg_w, 3))
            pipe.stdout.flush()


    ffmpeg_thread = Process(target=runFFMPEGCapture)
    ffmpeg_thread.start()

    wait_time = 0.0
    while image_converted is None:
        sys.stdout.write('\rWaiting for image acquisition to start ({:.2f} sec.)'.format(wait_time))
        sys.stdout.flush()
        time.sleep(0.5)
        wait_time += 0.5
    sys.stdout.write('\n')
    sys.stdout.flush()
elif use_ptgrey:
    def runPySpinCam(cam_id, _mode=0):
        global height, width, image_converted, cap_fps

        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        num_cameras = cam_list.GetSize()

        print("Number of cameras detected: {:d}".format(num_cameras))
        if num_cameras == 0:
            cam_list.Clear()
            system.ReleaseInstance()
            raise IOError("Not enough cameras!")

        cam = cam_list.GetByIndex(cam_id)
        try:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            try:
                node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode("DeviceInformation"))

                if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                    features = node_device_information.GetFeatures()
                    for feature in features:
                        node_feature = PySpin.CValuePtr(feature)
                        print("%s: %s" % (node_feature.GetName(),
                                          node_feature.ToString() if PySpin.IsReadable(node_feature) else
                                          "Node not readable"))

                else:
                    print("Device control information not available.")

            except PySpin.SpinnakerException as ex:
                raise IOError("Error in getting device info: %s" % ex)

            cam.Init()
            nodemap = cam.GetNodeMap()
            if rgb_mode == 1:
                pix_format_txt = "RGB8Packed"
            elif rgb_mode == 2:
                pix_format_txt = "BayerRG8"
            else:
                pix_format_txt = "Mono8"

            pixel_format_mode = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
            if not PySpin.IsAvailable(pixel_format_mode) or not PySpin.IsWritable(pixel_format_mode):
                raise IOError("Unable to set pixel format mode to RGB (enum retrieval). Aborting...")
            node_pixel_format_mode_rgb8 = pixel_format_mode.GetEntryByName(pix_format_txt)
            if not PySpin.IsAvailable(node_pixel_format_mode_rgb8) or not PySpin.IsReadable(
                    node_pixel_format_mode_rgb8):
                raise IOError("Unable to set pixel format mode to RGB (entry retrieval). Aborting...")
            pixel_format_mode.SetIntValue(node_pixel_format_mode_rgb8.GetValue())
            print("pixel format mode set to {:s}...".format(pix_format_txt))

            video_mode_txt = 'Mode{:d}'.format(video_mode)
            video_mode_node = PySpin.CEnumerationPtr(nodemap.GetNode("VideoMode"))
            if not PySpin.IsAvailable(video_mode_node) or not PySpin.IsWritable(video_mode_node):
                raise IOError("Unable to set video mode to {} (enum retrieval). Aborting...".format(video_mode_txt))
            node_video_mode_node = video_mode_node.GetEntryByName(video_mode_txt)
            if not PySpin.IsAvailable(node_video_mode_node) or not PySpin.IsReadable(node_video_mode_node):
                raise IOError("Unable to set video mode to {} (entry retrieval). Aborting...".format(video_mode_txt))
            video_mode_node.SetIntValue(node_video_mode_node.GetValue())
            print("video mode set to {:s}...".format(video_mode_txt))

            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                raise IOError(
                    "Unable to set acquisition mode to continuous (enum retrieval). Aborting...")

            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                raise IOError("Unable to set acquisition mode to continuous (entry retrieval). Aborting...")
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
            print("acquisition mode set to continuous...")
            cam.BeginAcquisition()

            # get first image
            while True:
                try:
                    # print('Getting the first image')
                    image_result = cam.GetNextImage()
                    if image_result.IsIncomplete():
                        print("Image incomplete with image status %d ..." % image_result.GetImageStatus())
                        continue
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()

                    image_converted = image_result

                    # if rgb_mode == 2:
                    #     image_converted = image_result.Convert(PySpin.PixelFormat_RGB8Packed, PySpin.HQ_LINEAR)
                    # else:
                    #     image_converted = image_result

                    # image_result.Release()
                    break
                except PySpin.SpinnakerException as ex:
                    raise IOError("Error in acquiring image: %s" % ex)

            while True:
                if stop_pt_grey_cam:
                    break
                try:
                    cap_start_t = time.time()
                    image_result = cam.GetNextImage()
                    if image_result.IsIncomplete():
                        print("Image incomplete with image status %d ..." % image_result.GetImageStatus())
                        continue
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    cap_end_t = time.time()
                    cap_fps = 1.0 / float(cap_end_t - cap_start_t)

                    with ptgrey_mutex:
                        # if rgb_mode == 2:
                        #     image_converted = image_result.Convert(PySpin.PixelFormat_RGB8Packed, PySpin.HQ_LINEAR)
                        # else:
                        #     image_converted = image_result
                        image_converted = image_result

                    # cap_end_t2 = time.time()
                    # cap_fps2 = 1.0 / float(cap_end_t2 - cap_start_t)

                    if _mode == 1:
                        image_np_gray = np.array(image_converted.GetData(), dtype=np.uint8).reshape(
                            (height, width)).copy()
                        image_np = cv2.cvtColor(image_np_gray, cv2.COLOR_GRAY2RGB)
                        cv2.imshow(win_title, image_np)
                        k = cv2.waitKey(1)
                        if k == ord('q') or k == 27:
                            break

                    # image_result.Release()
                except PySpin.SpinnakerException as ex:
                    raise IOError("Error in acquiring image: %s" % ex)
        except PySpin.SpinnakerException as ex:
            raise IOError("Error: %s" % ex)

        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()


    def printCameraInfo(cam):
        camInfo = cam.getCameraInfo()
        print("\n*** CAMERA INFORMATION ***\n")
        print("Serial number - ", camInfo.serialNumber)
        print("Camera model - ", camInfo.modelName)
        print("Camera vendor - ", camInfo.vendorName)
        print("Sensor - ", camInfo.sensorInfo)
        print("Resolution - ", camInfo.sensorResolution)
        print("Firmware version - ", camInfo.firmwareVersion)
        print("Firmware build time - ", camInfo.firmwareBuildTime)
        print()


    def printFormat7Capabilities(fmt7info):
        print("Max image pixels: ({}, {})".format(fmt7info.maxWidth, fmt7info.maxHeight))
        print("Image unit size: ({}, {})".format(fmt7info.imageHStepSize, fmt7info.imageVStepSize))
        print("Offset unit size: ({}, {})".format(fmt7info.offsetHStepSize, fmt7info.offsetVStepSize))
        print("Pixel format bitfield: 0x{}".format(fmt7info.pixelFormatBitField))
        print()


    def runPyCapture2Cam(cam_id, _mode=0):
        global height, width, image_converted, cap_fps

        bus = PyCapture2.BusManager()
        num_cameras = bus.getNumOfCameras()

        print("Number of cameras detected: {:d}".format(num_cameras))
        if num_cameras == 0:
            raise IOError("Not enough cameras!")

        cam = PyCapture2.Camera()
        uid = bus.getCameraFromIndex(cam_id)
        cam.connect(uid)

        printCameraInfo(cam)
        fmt7info, supported = cam.getFormat7Info(0)
        printFormat7Capabilities(fmt7info)

        if video_mode == -1:
            print('Setting pixel mode...')
            # Check whether pixel format mono8 is supported
            if PyCapture2.PIXEL_FORMAT.MONO8 & fmt7info.pixelFormatBitField == 0:
                raise IOError("Pixel format is not supported\n")

            # Configure camera format7 settings
            fmt7imgSet = PyCapture2.Format7ImageSettings(0, 0, 0, fmt7info.maxWidth, fmt7info.maxHeight,
                                                         PyCapture2.PIXEL_FORMAT.MONO8)
            fmt7pktInf, isValid = cam.validateFormat7Settings(fmt7imgSet)
            if not isValid:
                raise IOError("Format7 settings are not valid!")
            cam.setFormat7ConfigurationPacket(fmt7pktInf.recommendedBytesPerPacket, fmt7imgSet)
            print('done')

        cam.startCapture()

        while True:
            try:
                image_result = cam.retrieveBuffer()
            except PyCapture2.Fc2error as fc2Err:
                print("Error retrieving buffer : ", fc2Err)
                continue

            width = image_result.getCols()
            height = image_result.getRows()
            image_converted = image_result
            break

        while True:
            if stop_pt_grey_cam:
                break
            cap_start_t = time.time()
            try:
                image_result = cam.retrieveBuffer()
            except PyCapture2.Fc2error as fc2Err:
                print("Error retrieving buffer : ", fc2Err)
                continue

            width = image_result.getCols()
            height = image_result.getRows()

            cap_end_t = time.time()
            cap_fps = 1.0 / float(cap_end_t - cap_start_t)

            with ptgrey_mutex:
                image_converted = image_result

            if _mode == 1:
                image_np_gray = np.array(image.getData(), dtype=np.uint8).reshape((height, width)).copy()
                image_np = cv2.cvtColor(image_np_gray, cv2.COLOR_GRAY2RGB)
                cv2.imshow(win_title, image_np)
                k = cv2.waitKey(1)
                if k == ord('q') or k == 27:
                    break

        cam.stopCapture()
        cam.disconnect()


    if use_ptgrey == 1:
        runPtGreyCam = runPySpinCam
    elif use_ptgrey == 2:
        runPtGreyCam = runPyCapture2Cam

    if threaded_mode == 1:
        pt_grey_thread = Process(target=runPtGreyCam, args=(0, 0))
        pt_grey_thread.start()
        # pt_grey_thread.join()
        wait_time = 0.0
        while image_converted is None:
            sys.stdout.write('\rWaiting for image acquisition to start ({:.2f} sec.)'.format(wait_time))
            sys.stdout.flush()
            time.sleep(0.5)
            wait_time += 0.5
        sys.stdout.write('\n')
        sys.stdout.flush()
    elif threaded_mode == 2:
        runPtGreyCam(0, 1)
        sys.exit(0)
    else:
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        num_cameras = cam_list.GetSize()
        print("Number of cameras detected: {:d}".format(num_cameras))
        if num_cameras == 0:
            cam_list.Clear()
            system.ReleaseInstance()
            raise IOError("Not enough cameras!")
        cam = cam_list.GetByIndex(use_ptgrey - 1)
        try:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            try:
                node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode("DeviceInformation"))

                if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                    features = node_device_information.GetFeatures()
                    for feature in features:
                        node_feature = PySpin.CValuePtr(feature)
                        print("%s: %s" % (node_feature.GetName(),
                                          node_feature.ToString() if PySpin.IsReadable(node_feature) else
                                          "Node not readable"))
                else:
                    print("Device control information not available.")

            except PySpin.SpinnakerException as ex:
                raise IOError("Error in getting device info: %s" % ex)
            cam.Init()
            nodemap = cam.GetNodeMap()
            if rgb_mode:
                pix_format_txt = "RGB8Packed"
            else:
                pix_format_txt = "Mono8"

            pixel_format_mode = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
            if not PySpin.IsAvailable(pixel_format_mode) or not PySpin.IsWritable(pixel_format_mode):
                raise IOError(
                    "Unable to set pixel format mode to RGB (enum retrieval). Aborting...")
            node_pixel_format_mode_rgb8 = pixel_format_mode.GetEntryByName(pix_format_txt)
            if not PySpin.IsAvailable(node_pixel_format_mode_rgb8) or not PySpin.IsReadable(
                    node_pixel_format_mode_rgb8):
                raise IOError("Unable to set pixel format mode to RGB (entry retrieval). Aborting...")
            pixel_format_mode_rgb8 = node_pixel_format_mode_rgb8.GetValue()
            pixel_format_mode.SetIntValue(pixel_format_mode_rgb8)
            print("pixel format mode set to {:s}...".format(pix_format_txt))

            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                raise IOError(
                    "Unable to set acquisition mode to continuous (enum retrieval). Aborting...")
            # node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("SingleFrame")
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                raise IOError("Unable to set acquisition mode to single frame (entry retrieval). Aborting...")
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
            print("Acquisition mode set to continuous...")

            cam.BeginAcquisition()
            while True:
                image_result = cam.GetNextImage()
                pt_grey_start_time = time.time()
                if image_result.IsIncomplete():
                    print("Image incomplete with image status %d ..." % image_result.GetImageStatus())
                    continue
                width = image_result.GetWidth()
                height = image_result.GetHeight()
                break
        except PySpin.SpinnakerException as ex:
            raise IOError("Error: %s" % ex)
cap = None
data_buffer = []
if use_ffmpeg:
    width = ffmpeg_w
    height = ffmpeg_h

with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=session_config) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        for file_idx, file_name in enumerate(file_list):
            seq_name = os.path.splitext(os.path.basename(file_name))[0]

            if using_camera and seq_name.startswith('camera'):
                try:
                    cam_id = int(seq_name.split('_')[-1])
                except:
                    cam_id = 0
                if not use_ffmpeg and not use_ptgrey:
                    if use_mtf:
                        cap = mtf.VideoCapture(cam_id, mtf_args)
                    else:
                        cap = cv2.VideoCapture(cam_id)
                write_det = 0
            else:
                use_ptgrey = 0
                print('sequence {}/{}: {}: '.format(file_idx + 1, n_seq, seq_name))
                if os.path.isdir(file_name):
                    if use_mtf:
                        cap = mtf.VideoCapture(file_name, mtf_args)
                    else:
                        cap = cv2.VideoCapture(os.path.join(file_name, 'image%06d.jpg'))
                else:
                    if not os.path.exists(file_name):
                        raise IOError('Source video file: {} does not exist'.format(file_name))
                    seq_name = os.path.splitext(seq_name)[0]
                    if use_mtf:
                        cap = mtf.VideoCapture(file_name, mtf_args)
                    else:
                        cap = cv2.VideoCapture(file_name)
                    if not cap:
                        raise SystemError('Source video file: {} could not be opened'.format(file_name))

            if cap is not None:
                width = cap.get(3)
                height = cap.get(4)

            width = int(width)
            height = int(height)

            print('width: ', width)
            print('height: ', height)

            if _vis_height <= 0 or _vis_width <= 0:
                vis_height, vis_width = height, width
            else:
                vis_height, vis_width = _vis_height, _vis_width

            if not save_file_name:
                time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
                save_file_name = os.path.join(save_dir, '{}_{}.mkv'.format(seq_name, time_stamp))

            _save_dir = os.path.dirname(save_file_name)
            if not os.path.isdir(_save_dir):
                os.makedirs(_save_dir)

            if write_data:
                save_tp = os.path.join(_save_dir, seq_name, 'true_positives')
                save_fn = os.path.join(_save_dir, seq_name, 'false_negatives')
                save_fp = os.path.join(_save_dir, seq_name, 'false_positives')

                if not os.path.isdir(save_tp):
                    os.makedirs(save_tp)

                if not os.path.isdir(save_fn):
                    os.makedirs(save_fn)

                if not os.path.isdir(save_fp):
                    os.makedirs(save_fp)

                print('Writing true positives to: {}'.format(save_tp))
                print('Writing false negatives to: {}'.format(save_fn))
                print('Writing false positives to: {}'.format(save_fp))

            if write_det:
                if not csv_file_name:
                    csv_file_name = os.path.join(save_dir, '{}.csv'.format(seq_name))
                csv_save_dir = os.path.dirname(csv_file_name)
                if not os.path.isdir(csv_save_dir):
                    os.makedirs(csv_save_dir)

                print('Saving csv detections to {}'.format(csv_file_name))

                # if not map_folder:
                #     map_folder = os.path.join(save_dir, '{}_mAP'.format(seq_name))
                #
                # if not os.path.isdir(map_folder):
                #     os.makedirs(map_folder)
                # print('Saving mAP detections to {}'.format(map_folder))

            video_out = None
            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                # save_file_name = 'grizzly_bear_detection.avi'
                video_out = cv2.VideoWriter(save_file_name, fourcc, fps, (width, height))
                if not video_out:
                    raise SystemError('Output video file: {} could not be opened'.format(save_file_name))
                print('Saving visualizations to {}'.format(save_file_name))

            batch_id = 0
            avg_fps = 0
            frame_id = 0
            csv_raw = []
            exit_seq = 0
            while True:
                images = []
                for i in range(batch_size):
                    if use_ffmpeg:
                        with ffmpeg_mutex:
                            image_np = cv2.cvtColor(image_converted, cv2.COLOR_BGR2RGB)
                    elif use_ptgrey:
                        if threaded_mode == 0:
                            pt_grey_end_time = time.time()
                            capture_fps = 1.0 / float(pt_grey_end_time - pt_grey_start_time)
                            capture_ratio = (cap_fps / capture_fps) + capture_residual
                            capture_ratio_int = int(capture_ratio)
                            capture_residual = capture_ratio - capture_ratio_int
                            if capture_ratio_int < 1:
                                capture_ratio_int = 1
                            for i in range(capture_ratio_int):
                                image_result = cam.GetNextImage()
                                pt_grey_start_time = time.time()
                                if image_result.IsIncomplete():
                                    print("Image incomplete with image status %d ..." % image_result.GetImageStatus())
                                    continue
                                width = image_result.GetWidth()
                                height = image_result.GetHeight()
                            image_converted = image_result
                            # image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                            # image_converted = image_result.Convert(PySpin.PixelFormat_RGB8Packed, PySpin.HQ_LINEAR)
                        with ptgrey_mutex:
                            if use_ptgrey == 1:
                                image_data = image_converted.GetData()
                            else:
                                image_data = image_converted.getData()
                            # image_np = image_np_gray
                            raw_image = np.array(image_data, dtype=np.uint8)
                            # np.savetxt('raw_image.txt', raw_image, '%d')
                            if rgb_mode == 1:
                                image_np = cv2.cvtColor(raw_image.reshape((height, width, 3)), cv2.COLOR_BGR2RGB)
                            elif rgb_mode == 2:
                                image_np = cv2.cvtColor(raw_image.reshape((height, width)), cv2.COLOR_BayerRG2RGB)
                            else:
                                image_np = cv2.cvtColor(raw_image.reshape((height, width)), cv2.COLOR_GRAY2RGB)
                    else:
                        cap_start_t = time.time()
                        ret, image_np = cap.read()
                        if not ret:
                            break
                        cap_end_t = time.time()
                        cap_fps = 1.0 / float(cap_end_t - cap_start_t)
                        # cap_fps2 = cap_fps

                    images.append(image_np)

                # if video_ended:
                #     break

                curr_batch_size = len(images)

                if curr_batch_size == 0:
                    break

                # Actual detection
                _start_t = time.time()
                try:
                    images = np.asarray(images)
                    # print('images.shape', images.shape)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    # if curr_batch_size == 1:
                    #     images = np.expand_dims(images, axis=0)
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: images})
                    # print('boxes', boxes)
                    # print('scores', scores)
                    # print('classes', classes)
                except ValueError:
                    # images of different sizes
                    boxes = []
                    scores = []
                    classes = []
                    for image in images:
                        _image = np.expand_dims(image, axis=0)
                        (_boxes, _scores, _classes, num) = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: _image})
                        boxes.append(_boxes)
                        scores.append(_scores)
                        classes.append(_classes)

                _end_t = time.time()

                fps = float(curr_batch_size) / float(_end_t - _start_t)

                batch_id += 1
                avg_fps += (fps - avg_fps) / float(batch_id)

                # if curr_batch_size == batch_size or batch_size == 1:
                #     avg_fps += (fps - avg_fps) / float(batch_id)

                # print('num', num)
                # print('boxes', boxes)
                # print('scores', scores)
                # print('classes', classes)
                #
                # print('boxes.shape', boxes.shape)
                # print('scores.shape', scores.shape)
                # print('classes.shape', classes.shape)

                # Visualization of the detection results

                for i in range(curr_batch_size):
                    image_np = np.squeeze(images[i])

                    frame_id += 1

                    curr_boxes = np.squeeze(boxes[i])
                    curr_classes = np.squeeze(classes[i]).astype(np.int32)
                    curr_scores = np.squeeze(scores[i])

                    # print('curr_boxes: ', curr_boxes)
                    # print('curr_classes: ', curr_classes)
                    # print('curr_scores: ', curr_scores)

                    n_objs = len(list(curr_classes))

                    if classes_to_include:
                        _indices = [x for x in range(n_objs) if
                                    category_index_dict[curr_classes[x]] in classes_to_include]
                        # curr_classes = np.asarray([curr_classes[x] for x in _indices],dtype=np.int32)
                        # curr_boxes = np.asarray([curr_boxes[x] for x in _indices])
                        curr_scores = np.asarray([curr_scores[x] if x in _indices else 0 for x in range(n_objs)])
                        # for x in range(n_objs):
                        #     if not x in _indices:
                        #         curr_scores[x] = 0

                        # print('\n__curr_boxes: ', curr_boxes)
                        # print('__curr_classes: ', curr_classes)
                        # print('__curr_scores: ', curr_scores)

                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        curr_boxes,
                        curr_classes,
                        curr_scores,
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=4)

                    image_np = resizeAR(image_np, vis_width, vis_height)

                    if save_video and enable_recording:
                        video_out.write(image_np)

                    if write_det:
                        filename = 'image{:06d}.jpg'.format(frame_id)

                        # map_out_fname = os.path.join(map_folder, 'image{:06d}.txt'.format(frame_id))
                        # map_file = open(map_out_fname, 'w')

                        for _box, _class, _score in zip(curr_boxes, curr_classes, curr_scores):
                            if _score < 0.5:
                                continue

                            ymin, xmin, ymax, xmax = _box

                            xmin = xmin * width
                            xmax = xmax * width
                            ymin = ymin * height
                            ymax = ymax * height

                            label = category_index[_class]['name']

                            raw_data = {
                                'filename': filename,
                                'width': width,
                                'height': height,
                                'class': label,
                                'xmin': int(xmin),
                                'ymin': int(ymin),
                                'xmax': int(xmax),
                                'ymax': int(ymax),
                                'confidence': _score
                            }
                            csv_raw.append(raw_data)

                            # map_file.write('{:s} {:f} {:d} {:d} {:d} {:d}\n'.format(label, _score, int(xmin), int(ymin),
                            #                                                         int(xmax), int(ymax)))

                    if show_img:
                        img_text = "{:s} frame {:d} fps: {:5.2f} avg_fps: {:5.2f}".format(
                            seq_name, frame_id, fps, avg_fps)
                        if cap_fps is not None:
                            img_text = '{:s} cap_fps: {:5.2f}'.format(img_text, cap_fps)

                        cv2.putText(image_np, img_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

                        cv2.imshow(win_title, image_np)
                        k = cv2.waitKey(1 - _pause)

                        if write_data and data_mutex.acquire(False):
                            data_buffer.append(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                            if len(data_buffer) > write_data:
                                del data_buffer[0]
                            data_mutex.release()

                        if k == ord('q') or k == 27:
                            exit_seq = 1
                            break
                        elif k == 32:
                            enable_recording = 1 - enable_recording
                            if save_video:
                                if enable_recording:
                                    print('Video recording enabled')
                                else:
                                    print('Video recording disabled')
                        elif k == ord('1'):
                            registerTruePositive()
                        elif k == ord('2'):
                            registerFalseNegative()
                        elif k == ord('3'):
                            registerFalsePositive()

                    # if write_det:
                    #     map_file.close()

                if not show_img:
                    sys.stdout.write('\rDone {:d} frames fps: {:.4f} avg_fps: {:.4f} '.format(frame_id, fps, avg_fps))
                    sys.stdout.flush()

                if exit_seq:
                    break

                if n_frames > 0 and frame_id >= n_frames:
                    break

            sys.stdout.write('\n')
            sys.stdout.flush()

            if use_ffmpeg:
                stop_ffmpeg_stream = 1
            elif use_ptgrey:
                stop_pt_grey_cam = 1
            else:
                cap.release()
            if save_video:
                video_out.release()

            if write_det:
                df = pd.DataFrame(csv_raw)
                out_file_path = os.path.join(csv_file_name)
                df.to_csv(out_file_path)
            print('avg_fps: ', avg_fps)
            avg_fps_list[file_idx] = avg_fps

            save_file_name = ''
            csv_file_name = ''
            map_folder = ''

        if show_img:
            cv2.destroyWindow(win_title)

if not threaded_mode and cam is not None:
    cam.EndAcquisition()
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()

if show_img:
    print('n_gt: {}'.format(n_gt))
    print('n_tp: {}'.format(n_tp))
    print('n_fn: {}'.format(n_fn))
    print('n_fp: {}'.format(n_fp))

    _recall = float(n_tp) / float(n_gt) if n_gt > 0 else 1.0
    _recall *= 100.0

    n_det = n_tp + n_fp
    _precision = float(n_tp) / float(n_det) if n_det > 0 else 1.0
    _precision *= 100.0

    print('recall: {:.3f}%'.format(_recall))
    print('precision: {:.3f}%'.format(_precision))

overall_avg_fps = np.mean(avg_fps_list)
print('overall_avg_fps: ', overall_avg_fps)
