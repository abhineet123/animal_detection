import numpy as np
import os
import cv2
import glob
import re


def get_images_in_directory(dir_name, exts=None):
    if exts is None:
        exts = ['png', 'jpg', 'jpeg', 'bmp']
    files = []
    for ext in exts:
        files = files + glob.glob(os.path.join(dir_name, '*.{}'.format(ext)))

    def getint(fn):
        basename = os.path.basename(fn)
        num = re.sub("\D", "", basename)
        try:
            return int(num)
        except:
            return 0

    if len(files) > 0:
        files = sorted(files, key=getint)
    return files


class FramesReader:
    def __init__(self):
        self.num_frames = 0
        self.last_frame_read = -1
        self.height = 0
        self.width = 0

    def get_int_index(self, float_index):
        if float_index == -1:
            index = self.num_frames - 1
        elif float_index < 1:
            index = int(self.num_frames * float_index)
        else:
            index = np.max([0, int(float_index)])
        return index

    def get_last_frame_read(self):
        if self.last_frame_read > self.num_frames:
            return -1
        else:
            return float(self.last_frame_read) / self.num_frames

    def get_frames(self, start=0, end=-1):
        raise NotImplementedError

    def get_background_samples(self, num_background_samples=400):
        raise NotImplementedError


class VideoReader(FramesReader):
    def __init__(self, video_fn, **kwargs):
        FramesReader.__init__(self)
        self.video_fn = video_fn
        self.vidcap = cv2.VideoCapture(self.video_fn)
        self.num_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.num_frames, 'Cant read video. Please check path!'
        assert self.width, 'Cant read video. Please check path!'
        assert self.height, 'Cant read video. Please check path!'

    def get_frames(self, start=0, end=-1):
        start = self.get_int_index(start)
        end = self.get_int_index(end)
        self.vidcap.set(1, start)
        frame_id = start
        while True:
            if frame_id == end:
                break
            success, image = self.vidcap.read()
            if success:
                image = image[:, :, ::-1]
                self.last_frame_read = frame_id
                yield image
            else:
                break
            frame_id += 1

    def get_background_samples(self, num_background_samples=100):
        self.background_sample_interval = int(self.num_frames / num_background_samples)
        background_samples = []
        background_samples_frame_id = []
        vidcap = cv2.VideoCapture(self.video_fn)
        for j in np.arange(0, self.num_frames, self.background_sample_interval):
            vidcap.set(1, j)
            _, image = vidcap.read()
            image = image[:, :, ::-1]
            background_samples.append(image)
            background_samples_frame_id.append(j)
        return background_samples, background_samples_frame_id

    def get_frame(self, index):
        self.vidcap.set(1, index)
        success, image = self.vidcap.read()
        if not success:
            raise Exception
        else:
            return image[:, :, ::-1].copy()

    def get_frames_list(self):
        return [str(frame_no) for frame_no in range(self.num_frames)]

    def get_frame_by_name(self, name):
        try:
            return self.get_frame(int(name))
        except:
            return

    def get_path(self):
        return self.video_fn


class DirectoryReader(FramesReader):
    def __init__(self, directory_fn, exts=None, **kwargs):
        FramesReader.__init__(self)
        self.directory_fn = directory_fn
        self.exts = exts
        self.files = get_images_in_directory(directory_fn, exts)
        self.num_frames = len(self.files)
        self.height, self.width, _ = cv2.imread(self.files[0]).shape
        assert self.num_frames, 'Cant read any frames. Please check path!'
        assert self.width, 'Cant read any frames. Please check path!'
        assert self.height, 'Cant read any frames. Please check path!'

    def get_frames(self, start=0, end=-1):
        start = self.get_int_index(start)
        end = self.get_int_index(end)
        for j, file in enumerate(self.files[start:end]):
            image = cv2.imread(file)[:, :, ::-1]
            self.last_frame_read = j + start
            yield image

    def get_background_samples(self, num_background_samples=100):
        self.background_sample_interval = int(self.num_frames / num_background_samples)
        background_samples = []
        background_samples_frame_id = []
        for j in np.arange(0, self.num_frames, self.background_sample_interval):
            image = cv2.imread(self.files[j])[:, :, ::-1]
            background_samples.append(image)
            background_samples_frame_id.append(j)
        return background_samples, background_samples_frame_id

    def get_frame(self, index):
        image = cv2.imread(self.files[index])[:, :, ::-1]
        return image

    def get_frames_list(self):
        return [os.path.basename(file) for file in self.files]

    def get_frame_by_name(self, fn):
        image = cv2.imread(os.path.join(self.directory_fn, fn))[:, :, ::-1].copy()
        return image

    def get_path(self):
        return self.directory_fn


def get_frames_reader(path, **kwargs):
    if os.path.isdir(path):
        return DirectoryReader(path, **kwargs)
    else:
        return VideoReader(path, **kwargs)
