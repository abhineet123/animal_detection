import numpy as np
import os
import cv2
import re
import glob
import six
import shutil

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
        self.roi_height = 0
        self.roi_width = 0
        self.roi = None
        self.roi_bin_dir = None
        self.bin_dir = None

    def setROI(self, roi=None):
        if roi is not None:
            xmin = roi['xmin']
            ymin = roi['ymin']
            xmax = roi['xmax']
            ymax = roi['ymax']
            roi_width = xmax - xmin
            roi_height = ymax - ymin
            if roi_width <= 0 or roi_width > self.width or\
                            roi_height <= 0 or roi_height > self.height:
                print('Invalid ROI provided: ', roi)
                return False
            self.roi = roi
            self.roi_width = roi_width
            self.roi_height = roi_height
            self.roi_bin_dir = "{:s}_{:d}_{:d}_{:d}_{:d}".format(
                self.bin_dir, xmin, ymin, xmax, ymax )
            if not os.path.exists(self.roi_bin_dir):
                os.makedirs(self.roi_bin_dir)
        else:
            self.roi = None
            self.roi_bin_dir = self.bin_dir
            self.roi_height, self.roi_width = self.height, self.width
        # print('self.roi_bin_dir: ', self.roi_bin_dir)
        # print('self.bin_dir: ', self.bin_dir)
        return True

    def get_int_index(self, float_index):
        if float_index == -1:
            return self.num_frames - 1
        elif float_index < 1:
            return int(self.num_frames * float_index)
        else:
            return np.max([0, int(float_index)])

    def get_frame(self, index):
        raise NotImplementedError

    def get_frame_by_name(self, name):
        raise NotImplementedError

    def get_frames_list(self):
        raise NotImplementedError

    def get_path(self):
        raise NotImplementedError

    def get_file_path(self):
        raise NotImplementedError

    def get_background_samples(self, num_background_samples=100):
        raise NotImplementedError

    def close(self, delete_bin=True):
        if delete_bin and os.path.isdir(self.roi_bin_dir):
            # print('Deleting binary files...')
            shutil.rmtree(self.roi_bin_dir)


class VideoReader(FramesReader):
    def __init__(self, video_fn, save_as_bin=False, **kwargs):
        if six.PY3:
            super(VideoReader, self).__init__()
        else:
            FramesReader.__init__(self)

        if not video_fn:
            raise Exception("Must specify a path!")
        self.video_fn = os.path.abspath(video_fn)
        self.save_as_bin = save_as_bin
        self.vidcap = cv2.VideoCapture(self.video_fn)
        if cv2.__version__.startswith('2'):
            cv_prop = cv2.cv.CAP_PROP_FRAME_COUNT
        else:
            cv_prop = cv2.CAP_PROP_FRAME_COUNT

        self.num_frames = int(self.vidcap.get(cv_prop))
        directory_fn = os.path.dirname(self.video_fn)
        self.bin_dir = os.path.join(directory_fn, os.path.splitext(
            os.path.basename(self.video_fn))[0], "bin")
        self.roi_bin_dir = self.bin_dir
        if not os.path.exists(self.bin_dir):
            os.makedirs(self.bin_dir)

        if self.num_frames > 0:
            # self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            success, img = self.vidcap.read()
            if not success:
                raise Exception
            if len(img.shape) == 3:
                self.height, self.width, self.channels = img.shape
            else:
                self.height, self.width = img.shape
                self.channels = 1
            self.channels = 3
        else:
            self.height, self.width, self.channels = 0, 0, 0

    def get_frame(self, index, convert_to_rgb=1):
        if self.save_as_bin:
            bin_path = os.path.join(self.roi_bin_dir, "image{:06d}.bin".format(index + 1))
            # print('self.roi_bin_dir: ', self.roi_bin_dir)
            # print('bin_path: ', bin_path)
            if os.path.isfile(bin_path):
                try:
                    image = np.fromfile(open(bin_path, 'rb'), dtype=np.uint8)
                    if self.channels == 1:
                        image = image.reshape((self.roi_height, self.roi_width))
                    else:
                        image = image.reshape((self.roi_height, self.roi_width, self.channels))
                    return image
                except:
                    # print('Failed to read bin file')
                    pass

        if index != self.last_frame_read + 1:
            self.vidcap.set(1, index)
        success, image = self.vidcap.read()
        if not success:
            raise IOError('Frame {:d} could not be read'.format(index + 1))
        else:
            self.last_frame_read = index
            if self.roi is not None:
                xmin = self.roi['xmin']
                ymin = self.roi['ymin']
                xmax = self.roi['xmax']
                ymax = self.roi['ymax']
                if len(image.shape) == 3:
                    image = image[ymin:ymax, xmin:xmax, ::-1].copy()
                else:
                    image = image[ymin:ymax, xmin:xmax].copy()

                if self.save_as_bin:
                    image.astype(np.uint8).tofile(open(bin_path, 'wb'))
                return image
            if convert_to_rgb:
                image = image[:, :, ::-1].copy()
            if self.save_as_bin:
                try:
                    image.astype(np.uint8).tofile(open(bin_path, 'wb'))
                except IOError:
                    print('Bin file could not be written to disk')
                    # pass
            return image

    def get_frames_list(self):
        return [str(frame_no) for frame_no in range(self.num_frames)]

    def get_frame_by_name(self, name):
        try:
            return self.get_frame(int(name))
        except:
            return

    def get_path(self):
        return self.video_fn

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

    def get_file_path(self):
        return None


class DirectoryReader(FramesReader):
    def __init__(self, directory_fn, exts=None, save_as_bin=False, **kwargs):
        if six.PY3:
            super(DirectoryReader, self).__init__()
        else:
            FramesReader.__init__(self)

        if not directory_fn:
            raise Exception("Must specify a path!")
        self.directory_fn = os.path.abspath(directory_fn)
        self.save_as_bin = save_as_bin
        self.exts = exts
        self.files = get_images_in_directory(directory_fn, exts)
        self.num_frames = len(self.files)
        self.bin_dir = os.path.join(directory_fn, 'bin')
        self.roi_bin_dir = self.bin_dir
        if not os.path.exists(self.bin_dir):
            os.makedirs(self.bin_dir)

        if self.num_frames > 0:
            img = cv2.imread(self.files[0])
            if img is None:
                raise IOError('{} :: image could not be read: {}'.format(
                    directory_fn, os.path.basename(self.files[0])))
            if len(img.shape) == 3:
                self.height, self.width, self.channels = img.shape
            else:
                self.height, self.width = img.shape
                self.channels = 1
        else:
            self.height, self.width, self.channels = 0, 0, 0
        self.n_pix = self.height*self.width*self.channels
        self.roi_height, self.roi_width = self.height, self.width


    def get_frame(self, index, convert_to_rgb=1):
        fn = self.files[index]
        self.file_path = fn
        if self.save_as_bin:
            bin_path = os.path.join(self.roi_bin_dir, os.path.splitext(os.path.basename(fn))[0] + ".bin")
            if os.path.isfile(bin_path):
                try:
                    image = np.fromfile(open(bin_path, 'rb'), dtype=np.uint8)
                    if self.channels == 1:
                        image = image.reshape((self.roi_height, self.roi_width))
                    else:
                        image = image.reshape((self.roi_height, self.roi_width, self.channels))
                    return image
                except:
                    pass
        image = cv2.imread(self.file_path)
        if convert_to_rgb:
            image = image[:, :, ::-1].copy()
        if self.roi is not None:
            # print('applying ROI: ', self.roi)
            xmin = self.roi['xmin']
            ymin = self.roi['ymin']
            xmax = self.roi['xmax']
            ymax = self.roi['ymax']
            if self.channels == 1:
                image = image[ymin:ymax, xmin:xmax, :].copy()
            else:
                image = image[ymin:ymax, xmin:xmax].copy()
        if self.save_as_bin:
            try:
                image.astype(np.uint8).tofile(open(bin_path, 'wb'))
            except IOError:
                print('Bin file could not be written to disk')
                pass
        return image

    def get_frames_list(self):
        return [os.path.basename(file) for file in self.files]

    def get_frame_by_name(self, fn, convert_to_rgb=1):
        self.file_path = os.path.join(self.directory_fn, fn)
        if self.save_as_bin:
            bin_path = os.path.join(self.roi_bin_dir, os.path.splitext(fn)[0] + ".bin")
            if os.path.isfile(bin_path):
                try:
                    image = np.fromfile(open(bin_path, 'rb'), dtype=np.uint8)
                    if self.channels == 1:
                        image = image.reshape((self.roi_height, self.roi_width))
                    else:
                        image = image.reshape((self.roi_height, self.roi_width, self.channels))
                    return image
                except:
                    pass
        image = cv2.imread(self.file_path)
        if convert_to_rgb:
            image = image[:, :, ::-1].copy()

        if self.roi is not None:
            # print('applying ROI: ', self.roi)
            xmin = self.roi['xmin']
            ymin = self.roi['ymin']
            xmax = self.roi['xmax']
            ymax = self.roi['ymax']
            if len(image.shape) == 3:
                image = image[ymin:ymax, xmin:xmax, :].copy()
            else:
                image = image[ymin:ymax, xmin:xmax].copy()
        if self.save_as_bin:
            try:
                image.astype(np.uint8).tofile(open(bin_path, 'wb'))
            except IOError:
                print('Bin file could not be written to disk')
                pass
        return image

    def get_path(self):
        return self.directory_fn

    def get_background_samples(self, num_background_samples=100):
        self.background_sample_interval = int(self.num_frames / num_background_samples)
        background_samples = []
        background_samples_frame_id = []
        for j in np.arange(0, self.num_frames, self.background_sample_interval):
            image = cv2.imread(self.files[j])[:, :, ::-1]
            background_samples.append(image)
            background_samples_frame_id.append(j)
        return background_samples, background_samples_frame_id

    def get_file_path(self):
        return self.file_path


def get_frames_reader(path, save_as_bin=False, **kwargs):
    if os.path.isdir(path):
        return DirectoryReader(path, save_as_bin=save_as_bin, **kwargs)
    elif os.path.isfile(path):
        return VideoReader(path, save_as_bin=save_as_bin, **kwargs)
    else:
        raise SystemError('Invalid path provided: {:s}'.format(path))
