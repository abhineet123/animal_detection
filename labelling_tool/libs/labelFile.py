# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage

from base64 import b64encode, b64decode
from libs.pascal_voc_io import PascalVocWriter
from libs.pascal_voc_io import XML_EXT
import os.path
import sys


class LabelFileError(Exception):
    pass


class LabelFile(object):
    # It might be changed as window creates. By default, using XML ext
    # suffix = '.lif'
    suffix = XML_EXT

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        self.verified = False

    def savePascalVocFormat(self, filename, shapes, imagePath, imageData,
                            lineColor=None, fillColor=None, databaseSrc=None):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        # imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        imageShape = [imageData.height(), imageData.width(),
                      1 if imageData.isGrayscale() else 3]
        writer = PascalVocWriter(imgFolderName, imgFileName,
                                 imageShape, localImgPath=imagePath)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            bbox_source = shape['bbox_source']
            id_number = shape['id_number']
            score = shape['score']
            mask = shape['mask']
            # try:
            #     score = shape['score']
            # except KeyError:
            #     score = -1
            # Add Chris
            difficult = int(shape['difficult'])
            bndbox = LabelFile.convertPoints2BndBox(points, label)
            writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3],
                             label, difficult, bbox_source, id_number, score, mask)

        writer.save(targetFile=filename)
        return

    def toggleVerify(self):
        self.verified = not self.verified

    @staticmethod
    def isLabelFile(filename):
        fileSuffix = os.path.splitext(filename)[1].lower()
        return fileSuffix == LabelFile.suffix

    @staticmethod
    def convertPoints2BndBox(points, label, allow_zero=0, convert_to_int=1):
        if label == 'gate':
            xmin, ymin = points[0]
            xmax, ymax = points[1]
        else:
            xmin = float('inf')
            ymin = float('inf')
            xmax = float('-inf')
            ymax = float('-inf')
            for p in points:
                x = p[0]
                y = p[1]
                xmin = min(x, xmin)
                ymin = min(y, ymin)
                xmax = max(x, xmax)
                ymax = max(y, ymax)

        if not allow_zero:
            # Martin Kersner, 2015/11/12
            # 0-valued coordinates of BB caused an error while
            # training faster-rcnn object detector.
            if xmin < 1:
                xmin = 1

            if ymin < 1:
                ymin = 1

        if convert_to_int:
            return int(xmin), int(ymin), int(xmax), int(ymax)
        return xmin, ymin, xmax, ymax
