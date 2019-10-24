#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
import os
import cv2
from pprint import pprint

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import numpy as np
from libs.shape import Shape

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize,databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.out_fname = None
        if self.filename is not None:
            self.out_fname = self.filename + XML_EXT
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult, bbox_source, id_number, score,
                  mask, mask_img=None):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        bndbox['bbox_source'] = bbox_source
        bndbox['id_number'] = id_number
        bndbox['score'] = score
        bndbox['mask'] = mask
        bndbox['mask_img'] = mask_img
        self.boxlist.append(bndbox)

    def appendObjects(self, top):

        # self.mask_images = {}

        for obj_id, each_object in enumerate(self.boxlist):
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            try:
                name.text = unicode(each_object['name'])
            except NameError:
                # Py3: NameError: name 'unicode' is not defined
                name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(each_object['ymax']) == int(self.imgSize[0]) or (int(each_object['ymin'])== 1):
                truncated.text = "1" # max == height or min
            elif (int(each_object['xmax'])==int(self.imgSize[1])) or (int(each_object['xmin'])== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])
            bbox_source = SubElement(object_item, 'bbox_source')
            bbox_source.text = str(each_object['bbox_source'])
            id_number = SubElement(object_item, 'id_number')
            id_number.text = str(each_object['id_number'])
            score = SubElement(object_item, 'score')
            score.text = str(each_object['score'])
            mask_pts = each_object['mask']
            mask_img = each_object['mask_img']

            if mask_img is not None:
                mask_img_name = '{}_{}.png'.format(os.path.splitext(self.filename)[0], obj_id)
                mask_img_dir = self.foldername if self.out_fname is None else os.path.dirname(self.out_fname)
                mask_img_path = os.path.join(mask_img_dir, mask_img_name)

                mask_img_name_elem = SubElement(object_item, 'mask_filename')
                mask_img_name_elem.text = mask_img_name

                _mask_img = (mask_img*255.0).astype(np.uint8)

                cv2.imwrite(mask_img_path, _mask_img)

                # self.mask_images[mask_img_name] = mask_img
                if mask_pts is None or not mask_pts:
                    _, mask_img_bin = cv2.threshold(mask_img.astype(np.float64), 0.5, 1, cv2.THRESH_BINARY)
                    _, mask_pts = Shape.contourPtsFromMask(mask_img_bin.astype(np.uint8))
                    mask_pts = [[x + each_object['xmin'], y + each_object['ymin'], f] for x, y, f in mask_pts]

            if mask_pts is not None and mask_pts:
                mask_txt = '{},{},{};'.format(*mask_pts[0])
                for _pt in mask_pts[1:]:
                    mask_txt = '{} {},{},{};'.format(mask_txt, _pt[0], _pt[1], _pt[2])

                mask = SubElement(object_item, 'mask')
                mask.text = mask_txt


    def save(self, targetFile=None, _filename=None, _imgSize=None):

        if _filename is not None:
            self.filename =  _filename
        if _imgSize is not None:
            self.imgSize =  _imgSize

        if targetFile is None:
            if self.out_fname is None:
                raise IOError('targetFile must be provided when out_fname is None')
            self.out_fname = self.filename + XML_EXT
        else:
            self.out_fname = targetFile

        root = self.genXML()

        self.appendObjects(root)
        out_file = codecs.open(self.out_fname, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult, bbox_source, id_number]
        self.shapes = []
        self.filepath = filepath
        self.verified = False

        self.filename = None
        # self.path = None
        self.width = None
        self.height = None
        self.depth = None

        self.parseXML()
        # try:
        #     self.parseXML()
        # except:
        #     pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, bndbox, difficult, bbox_source, id_number, score, mask, mask_img=None):
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        if label == "gate":
            points = [(xmin, ymin), (xmax, ymax)]
        else:
            points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, difficult, bbox_source, id_number, score, mask, mask_img))

    def parseXML(self):
        # assert self.filepath.endswith(XML_EXT), "Unsupported file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        self.filename = xmltree.find('filename').text
        # self.path = xmltree.find('path').text
        size_iter = xmltree.find('size')
        self.width = int(size_iter.find("width").text)
        self.height = int(size_iter.find("height").text)
        self.depth = int(size_iter.find("depth").text)
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            # print('Here we are !')
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text
            # Add chris
            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            try:
                bbox_source = object_iter.find('bbox_source').text
            except AttributeError:
                bbox_source = 'ground_truth'
            try:
                id_number = object_iter.find('id_number').text
            except AttributeError:
                id_number = 0

            score_elm = object_iter.find('score')
            if score_elm is None:
                score = '-1'
            else:
                score = score_elm.text
            if id_number == "None":
                id_number = None
            else:
                id_number = int(id_number)

            try:
                mask = object_iter.find('mask').text
                mask = [k.strip().split(',') for k in mask.strip().split(';') if k]
                # pprint(mask)
                mask = [(float(k[0]), float(k[1]), int(k[2])) for k in mask]
            except AttributeError:
                mask = None

            try:
                mask_filename = object_iter.find('mask_filename').text
                if mask_filename:
                    mask_filepath = os.path.join(os.path.dirname(self.filepath), mask_filename)
                    mask_img = cv2.imread(mask_filepath)
                else:
                    mask_img = None
            except AttributeError:
                mask_img = None

            self.addShape(label, bndbox, difficult, bbox_source, id_number, score, mask, mask_img)

        return True

