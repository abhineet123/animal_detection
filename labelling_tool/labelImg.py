#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import os.path
import os
import re
import sys
import cv2
import subprocess
import time
import shutil
import logging
import glob, argparse
import pandas as pd

from functools import partial
from collections import defaultdict
import threading
import socket
from libs.netio import send_msg_to_connection, recv_from_connection
from tracking.Visualizer import Visualizer, VisualizerParams
from tracking.Utilities import processArguments
from augment_mask import generateAugmentedSequence, readBackgroundData

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip

        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import resources
# Add internal libs
from libs.constants import *
from libs.lib import struct, newAction, newIcon, addActions, fmtShortcut, \
    computeOverlap, computeASTDLT, normalizePts, dotdict
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR_GROUND_TRUTH, DEFAULT_FILL_COLOR
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError
from libs.pascal_voc_io import PascalVocReader, PascalVocWriter
from libs.pascal_voc_io import XML_EXT
from libs.ustr import ustr
from libs.frames_readers import VideoReader, DirectoryReader
from pyqtwindows.WindowMixin import WindowMixin
# from pyqtwindows.tracking_settings_dialog import TrackingSettingDialog
# from pyqtwindows.tracking_server_log import TrackingServerLog
from pyqtwindows.run_dialog import RunDialog
from to_mask_seq import save_masks

from libs.netio import bindToPort

import numpy as np

__appname__ = 'labelImg'


# Utility functions and classes.

def have_qstring():
    '''p3/qt5 get rid of QString wrapper as py3 has native unicode str type'''
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))


def util_qt_strlistclass():
    return QStringList if have_qstring() else list


# PyQt5: TypeError: unhashable type: 'QListWidgetItem'
class HashableQListWidgetItem(QListWidgetItem):
    def __init__(self, *args):
        super(HashableQListWidgetItem, self).__init__(*args)

    def __hash__(self):
        return hash(id(self))


class MaskParams:
    """
    :type disp_size: tuple
    :type border_size: tuple
    :type del_thresh: int
    :type mag_thresh_t: float
    :type mag_win_size: int
    :type mag_patch_size: int
    :type gen_method: int
    :type load_boxes: int | bool
    :type show_magnified_window: int | bool
    :type show_binary: int | bool
    :type show_pts: int | bool
    """

    def __init__(self):
        self.disp_size = (1000, 1000)
        self.border_size = (10, 10)
        self.min_box_border = (1, 1)
        self.del_thresh = 15
        self.show_magnified_window = 1
        self.mag_patch_size = 50
        self.mag_win_size = 800
        self.mag_thresh_t = 0.05
        self.load_boxes = 0
        self.gen_method = 1
        self.show_binary = 0
        self.show_pts = 0
        self.save_test = 0
        self.save_raw = 0
        self.hed_model_path = '../hed_cv/hed_model'
        self.help = {
            'disp_size': "Size of the window shown for drawing the mask",
            'border_size': "Size of border around the bounding box to include in the mask window",
            'min_box_border': "minimum border to be left around the bounding box to avoid image boundary "
                              "aligned boxes that csn mess up training",
            'del_thresh': "Distance threshold for deleting the existing mask points",
            'show_magnified_window': "Show magnified window around the cursor location",
            'mag_patch_size': "size of patch around the cursor location shown in the magnified window ",
            'mag_win_size': "magnified window size",
            'mag_thresh_t': "minimum time in seconds between successive updates of the magnifying window window size",
            'gen_method': "method used for generating masks: "
                          "0: normalized AST computed by DLT;"
                          "1: simple translation and scaling",
            'show_binary': "Show binary mask in painting mode",
            'show_pts': "Show individual points",
            'save_test': "save unlabeled objects in a separate test sequence",
            'save_raw': "save rsw labels while saving mask sequences",
            'hed_model_path': "hed_model_path",
        }


class LabelingParams:
    """
    :type cfg: str
    :type predef_class_file: str
    :type file_name: str
    :type load_prev: int
    :type verbose: int | bool
    :type mask: MaskParams
    :type visualizer: VisualizerParams
    """

    def __init__(self):
        self.cfg = 'params.cfg'
        self.load_prev = 1
        self.file_name = ''
        self.predef_class_file = 'data/predefined_classes_cell.txt'
        self.verbose = 0
        self.max_boxes = 0
        self.mask = MaskParams()
        self.visualizer = VisualizerParams()

        self.help = {
            'cfg': 'optional ASCII text file from where parameter values can be read;'
                   'command line parameter values will override the values in this file',
            'load_prev': "Load previously opened file or directory",
            'file_name': "File or folder to load",
            'predef_class_file': "Path to the file containing the list of predefined classes",
            'verbose': 'show detailed diagnostic messages',
            'max_boxes': 'max_boxes to retain per image - can be used to remove nonexistent annotations from xml files',
            'mask': 'parameters for the mask labeling module',
            'visualizer': 'parameters for the visualizer module',
        }


class MainWindow(QMainWindow, WindowMixin):
    """
    :type params: LabelingParams
    :type logger: logging
    :type base_title: str
    :type frames_reader: VideoReader | DirectoryReader | None
    """

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, params, logger):
        super(MainWindow, self).__init__()

        self.params = params
        self.logger = logger

        # self.args = args
        # print('mask_disp_size : {}'.format(self.args.mask_disp_size))

        defaultFilename = self.params.file_name
        defaultPrefdefClassFile = self.params.predef_class_file
        load_prev = self.params.load_prev

        hed_model_path = params.mask.hed_model_path
        if hed_model_path:
            try:
                sys.path.append("../..")
                from hed_cv.detect_edges_image import CropLayer
                protoPath = os.path.sep.join([hed_model_path,
                                              "deploy.prototxt"])
                modelPath = os.path.sep.join([hed_model_path,
                                              "hed_pretrained_bsds.caffemodel"])
                # print('readNetFromCaffe')
                self.hed_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
                # print('dnn_registerLayer')
                cv2.dnn_registerLayer("Crop", CropLayer)
            except BaseException as e:
                print('HED is not available: {}'.format(e))
                self.hed_net = None

        self.base_title = 'Labelling Tool'
        self.full_title = self.base_title
        self.setWindowTitle(self.full_title)
        # Save as Pascal voc xml
        self.saveDir = None
        self.usingPascalVocFormat = True
        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None
        self.lastMaskDir = None
        self.lastOpenVideoFile = None

        self.run_dialog = RunDialog(main_window=self)

        self.CWD = os.path.abspath(os.getcwd())
        # os.system('bash {}'.format(os.path.join(self.CWD, 'free_all_ports.sh')))

        # Whether we need to save or not.
        self.dirty = False

        self.frames_reader = None

        self.show_ground_truth_bboxes_bool = True
        self.show_detector_bboxes_bool = True
        self.show_tracker_bboxes_bool = True
        self.show_gates_bool = True

        self.port = 3000
        self.start_server_thread()
        self.add_bboxes_threads = []
        self.get_bboxes_threads = []
        self.curr_frames_list = []

        self.jump_interval = 1

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = "firefox"
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        self.class_labels = ['bear', 'moose', 'deer', 'coyote']

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Create a widget for using default label
        self.useDefaultLabelCheckbox = QCheckBox(u'Use default label')
        self.useDefaultLabelCheckbox.setChecked(True)
        self.useDefaultLabelCheckbox.stateChanged.connect(self.default_label_checkbox)

        self.default_label = self.labelHist[0]

        self.defaultLabelComboBox = QComboBox()
        self.defaultLabelComboBox.setEditable(False)
        self.defaultLabelComboBox.addItems(self.labelHist)

        self.defaultLabelComboBox.activated.connect(self.set_default_label)
        self.defaultLabelComboBox.currentIndexChanged.connect(self.set_default_label)
        self.defaultLabelComboBox.editTextChanged.connect(self.set_default_label)

        self.defaultLabelComboBox.setCurrentText(self.default_label)

        # self.defaultLabelTextLine = QLineEdit()
        # self.defaultLabelTextLine.setText('bear')

        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelComboBox)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        # Create a widget for edit and diffc button
        # self.diffcButton = QCheckBox(u'difficult')
        # self.diffcButton.setChecked(False)
        # self.diffcButton.stateChanged.connect(self.btnstate)
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout
        listLayout.addWidget(self.editButton)
        # listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(useDefaultLabelContainer)

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)

        # self.show_label = QLabel("Show: ")
        # show_label_layout = QHBoxLayout()
        # show_label_layout.addWidget(self.show_label)
        # show_label_widget = QWidget()
        # show_label_widget.setLayout(show_label_layout)
        # listLayout.addWidget(show_label_widget)

        # self.show_annotated_bboxes_checkbox = QCheckBox(u'annotations')
        # self.show_annotated_bboxes_checkbox.setChecked(True)
        # self.show_detected_bboxes_checkbox = QCheckBox(u'detections')
        # self.show_detected_bboxes_checkbox.setChecked(True)
        # self.show_tracked_bboxes_checkbox = QCheckBox(u'tracking result')
        # self.show_tracked_bboxes_checkbox.setChecked(True)
        # self.show_text_checkbox = QCheckBox(u'text')
        # self.show_text_checkbox.setChecked(False)
        # show_boxes_qh_box_layout = QHBoxLayout()
        # show_boxes_qh_box_layout.addWidget(self.show_annotated_bboxes_checkbox)
        # show_boxes_qh_box_layout.addWidget(self.show_detected_bboxes_checkbox)
        # show_boxes_qh_box_layout.addWidget(self.show_tracked_bboxes_checkbox)
        # show_boxes_qh_box_layout.addWidget(self.show_text_checkbox)
        # show_boxes_container = QWidget()
        # show_boxes_container.setLayout(show_boxes_qh_box_layout)
        # listLayout.addWidget(show_boxes_container)
        # 
        # self.track_worked_frames_checkbox = QCheckBox(u'Track frames user worked on')
        # self.track_worked_frames_checkbox.setChecked(True)
        # misc_qh_box_layout = QHBoxLayout()
        # misc_qh_box_layout.addWidget(self.track_worked_frames_checkbox)
        # misc_container = QWidget()
        # misc_container.setLayout(misc_qh_box_layout)
        # listLayout.addWidget(misc_container)

        self.dock = QDockWidget(u'Box Labels', self)
        self.dock.setObjectName(u'Labels')
        self.dock.setWidget(labelListContainer)

        # self.useDefaultLabelCheckbox = QCheckBox(u'Use default label')
        # self.useDefaultLabelCheckbox.setChecked(False)
        # self.defaultLabelTextLine = QLineEdit()
        # useDefaultLabelQHBoxLayout = QHBoxLayout()
        # useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        # useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        # useDefaultLabelContainer = QWidget()
        # useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        # Tzutalin 20160906 : Add file list and dock to move faster

        jump_interval_label = QLabel("Jump interval:")
        # self.jump_interval_textbox = QLineEdit()
        # self.jump_interval_textbox.textChanged.connect(self.set_jump_interval)

        self.jump_interval_textbox = QComboBox()
        self.jump_interval_textbox.setEditable(True)
        self.jump_interval_textbox.addItems(['1', '2', '5', '10', '20', '50',
                                             '100', '200', '500', '1000'])
        self.jump_interval_textbox.activated.connect(self.set_jump_interval)
        self.jump_interval_textbox.currentIndexChanged.connect(self.set_jump_interval)
        self.jump_interval_textbox.editTextChanged.connect(self.set_jump_interval)

        self.jump_interval_textbox.setCurrentText(str(self.jump_interval))

        jump_interval_layout = QHBoxLayout()
        jump_interval_layout.addWidget(jump_interval_label)
        jump_interval_layout.addWidget(self.jump_interval_textbox)
        jump_interval_widget = QWidget()
        jump_interval_widget.setLayout(jump_interval_layout)

        self.current_frame_label = QLabel("Current frame: ")
        current_frame_layout = QHBoxLayout()
        current_frame_layout.addWidget(self.current_frame_label)
        current_frame_widget = QWidget()
        current_frame_widget.setLayout(current_frame_layout)

        go_to_label = QLabel("Go to:")
        self.go_to_textbox = QLineEdit()
        self.go_to_button = QPushButton("Go")
        self.go_to_button.clicked.connect(self.go_to_button_clicked)
        go_to_layout = QHBoxLayout()
        go_to_layout.addWidget(go_to_label)
        go_to_layout.addWidget(self.go_to_textbox)
        go_to_layout.addWidget(self.go_to_button)
        go_to_widget = QWidget()
        go_to_widget.setLayout(go_to_layout)
        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(jump_interval_widget)
        filelistLayout.addWidget(current_frame_widget)
        filelistLayout.addWidget(go_to_widget)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(u'File List', self)
        self.filedock.setObjectName(u'Files')
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(main_window=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        # Tzutalin 20160906 : Add file list and dock to move faster
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.dockFeatures = QDockWidget.DockWidgetClosable \
                            | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        # Actions
        action = partial(newAction, self)
        quit = action('&Quit', self.close,
                      ['Ctrl+Q', 'Escape'], 'quit', u'Quit application Ctrl+Q')

        openVideo = action('&Open Video File', self.openVideoFile,
                           'Ctrl+O', 'open', u'Open Video File (Ctrl+O)')

        opendir = action('&Open Directory', self.openDir,
                         'Ctrl+u', 'open', u'Open Directory containing Images (Ctrl+U)')

        # changeSavedir = action('&Change Save Dir', self.changeSavedir,
        #                        'Ctrl+r', 'open', u'Change default saved Annotation dir')

        openNextImg = action('&Next Image', self.openNextImg,
                             ['d', 'PgUp', "'"], 'next', u'Open Next Image (d, PageUp)')

        openPrevImg = action('&Previous Image', self.openPrevImg,
                             ['a', 'PgDown', '"'], 'prev', u'Open Previous Image (a, PageDown)')

        # verify = action('&Verify Image', self.verifyImg,
        #                 'space', 'verify', u'Verify Image')

        save = action('&Save', self.saveFile,
                      'Ctrl+S', 'save', u'Save labels to file (Ctrl+S)', enabled=False)
        saveAs = action('&Save As', self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', u'Save labels to a different file (Ctrl+Shift+S)',
                        enabled=False)
        close = action('&Close', self.closeFile,
                       'Ctrl+W', 'close', u'Close current file (Ctrl+W)')

        createMode = action('Create\nRectBox', self.setCreateMode,
                            'Ctrl+N', 'new', u'Start drawing Boxes (Ctrl+N)', enabled=False)

        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxes (Ctrl+J)', enabled=False)

        create = action('Create\nRectBox', self.createShape,
                        ['w', 'Insert'], 'new', u'Draw a new Box (w, Insert)', enabled=False)

        create_mask = action('Create\nMask', self.createMask,
                             ['Shift+w', 'Shift+Insert', 'Ctrl+Space', 'Home', '/'], 'mask',
                             u'Add a mask to the selected box (or edit an existing mask)'
                             u' (Shift+w, Shift+Insert, Ctrl+Space, Home, Double Click)',
                             enabled=False)

        delete = action('Delete\nRectBox', self.deleteSelectedShape,
                        ['Delete', 's'], 'delete', u'Delete the selected box (Delete, s)', enabled=False)
        delete_target = action('Delete\nTarget', self.deleteSelectedTarget,
                               'Shift+Delete', 'delete',
                               u'Delete the selected target in current and future frames (Shift+Delete)', enabled=False)
        copy = action('&Duplicate\nRectBox', self.copySelectedShape,
                      'Ctrl+D', 'copy', u'Create a duplicate of the selected Box',
                      enabled=False)
        track = action('&Track\nTarget', self.trackSelectedObject,
                       ['space', 't', 'Ctrl+Shift+T'], 'track', u'Track the target within the selected box',
                       enabled=False)
        set_roi = action('&Set ROI', self.setROI,
                         'Ctrl+Alt+R', 'set_roi', u'Set the selected box as the Region Of Interest',
                         enabled=False)
        delete_all_in_curr_image = action("Delete all\nin current image", self.delete_all_in_current_image,
                                          None, 'delete', enabled=False)
        delete_all_annotations = action("Delete all annotations", self.delete_all_annotations, None,
                                        'delete', enabled=False)

        advancedMode = action('&Advanced Mode', self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', u'Switch to advanced mode',
                              checkable=True)

        hideAll = action('&Hide\nRectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', u'Hide all Boxs',
                         enabled=False)
        showAll = action('&Show\nRectBox', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', u'Show all Boxs',
                         enabled=False)

        help = action('&Tutorial', self.tutorial, 'Ctrl+T', 'help',
                      u'Show demos')

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action('Zoom &In', partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', u'Increase zoom level', enabled=False)
        zoomOut = action('&Zoom Out', partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', u'Decrease zoom level', enabled=False)
        zoomOrg = action('&Original size', partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', u'Zoom to original size', enabled=False)
        fitWindow = action('&Fit Window', self.setFitWindow,
                           'Ctrl+F', 'fit-window', u'Zoom follows window size',
                           checkable=True, enabled=False)
        fitWidth = action('Fit &Width', self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', u'Zoom follows window width',
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action('&Edit Label', self.editLabel,
                      'Ctrl+E', 'edit', u'Modify the label of the selected Box',
                      enabled=False)
        self.editButton.setDefaultAction(edit)

        labels = self.dock.toggleViewAction()
        labels.setText('Show/Hide Label Panel')
        labels.setShortcut('Ctrl+Shift+L')

        # Lavel list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete, delete_target))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        save_map = action("Save as mAP", self.save_map, None, 'save')
        save_yolo = action("Save as YOLO", self.save_yolo, None, 'save')
        save_csv = action("Save as CSV", self.save_csv, None, 'save')
        visualize = action("Visualize", self.visualize, None, 'visualize')
        augment = action("Augment", self.augment, None, 'augment')
        generate_mask = action("Motion Detection", self.generate_mask, '?', 'generate_mask',
                               tip='Generate mask for one or more target boxes by extrapolating from a previous frame '
                                   'using motion information from the corresponding boxes')
        edge_detection = action("Edge Detection", self.edge_detection, '?', 'edge',
                                tip='Generate mask for one or more target boxes by extrapolating from a previous frame '
                                    'using motion information from the corresponding boxes')
        save_masks = action("Save Masks", self.save_masks, None, 'save',
                            tip='Save each object mask as a binary image along with the corresponding patch image'
                            )

        load_masks = action("Load Masks", self.load_masks, None, 'load',
                            tip='Load masks from existing binary images'
                            )
        delete_masks = action("Delete Masks", self.delete_masks, None, 'delete',
                              tip='Delete all masks in the current sequence'
                              )
        opendir = action('&Open Directory', self.openDir,
                         'Ctrl+u', 'open', u'Open Directory containing Images (Ctrl+U)')

        remove_roi = action("Remove\nROI", self.removeROI)
        load_roi = action("Load\nROI", self.loadROI)
        save_roi = action("Save\nROI", self.saveROI)
        # add_gate = action("Add Gate", self.addGate)

        record_frames_list = action("Track frames\nuser worked on", checkable=True)
        record_frames_list.setChecked(True)

        # tracking_settings_dialog = action("Tracker\nSettings", self.tracking_settings_dialog, None)
        # view_tracking_server_log = action("Tracker\nLog", self.view_tracking_server_log)

        # Toggle bboxes
        show_ground_truth_bboxes = action("Show\nGround Truth BBoxes", self.show_ground_truth_bboxes, checkable=True)
        show_tracking_bboxes = action("Show\nTracked BBoxes", self.show_tracking_bboxes, checkable=True)
        show_ground_truth_bboxes.setChecked(self.show_ground_truth_bboxes_bool)
        show_tracking_bboxes.setChecked(self.show_tracker_bboxes_bool)
        toggle_show_text = action("Show labels", self.toggle_show_text, checkable=True)
        toggle_show_text.setChecked(True)

        self.delete_bin = 1
        toggle_delete_bin = action("Delete\nbinary files", self.toggle_delete_bin, checkable=True)
        toggle_delete_bin.setChecked(True)

        # clean action
        clear_saved_settings = action("Clear\nSaved Settings", self.clear_saved_settings)

        # Store actions for further handling.
        self.actions = struct(save=save, saveAs=saveAs, open=openVideo, close=close,
                              create=create, create_mask=create_mask,
                              delete=delete, delete_target=delete_target,
                              edit=edit, copy=copy,
                              track=track,

                              set_roi=set_roi,
                              createMode=createMode, editMode=editMode, advancedMode=advancedMode,
                              delete_all_in_curr_image=delete_all_in_curr_image,
                              delete_all_annotations=delete_all_annotations,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(
                                  openVideo, opendir, save, saveAs, close, quit),
                              beginner=(), advanced=(),
                              editMenu=(
                                  edit, create_mask, copy, delete, delete_target, delete_all_in_curr_image,
                                  delete_all_annotations,
                                  track, set_roi),
                              beginnerContext=(create, create_mask, edit, copy, delete, delete_target, track, set_roi),
                              advancedContext=(createMode, create_mask, editMode, edit, copy,
                                               delete, delete_target, track, set_roi),
                              onLoadActive=(
                                  close, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll),

                              save_map=save_map,
                              save_yolo=save_yolo,
                              save_csv=save_csv,
                              save_masks=save_masks,

                              load_masks=load_masks,
                              delete_masks=delete_masks,

                              visualize=visualize,
                              augment=augment,

                              generate_mask=generate_mask,
                              edge_detection=edge_detection,

                              load_roi=load_roi,
                              remove_roi=remove_roi,
                              save_roi=save_roi,

                              record_frames_list=record_frames_list,
                              # tracking_settings_dialog=tracking_settings_dialog,
                              # view_tracking_server_log=view_tracking_server_log,

                              show_ground_truth_bboxes=show_ground_truth_bboxes,
                              show_tracking_bboxes=show_tracking_bboxes,

                              toggle_show_text=toggle_show_text,
                              toggle_delete_bin=toggle_delete_bin,

                              run_tools=(load_roi,
                                         remove_roi,
                                         save_roi,
                                         # None,
                                         # tracking_settings_dialog,
                                         # view_tracking_server_log,
                                         None,
                                         show_ground_truth_bboxes,
                                         show_tracking_bboxes,
                                         toggle_show_text,
                                         None,
                                         toggle_delete_bin,
                                         None,
                                         save_map,
                                         save_yolo,
                                         save_csv,
                                         save_masks,
                                         None,
                                         load_masks,
                                         delete_masks,
                                         None,
                                         visualize,
                                         augment,
                                         generate_mask,
                                         edge_detection,
                                         ))

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu,
            settings=self.menu('&Settings'),
            object_detection=self.menu('R&un'),
            help=self.menu('&Help'),
        )

        # Auto saving : Enable auto saving if pressing next
        self.autoSaving = QAction("Auto Saving Mode", self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(True)

        # Sync single class mode from PR#106
        self.singleClassMode = QAction("Single Class Mode", self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.lastLabel = None

        addActions(self.menus.file,
                   (openVideo, opendir,
                    self.menus.recentFiles,
                    save, saveAs, close,
                    None,
                    clear_saved_settings,
                    None,
                    quit))
        addActions(self.menus.help, (help,))
        addActions(self.menus.view, (
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth, None,
            show_ground_truth_bboxes,
            show_tracking_bboxes,
            toggle_show_text,
            toggle_delete_bin
        ))
        addActions(self.menus.settings, (
            self.autoSaving,
            record_frames_list,
            self.singleClassMode,
            # tracking_settings_dialog
        ))
        addActions(self.menus.object_detection, (load_roi,
                                                 remove_roi,
                                                 save_roi,
                                                 # None,
                                                 # None,
                                                 # view_tracking_server_log,
                                                 # None,
                                                 None,
                                                 save_map,
                                                 save_yolo,
                                                 save_csv,
                                                 save_masks,
                                                 None,
                                                 load_masks,
                                                 delete_masks,
                                                 None,
                                                 visualize,
                                                 augment,
                                                 None,
                                                 generate_mask,
                                                 edge_detection,
                                                 ))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.run_tools = self.run_toolbar('Run Tools', position=Qt.TopToolBarArea)
        self.actions.beginner = (
            openVideo, opendir, openNextImg, openPrevImg, save, None, create, create_mask,
            copy, delete, delete_target, delete_all_in_curr_image,
            None, zoomIn, zoom, zoomOut, fitWindow, fitWidth, track, set_roi)

        self.actions.advanced = (
            openVideo, opendir, openNextImg, openPrevImg, save, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()

        self.filePath = ustr(defaultFilename)
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        self.seq_settings = None
        self.deleted_targets = {}
        self.max_id = 0
        self.track_id = 0

        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        self.roi = settings.get(SETTING_ROI)
        if self.roi is not None:
            print('Using ROI: ', self.roi)

        self.jump_interval = settings.get(SETTING_JUMP_INTERVAL)
        if self.jump_interval is None:
            self.jump_interval = 1
        self.jump_interval_textbox.setCurrentText(str(self.jump_interval))

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = settings.get(SETTING_WIN_POSE, QPoint(0, 0))
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        self.lastMaskDir = ustr(settings.get(SETTING_LAST_MASK_DIR, None))
        self.lastBkgDir = ustr(settings.get(SETTING_LAST_BKG_DIR, None))
        self.lastOpenVideoFile = ustr(settings.get(SETTING_LAST_OPEN_VIDEO_FILE, None))
        if saveDir is not None and os.path.exists(saveDir):
            self.saveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (self.base_title, self.saveDir))
            self.statusBar().show()

        # or simply:
        # self.restoreGeometry(settings[SETTING_WIN_GEOMETRY]
        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, Shape.line_color))
        self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, Shape.fill_color))
        Shape.line_color = self.lineColor
        Shape.fill_color = self.fillColor
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Set detection settings
        self.DEFAULT_DETECTION_SAVE_DIR = 'tmp/trained_models'
        if not os.path.exists(self.DEFAULT_DETECTION_SAVE_DIR):
            os.makedirs(self.DEFAULT_DETECTION_SAVE_DIR)

        self.DEFAULT_TRACKING_SAVE_DIR = 'tmp/trained_targets'
        if not os.path.exists(self.DEFAULT_TRACKING_SAVE_DIR):
            os.makedirs(self.DEFAULT_TRACKING_SAVE_DIR)
        self.tracking_settings = settings.get(SETTING_TRACKING_SETTINGS)
        if self.tracking_settings is None:
            self.tracking_settings = struct(
                setting_choice="intersection",
                read_from_file_path=os.path.join(self.CWD, 'tracking_module', 'log'),
                cfg_path=os.path.join(self.CWD, 'tracking_module', 'cfg', 'params.cfg'),
                trained_target_path=os.path.join(self.CWD, 'tracking_module', 'log',
                                                 'trained_GRAM_detrac_1_MVI_20011_1_66.zip'),
                save_path=os.path.join(self.CWD, 'tracking_module', 'log'),
                mtf_cfg_path=os.path.join(self.CWD, 'tracking_module', 'cfg', 'mtf'),
            )

        self.curr_index = settings.get(SETTING_CURRENT_INDEX)
        # print('SETTING_CURRENT_INDEX: {}'.format(self.curr_index))
        if self.curr_index is None:
            self.curr_index = 0
        self.go_to_textbox.setText(str(self.curr_index))

        self.out_mask_size = settings.get(SETTING_OUT_MASK_SIZE)
        if self.out_mask_size is None:
            self.out_mask_size = self.params.mask.disp_size
        self.out_mask_border = settings.get(SETTING_OUT_MASK_BORDER)
        if self.out_mask_border is None:
            self.out_mask_border = self.params.mask.border_size

        self.out_mask_border = list(self.out_mask_border)

        self.in_mask_border = settings.get(SETTING_IN_MASK_BORDER)
        if self.in_mask_border is None:
            self.in_mask_border = self.params.mask.border_size

        self.in_mask_border = list(self.in_mask_border)

        # Populate the File menu dynamically.l
        self.updateFileMenu()
        # Since loading the file may take some time, make sure it runs in the
        # background.

        self.labelFile = None

        if load_prev == 1:
            prev_dir = self.settings.get(SETTING_LAST_OPEN_DIR)
            if prev_dir is not None:
                if os.path.isdir(prev_dir):
                    print('Loading previous directory: {}'.format(prev_dir))
                    # self.queueEvent(partial(self.loadFile, prev_dir))
                    # self.openDir(dirpath=prev_dir)
                    # self.openNextImg()
                    # self.openPrevImg()
                    self.queueEvent(partial(self.openDir, prev_dir))
                    if self.curr_index != 0:
                        self.queueEvent(self.go_to_button_clicked)
                else:
                    print('Previous directory does not exit: {}'.format(prev_dir))

        elif load_prev == 2:
            prev_file = self.settings.get(SETTING_LAST_OPEN_VIDEO_FILE)
            if prev_file is not None:
                if os.path.isfile(prev_file):
                    print('Loading previous file: {}'.format(prev_file))
                    self.openVideoFile(filename=prev_file)
                    self.openNextImg()
                    self.openPrevImg()
                    if self.curr_index != 0:
                        self.queueEvent(self.go_to_button_clicked)
                else:
                    print('Previous file does not exit: {}'.format(prev_file))
        else:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Create other dialogs
        # tracking_settings_dialog = TrackingSettingDialog(main_window=self)
        # tracking_settings_dialog.hide()

        # tracking_server_log = TrackingServerLog(main_window=self, settings_dialog=tracking_settings_dialog)

        # self.other_windows = struct(
        # tracking_settings_dialog=tracking_settings_dialog,
        # tracking_server_log=tracking_server_log,
        # all=(tracking_settings_dialog, tracking_server_log)
        # )

        self.toggle_show_text()

    ## Support Functions ##

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner() \
            else (self.actions.createMode, self.actions.editMode, self.actions.delete_all_in_curr_image,
                  self.actions.delete_all_annotations)
        addActions(self.menus.edit, actions + self.actions.editMenu)
        self.run_tools.clear()
        addActions(self.run_tools, self.actions.run_tools)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        # return self._beginner
        return True

    def advanced(self):
        return not self.beginner()

    ## Callbacks ##
    def tutorial(self):
        # subprocess.Popen([self.screencastViewer, self.screencast])
        pass

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f is not None and f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self, item=None):
        if not self.canvas.editing():
            return
        item = item if item else self.currentItem()
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            self.setDirty()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(ustr(item.text()))
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.curr_index = currIndex
                if self.autoSaving.isChecked():
                    self.saveFile()
                self.loadFile(filename)

    # Add chris
    def btnstate(self, item=None):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)

        # difficult = self.diffcButton.isChecked()

        try:
            shape = self.itemsToShapes[item]
        except:
            pass
            # Checked and Update
            # try:
            #     if difficult != shape.difficult:
            #         shape.difficult = difficult
            #         self.setDirty()
            #     else:  # User probably changed item visibility
            #         self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
            # except:
            #     pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.delete_target.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.track.setEnabled(selected)
        self.actions.create_mask.setEnabled(selected)
        self.actions.set_roi.setEnabled(selected)

    def addLabel(self, shape):
        if shape is None:
            return

        if shape.id_number is not None:
            label = '{:s}: {:s}'.format(
                str(shape.id_number), shape.label)
        else:
            label = shape.label
        item = HashableQListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult, bbox_source, id_number, score, mask, mask_img in shapes:
            shape = Shape(label=label, bbox_source=bbox_source, id_number=id_number,
                          score=score, mask=mask, mask_img=mask_img)
            for x, y in points:
                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.close()
            s.append(shape)
            self.addLabel(shape)
        self.hide_all_bboxes_sources(shapes=s)
        self.canvas.loadShapes(s)

    def format_shape(self, s):
        return dict(label=s.label,
                    line_color=s.line_color.getRgb()
                    if s.line_color != self.lineColor else None,
                    fill_color=s.fill_color.getRgb()
                    if s.fill_color != self.fillColor else None,
                    points=[(p.x(), p.y()) for p in s.points],
                    # add chris
                    difficult=s.difficult,
                    score=s.score,
                    bbox_source=s.bbox_source,
                    id_number=s.id_number,
                    mask=s.mask)

    def saveLabels(self, annotationFilePath):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        shapes = [self.format_shape(shape) for shape in self.canvas.shapes]
        # Can add different annotation formats here
        try:
            if self.usingPascalVocFormat is True:
                # print('Img: ' + self.filePath + ' -> Its xml: ' + annotationFilePath)
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                   self.lineColor.getRgb(), self.fillColor.getRgb())
            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
                                    self.lineColor.getRgb(), self.fillColor.getRgb())
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data',
                              u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            # self.diffcButton.setChecked(shape.difficult)

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """

        if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelComboBox.currentText():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            # Sync single class mode from PR#106
            if self.singleClassMode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.lastLabel = text
        else:
            text = self.defaultLabelComboBox.currentText()

        # Add Chris
        # self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            self.addLabel(self.canvas.setLastLabel(text))
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # print('filePath: {}'.format(filePath))

        unicodeFilePath = ustr(filePath)

        if unicodeFilePath and self.fileListWidget.count() > 0:
            index = self.mImgList.index(unicodeFilePath)
            fileWidgetItem = self.fileListWidget.item(index)
            fileWidgetItem.setSelected(True)

        if not unicodeFilePath:
            return False

        # Load image:
        # read data first and store for saving into label file.
        self.imageData = self.readImage(unicodeFilePath)
        self.labelFile = None
        image = self.imageData
        if image.isNull():
            self.errorMessage(u'Error opening file',
                              u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
            self.status("Error reading %s" % unicodeFilePath)
            return False
        self.status("Loaded %s" % os.path.basename(unicodeFilePath))
        self.image = image
        self.canvas.loadPixmap(QPixmap.fromImage(image))
        if self.labelFile:
            print('Here we are in labelFile')
            self.loadLabels(self.labelFile.shapes)
        self.setClean()
        self.canvas.setEnabled(True)
        self.adjustScale(initial=True)
        self.paintCanvas()
        self.addRecentFile(self.filePath)
        self.toggleActions(True)

        self.saveDir = self.get_annotation_path()
        self.filePath = unicodeFilePath

        currIndex = self.mImgList.index(self.filePath)

        if self.saveDir is not None:
            basename = os.path.basename(
                os.path.splitext(self.filePath)[0]) + XML_EXT
            xmlPath = os.path.join(self.saveDir, basename)
            # print('Reading boxes for frame {:s} from {:s}'.format(
            #     unicodeFilePath, xmlPath))
            self.loadPascalXMLByFilename(xmlPath, currIndex)
        else:
            xmlPath = os.path.splitext(filePath)[0] + XML_EXT
            # print('Reading boxes for frame {:s} from {:s}'.format(
            #     unicodeFilePath, xmlPath))
            self.loadPascalXMLByFilename(xmlPath, currIndex)

        if self.actions.record_frames_list.isChecked():
            jump_interval = int(self.jump_interval_textbox.currentText())

        self.setWindowTitle('{} ({})'.format(self.full_title, filePath))
        self.current_frame_label.setText("Current frame: {} / {}".format(currIndex,
                                                                         self.frames_reader.num_frames - 1))

        # Default : select last item if there is at least one item
        if self.labelList.count():
            self.labelList.setCurrentItem(self.labelList.item(self.labelList.count() - 1))
            self.labelList.item(self.labelList.count() - 1).setSelected(True)

        self.canvas.setFocus(True)
        self.actions.delete_all_in_curr_image.setEnabled(True)
        self.actions.delete_all_annotations.setEnabled(True)

        self.curr_index = currIndex
        return True

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull() \
                and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        # for window in self.other_windows.all:
        #     if window._is_running():
        #         if window != self.other_windows.tracking_server_log:
        #             window.show()
        #         window.stop()

        all_windows_closed = True

        # for window in self.other_windows.all:
        #     if window._is_running():
        #         print('Not all windows closed')
        #         all_windows_closed = False

        if not all_windows_closed:  # or not self.mayContinue():
            # ports = [3000, 3001, 3002]
            # for port in ports:
            #     cmd = 'lsof -t -i tcp:{:d} | xargs kill -9'.format(port)
            #     pid = subprocess.check_output(cmd, shell=True)
            event.ignore()
        else:
            if self.autoSaving.isChecked():
                self.saveFile()

            # for window in self.other_windows.all:
            #     window.destroy()

            settings = self.settings
            # If it loads images from dir, don't load it at the begining
            if self.dirname is None:
                settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
            else:
                settings[SETTING_FILENAME] = ''

            settings[SETTING_WIN_SIZE] = self.size()
            settings[SETTING_WIN_POSE] = self.pos()
            settings[SETTING_WIN_STATE] = self.saveState()
            settings[SETTING_LINE_COLOR] = self.lineColor
            settings[SETTING_FILL_COLOR] = self.fillColor
            settings[SETTING_RECENT_FILES] = self.recentFiles
            settings[SETTING_ADVANCE_MODE] = not self._beginner
            if self.saveDir is not None and len(self.saveDir) > 1:
                settings[SETTING_SAVE_DIR] = ustr(self.saveDir)
            else:
                settings[SETTING_SAVE_DIR] = ""

            if self.lastOpenDir is not None and len(self.lastOpenDir) > 1:
                settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
            else:
                settings[SETTING_LAST_OPEN_DIR] = ""

            if self.lastMaskDir is not None and len(self.lastMaskDir) > 1:
                settings[SETTING_LAST_MASK_DIR] = self.lastMaskDir
            else:
                settings[SETTING_LAST_MASK_DIR] = ""

            if self.lastBkgDir is not None and len(self.lastBkgDir) > 1:
                settings[SETTING_LAST_BKG_DIR] = self.lastBkgDir
            else:
                settings[SETTING_LAST_BKG_DIR] = ""

            settings[SETTING_TRACKING_SETTINGS] = self.tracking_settings
            settings[SETTING_CURRENT_INDEX] = self.curr_index

            settings[SETTING_CURRENT_INDEX] = self.curr_index

            settings[SETTING_OUT_MASK_SIZE] = self.out_mask_size
            settings[SETTING_OUT_MASK_BORDER] = self.out_mask_border

            settings[SETTING_IN_MASK_BORDER] = self.in_mask_border

            # print('SETTING_CURRENT_INDEX: {}'.format(settings[SETTING_CURRENT_INDEX]))

            if self.lastOpenVideoFile is not None:
                settings[SETTING_LAST_OPEN_VIDEO_FILE] = self.lastOpenVideoFile
            else:
                settings[SETTING_LAST_OPEN_VIDEO_FILE] = ""

            settings[SETTING_ROI] = self.roi
            settings[SETTING_JUMP_INTERVAL] = self.jump_interval

            settings.save()

            self.saveSeqSettings()

            event.accept()

            if self.frames_reader is not None:
                self.frames_reader.close(self.delete_bin)

            if os.name != 'nt':
                print('\nExiting...')
                ports = [3000]
                for port in ports:
                    cmd = 'lsof -t -i tcp:{:d} | xargs kill -SIGTERM'.format(port)
                    subprocess.check_output(cmd, shell=True)

                # force kill the process if sending terminate signal did not work
                for port in ports:
                    cmd = 'lsof -t -i tcp:{:d} | xargs kill -SIGKILL'.format(port)
                    subprocess.check_output(cmd, shell=True)
            else:
                os.kill(os.getpid(), 9)
                # os.system('kill %d' % os.getpid())

    ## User Dialogs ##

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def delete_masks(self):
        if not self.mayContinue():
            return

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)

        msg.setText("This operation cannot be undone. Press OK to continue.")
        msg.setWindowTitle("Delete Masks")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        # msg.buttonClicked.connect(msgbtn)

        retval = msg.exec_()
        # print(
        # "value of pressed message box button:", retval)

        if retval != QMessageBox.Ok:
            return

        if not self.saveDir or not os.path.isdir(self.saveDir):
            print('Folder containing the loaded boxes does not exist: {}'.format(self.saveDir))
            return None

        files = glob.glob(os.path.join(self.saveDir, '*.xml'))
        n_files = len(files)
        if n_files == 0:
            print('No xml annotation files found in {}'.format(self.saveDir))
            return None

        def getint(fn):
            basename = os.path.basename(fn)
            num = re.sub("\D", "", basename)
            try:
                return int(num)
            except:
                return 0

        files = sorted(files, key=getint)

        print('Loading annotations from {:d} files...'.format(n_files))

        file_id = 0
        n_boxes = 0
        for file in files:

            xml_reader = PascalVocReader(file)
            shapes = xml_reader.getShapes()
            n_shapes = len(shapes)

            filename_no_ext = os.path.splitext(xml_reader.filename)[0]

            channel = 1 if self.imageData.isGrayscale() else 3
            imageShape = [xml_reader.height, xml_reader.width, channel]

            xml_writer = PascalVocWriter(self.saveDir, xml_reader.filename, imageShape)

            for _id in range(n_shapes):
                shape = shapes[_id]
                label, points, _, _, difficult, bbox_source, id_number, score, _, _ = shape

                bndbox = LabelFile.convertPoints2BndBox(points, label)

                xml_writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3],
                                     label, difficult, bbox_source, id_number, score, None)
                n_boxes += 1

            xml_writer.save(targetFile=file)

            file_id += 1
            sys.stdout.write('\rDone {:d}/{:d} files with {:d} boxes'.format(
                file_id, n_files, n_boxes))
            sys.stdout.flush()

        basename = os.path.basename(
            os.path.splitext(self.filePath)[0]) + XML_EXT
        xmlPath = os.path.join(self.saveDir, basename)
        currIndex = self.mImgList.index(self.filePath)
        self.loadPascalXMLByFilename(xmlPath, currIndex)

    def loadMasksButton(self, b_text, mask_dialog_box, button_pushed):

        button_pushed[0] = 1

        mask_dialog_box.close()

        b_text = b_text.text()
        if '.' in b_text:
            self.in_mask_border[0] = float(b_text)
        else:
            self.in_mask_border[0] = int(b_text)

    def loadMasksDialogue(self):
        mask_dialog_box = QDialog()

        pos = dotdict()
        size = dotdict()

        # w_label = QCheckBox(mask_dialog_box)
        # w_label.setText('fixed_ar ')
        # w_label.move(*pos.w_label)
        #
        # pos.w_text = pos.w_label[0] + size.w_label, 25
        # size.w_text = 40, 20
        #
        # fixed_ar_cb = QCheckBox(mask_dialog_box)
        # fixed_ar_cb.setChecked(False)
        # fixed_ar_cb.move(*pos.w_text)
        # fixed_ar_cb.resize(*size.w_text)

        pos.b_label = 20, 30
        size.b_label = 80

        b_label = QLabel(mask_dialog_box)
        b_label.setText('border ')
        b_label.move(*pos.b_label)

        pos.b_text = pos.b_label[0] + size.b_label, 25
        size.b_text = 40, 20
        b_text = QLineEdit(mask_dialog_box)
        b_text.setText(str(self.in_mask_border[0]))
        b_text.move(*pos.b_text)
        b_text.resize(*size.b_text)

        mask_button = QPushButton("OK", mask_dialog_box)
        mask_button.move(pos.b_text[0] / 2, 65)

        button_pushed = [0, ]
        mask_button.clicked.connect(partial(self.loadMasksButton, b_text, mask_dialog_box, button_pushed))

        size.mask_dialog_box = pos.b_text[0] + size.b_text[0] + 20, 100

        mask_dialog_box.resize(*size.mask_dialog_box)
        mask_dialog_box.setWhatsThis(
            'floating point value for border specifies a fixed aspect ratio that was '
            'achieved by adding the appropriate horizontal or vertical border '
            'adaptively for each frame'
        )

        mask_dialog_box.setWindowTitle("Mask Border Settings")
        mask_dialog_box.setWindowModality(Qt.ApplicationModal)
        mask_dialog_box.exec_()

        return button_pushed[0]

    def load_masks(self):
        if not self.loadMasksDialogue():
            return

        in_border = self.in_mask_border[0]
        fixed_ar = 0
        if isinstance(in_border, float):
            fixed_ar = in_border

        if self.lastMaskDir is not None and len(self.lastMaskDir) > 1:
            path = os.path.dirname(self.lastMaskDir)
        else:
            path = os.path.dirname(self.frames_reader.get_path()) if self.frames_reader else os.path.expanduser("~")

        mask_src_dir = ustr(QFileDialog.getExistingDirectory(self,
                                                             '%s - Select Mask Directory' % self.base_title, path,
                                                             QFileDialog.ShowDirsOnly |
                                                             QFileDialog.DontResolveSymlinks))
        if not mask_src_dir or mask_src_dir is None:
            return

        self.lastMaskDir = mask_src_dir

        if fixed_ar:
            print('Using fixed aspect ratio: {}'.format(fixed_ar))

        img_exts = ('.jpg', '.bmp', '.jpeg', '.png')

        mask_src_files = [k for k in os.listdir(mask_src_dir) if
                          os.path.splitext(k.lower())[1] in img_exts]
        # src_file_list.sort(key=sortKey)

        if not self.saveDir or not os.path.isdir(self.saveDir):
            print('Folder containing the loaded boxes does not exist')
            return None

        files = glob.glob(os.path.join(self.saveDir, '*.xml'))
        n_files = len(files)
        if n_files == 0:
            print('No xml annotation files found')
            return None

        def getint(fn):
            basename = os.path.basename(fn)
            num = re.sub("\D", "", basename)
            try:
                return int(num)
            except:
                return 0

        files = sorted(files, key=getint)

        print('Loading annotations from {:d} files...'.format(n_files))

        file_id = 0
        n_boxes = 0
        for file in files:

            xml_reader = PascalVocReader(file)
            shapes = xml_reader.getShapes()
            n_shapes = len(shapes)

            filename_no_ext = os.path.splitext(xml_reader.filename)[0]

            matching_mask_files = [k for k in mask_src_files if k.startswith(filename_no_ext)]
            if not matching_mask_files:
                print('\nNo matching mask files found for {}'.format(xml_reader.filename))
                continue

            n_matching_mask_files = len(matching_mask_files)

            if n_matching_mask_files != n_shapes:
                raise IOError('{} :: number of matching mask files:{} does not match number of objects: {}'.format(
                    xml_reader.filename, n_matching_mask_files, n_shapes))

            self.frames_reader.get_frame_by_name(xml_reader.filename)

            channel = 1 if self.imageData.isGrayscale() else 3
            imageShape = [xml_reader.height, xml_reader.width, channel]

            xml_writer = PascalVocWriter(self.saveDir, xml_reader.filename, imageShape)

            for _id in range(n_shapes):
                shape = shapes[_id]
                mask_path = os.path.join(mask_src_dir, matching_mask_files[_id])

                new_mask_img = cv2.imread(mask_path)

                label, points, _, _, difficult, bbox_source, id_number, score, mask, mask_img = shape

                if mask is not None or mask_img is not None:
                    print('\n{} :: overwriting existing mask for object {}'.format(
                        xml_reader.filename, _id))

                bndbox = LabelFile.convertPoints2BndBox(points, label)

                xmin, ymin, xmax, ymax = bndbox

                if fixed_ar:
                    w, h = xmax - xmin, ymax - ymin
                    src_ar = float(w) / float(h)
                    if fixed_ar > src_ar:
                        border_x = int((h * fixed_ar - w) / 2.0)
                        border_y = 0
                    else:
                        border_y = int((w / fixed_ar - h) / 2.0)
                        border_x = 0
                else:
                    border_x = border_y = in_border

                xmin -= border_x
                ymin -= border_y

                xmax += border_x
                ymax += border_y

                box_w, box_h = xmax - xmin, ymax - ymin
                mask_h, mask_w = new_mask_img.shape[:2]

                scale_factor_x, scale_factor_y = float(mask_w) / float(box_w), float(mask_h) / float(box_h)

                contour_pts, mask_pts = Shape.contour_pts_from_mask(new_mask_img)

                mask_img = cv2.resize(new_mask_img, (box_w, box_h))

                mask = [(xmin + x / scale_factor_x, ymin + y / scale_factor_y, f)
                        for (x, y, f) in mask_pts]
                if self.params.mask.load_boxes:
                    mask_arr = np.asarray([(x, y) for x, y, _ in shape.mask])
                    _xmin, _ymin = np.min(mask_arr, axis=0)
                    _xmax, _ymax = np.max(mask_arr, axis=0)

                    if _xmin < _xmax and _ymin < _ymax:
                        # shape.points[0] = QPointF(xmin, ymin)
                        # shape.points[1] = QPointF(xmax, ymin)
                        # shape.points[2] = QPointF(xmax, ymax)
                        # shape.points[3] = QPointF(xmin, ymax)
                        bndbox = [_xmin, _ymin, _xmax, _ymax]
                    else:
                        raise IOError('Mask with invalid bounding box: {}'.format(
                            [xmin, ymin, xmax, ymax]))
                xml_writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3],
                                     label, difficult, bbox_source, id_number, score, mask, mask_img)
                n_boxes += 1

            xml_writer.save(targetFile=file)

            file_id += 1
            sys.stdout.write('\rDone {:d}/{:d} files with {:d} boxes'.format(
                file_id, n_files, n_boxes))
            sys.stdout.flush()

        basename = os.path.basename(
            os.path.splitext(self.filePath)[0]) + XML_EXT
        xmlPath = os.path.join(self.saveDir, basename)
        currIndex = self.mImgList.index(self.filePath)
        self.loadPascalXMLByFilename(xmlPath, currIndex)

    def openDir(self, dirpath=None):
        if not self.mayContinue():
            return

        if self.autoSaving.isChecked():
            self.saveFile()

        if self.frames_reader is not None:
            self.frames_reader.close(self.delete_bin)

        # print('dirpath: {}'.format(dirpath))

        if not dirpath or dirpath is None:
            if self.lastOpenDir is not None and len(self.lastOpenDir) > 1:
                path = os.path.dirname(self.lastOpenDir)
            else:
                path = os.path.dirname(self.frames_reader.get_path()) if self.frames_reader else os.path.expanduser("~")

            dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                            '%s - Open Directory' % self.base_title, path,
                                                            QFileDialog.ShowDirsOnly))

        if self.lastOpenDir is None or dirpath != self.lastOpenDir:
            print('Disabling ROI')
            self.roi = None

        # print('dirpath: {}'.format(dirpath))
        if dirpath is not None and len(dirpath) > 1:
            self.lastOpenDir = dirpath

        if dirpath:
            self.saveSeqSettings()

            frames_reader = DirectoryReader(dirpath, save_as_bin=0)
            if not frames_reader.setROI(self.roi):
                self.roi = None
            if frames_reader.num_frames == 0:
                QMessageBox.critical(self, "Error", "No frames detected! Please choose different path.", QMessageBox.Ok)
                del frames_reader
                return
            self.frames_reader = frames_reader
            self.dirname = dirpath
            self.filePath = None
            self.fileListWidget.clear()
            self.mImgList = self.frames_reader.get_frames_list()
            if not self.set_jump_interval():
                self.jump_interval_textbox.setCurrentText(str(len(self.mImgList) / 1000))
                self.set_jump_interval()
            annotation_path = self.get_annotation_path()

            self.openNextImg()
            self.max_id = 0
            self.loadSeqSettings()

            self.fileListWidget.clear()
            for imgPath in self.mImgList:
                item = QListWidgetItem(imgPath)
                self.fileListWidget.addItem(item)
            self.full_title = '{} ({})'.format(self.base_title, os.path.basename(dirpath))
            self.setWindowTitle(self.full_title)

    def edge_detection(self):

        # print('Running edge detection...')

        if self.canvas.selectedShape:
            curr_shape = self.canvas.selectedShape
        else:
            curr_shape = self.canvas.shapes[0]

        try:
            xmin = int(curr_shape.points[0].x())
            ymin = int(curr_shape.points[0].y())
            xmax = int(curr_shape.points[2].x())
            ymax = int(curr_shape.points[2].y())
        except BaseException as e:
            print('Something weird going on: {}'.format(e))
            return

        height, width = self.image_np.shape[:2]

        border_x, border_y = self.params.mask.border_size

        if border_x > 0:
            xmin = max(0, xmin - border_x)
            xmax = min(width - 1, xmax + border_x)

        if border_y > 0:
            ymin = max(0, ymin - border_y)
            ymax = min(height - 1, ymax + border_y)

        shape_patch_orig = self.image_np[ymin:ymax, xmin:xmax, :]
        h, w = shape_patch_orig.shape[:2]
        scale_x, scale_y = self.params.mask.disp_size[0] / w, self.params.mask.disp_size[1] / h
        scale_factor = min(scale_x, scale_y)
        shape_patch = cv2.resize(shape_patch_orig, (0, 0), fx=scale_factor, fy=scale_factor)

        hed_mask = curr_shape.runHED(shape_patch, self.hed_net)
        if hed_mask is not None:
            _, mask_pts = curr_shape.contour_pts_from_mask(hed_mask)
            curr_shape.mask = [(xmin + x / scale_factor, ymin + y / scale_factor, f)
                               for (x, y, f) in mask_pts]

    def generate_mask(self):

        print('Generating mask...')

        start_t = time.time()

        currIndex = self.mImgList.index(self.filePath)
        prevIndex = currIndex - self.jump_interval

        if prevIndex < 0:
            print('No previous frame to get motion info from')
            return

        filename = self.mImgList[prevIndex]
        print('Getting motion info from frame {} :: {}'.format(prevIndex, filename))

        saveDir = self.get_annotation_path()

        filePath = ustr(filename)

        basename = os.path.basename(
            os.path.splitext(filePath)[0]) + XML_EXT
        xmlPath = os.path.join(saveDir, basename)

        if os.path.isfile(xmlPath):
            tVocParseReader = PascalVocReader(xmlPath)
            prev_shapes = tVocParseReader.getShapes()
        else:
            prev_shapes = []

        if len(self.deleted_targets) > 0:
            prevIndex = self.mImgList.index(self.filePath)
            prev_shapes = [shape for shape in prev_shapes if shape[6] not in self.deleted_targets or
                           prevIndex < self.deleted_targets[shape[6]]]

        # id_to_shapes = {shape.id_number: shape for shape in prev_shapes
        #                 if shape.id_number > 0}
        if len(prev_shapes) == 0:
            return

        print('Done reading previous frame shapes in {} secs'.format(time.time() - start_t))

        id_to_shapes = {}
        for label, points, line_color, fill_color, difficult, bbox_source, id_number, score, mask, _ in prev_shapes:
            if id_number <= 0:
                continue
            id_to_shapes[id_number] = (points, mask)

        if self.canvas.selectedShape:
            curr_shapes = [self.canvas.selectedShape, ]
        else:
            curr_shapes = self.canvas.shapes

        for shape in curr_shapes:
            try:
                prev_pts, prev_mask = id_to_shapes[shape.id_number]
            except KeyError:
                continue
            if not prev_mask:
                continue
            curr_pts = [(k.x(), k.y()) for k in shape.points]
            prev_mask_pts = shape.getContourPts(prev_mask)

            if self.params.mask.gen_method == 0:
                prev_pts_norm, _ = normalizePts(prev_pts)
                curr_pts_norm, _ = normalizePts(curr_pts)

                ast_mat = computeASTDLT(prev_pts_norm, curr_pts_norm)
                prev_mask_pts_norm, inv_norm_mat = normalizePts(prev_mask_pts)

                # print('prev_mask_pts: ', prev_mask_pts)

                prev_mask_mat = np.array([[x, y, 1] for x, y in prev_mask_pts_norm]).transpose()

                # print('ast_mat: ', ast_mat)
                # print('prev_mask_mat: ', prev_mask_mat)
                curr_mask = np.matmul(inv_norm_mat, np.matmul(ast_mat, prev_mask_mat)).transpose()

                # print('curr_mask: ', curr_mask)

                # tx, ty = np.mean(curr_pts, axis=0) - np.mean(prev_pts, axis=0)
                #
                # prev_bbox = LabelFile.convertPoints2BndBox(prev_pts, label='points', allow_zero=1, convert_to_int=0)
                # xmin, ymin, xmax, ymax = prev_bbox
                # prev_w, prev_h = xmax - xmin, ymax - ymin
                #
                # curr_bbox = LabelFile.convertPoints2BndBox(curr_pts, label='points', allow_zero=1, convert_to_int=0)
                # xmin, ymin, xmax, ymax = curr_bbox
                # curr_w, curr_h = xmax - xmin, ymax - ymin
                #
                # sx, sy = curr_w / prev_w, curr_h / prev_h                #

                # curr_mask = [(x * sx + tx, y * sy + ty) for x, y in prev_mask]
                shape.mask = [[x, y, 1] for x, y, _ in curr_mask]
            else:
                curr_min_x, curr_min_y = min(k[0] for k in curr_pts), min(k[1] for k in curr_pts)
                prev_min_x, prev_min_y = min(k[0] for k in prev_pts), min(k[1] for k in prev_pts)

                curr_max_x, curr_max_y = max(k[0] for k in curr_pts), max(k[1] for k in curr_pts)
                prev_max_x, prev_max_y = max(k[0] for k in prev_pts), max(k[1] for k in prev_pts)

                tx, ty = curr_min_x - prev_min_x, curr_min_y - prev_min_y
                sx = float(curr_max_x - curr_min_x) / float(prev_max_x - prev_min_x)
                sy = float(curr_max_y - curr_min_y) / float(prev_max_y - prev_min_y)

                shape.mask = [[(x + tx) * sx, (y + ty) * sy, 1] for x, y in prev_mask_pts]
        print('Done generating new masks in {} secs'.format(time.time() - start_t))

    def openPrevImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        # if self.autoSaving.isChecked():
        #     if self.saveDir is not None:
        #         if self.dirty is True:
        #             self.saveFile()
        #     else:
        #         # self.changeSavedir()
        #         return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - self.jump_interval >= 0:
            filename = self.mImgList[currIndex - self.jump_interval]
            currIndex = currIndex - self.jump_interval

            if filename:
                self.curr_index = currIndex
                if self.autoSaving.isChecked():
                    self.saveFile()
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        # if self.autoSaving.isChecked():
        #     if self.saveDir is not None:
        #         if self.dirty is True:
        #             self.saveFile()
        #     else:
        #         # self.changeSavedir()
        #         return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            currIndex = 0
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + self.jump_interval < len(self.mImgList):
                filename = self.mImgList[currIndex + self.jump_interval]
                currIndex = currIndex + self.jump_interval

        if filename:
            self.curr_index = currIndex
            if self.autoSaving.isChecked():
                self.saveFile()
            self.loadFile(filename)

    def loadSeqSettings(self):
        if self.saveDir is None or len(ustr(self.saveDir)) == 0:
            return

        if not os.path.isdir(self.saveDir):
            os.makedirs(self.saveDir)

        settings_saved_path = os.path.join(self.saveDir, "labelImgSettings.pkl")
        self.seq_settings = Settings(settings_saved_path)
        self.seq_settings.load()

        deleted_targets = self.seq_settings.get(SETTING_DELETED_TARGETS)
        if deleted_targets is not None:
            self.deleted_targets = deleted_targets
        else:
            self.deleted_targets = {}

        max_id = self.seq_settings.get(SETTING_MAX_ID)
        if max_id is not None:
            self.max_id = max_id

    def saveSeqSettings(self):
        if self.seq_settings is None:
            return

        settings = self.seq_settings

        settings[SETTING_DELETED_TARGETS] = self.deleted_targets
        settings[SETTING_MAX_ID] = self.max_id
        settings.save()

    def openVideoFile(self, filename=None):
        if not self.mayContinue():
            print('mayContinue')
            return

        if self.autoSaving.isChecked():
            print('saveFile')
            self.saveFile()

        if self.frames_reader is not None:
            self.frames_reader.close(self.delete_bin)

        if self.lastOpenVideoFile is not None:
            path = self.lastOpenVideoFile
        else:
            path = self.frames_reader.get_path() \
                if self.frames_reader and os.path.isfile(self.frames_reader.get_path()) \
                else os.path.expanduser("~")

        formats = None
        filters = "Video Files (%s)" % ' '.join('*')
        if filename is None or not filename:
            filename = QFileDialog.getOpenFileName(self, '%s - Choose video file' % self.base_title, path, filters)
        if not filename or len(filename) == 0:
            print('Invalid filename: {}'.format(filename))
            return

        if isinstance(filename, (tuple, list)):
            filename = filename[0]

        if not filename:
            return

        self.saveSeqSettings()

        frames_reader = VideoReader(filename, save_as_bin=0)
        if frames_reader.num_frames == 0:
            QMessageBox.critical(self, "Error", "No frames detected! Please choose different path.", QMessageBox.Ok)
            del frames_reader
            return

        if self.lastOpenVideoFile is None or filename != self.lastOpenVideoFile:
            print('Disabling ROI')
            self.roi = None

        if not frames_reader.setROI(self.roi):
            self.roi = None

        self.lastOpenVideoFile = filename
        self.frames_reader = frames_reader
        self.mImgList = self.frames_reader.get_frames_list()
        if not self.set_jump_interval():
            self.jump_interval_textbox.setCurrentText(str(len(self.mImgList) / 1000))
            self.set_jump_interval()
        annotation_path = self.get_annotation_path()

        self.max_id = 0

        self.loadFile(self.mImgList[0])
        self.loadSeqSettings()

        self.fileListWidget.clear()
        self.full_title = '{} ({})'.format(self.base_title, os.path.basename(filename))
        self.setWindowTitle(self.full_title)
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)

    def saveFile(self, _value=False):
        if self.filePath is None:
            return
        if self.saveDir is not None and len(ustr(self.saveDir)):
            if not os.path.isdir(self.saveDir):
                print('Annotations save dir does not exist: {}'.format(self.saveDir))
                return
            if not self.filePath:
                # print('Invalid filePath')
                return
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0] + XML_EXT
            savedPath = os.path.join(ustr(self.saveDir), savedFileName)
            self._saveFile(savedPath)
            settings_saved_path = os.path.join(ustr(self.saveDir), "labelImgSettings.pkl")
        else:
            imgFileDir = os.path.dirname(self.filePath)
            if not os.path.isdir(imgFileDir):
                print('Annotations save dir does not exist: {}'.format(imgFileDir))
                return
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0] + XML_EXT
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = '%s - Choose File' % self.base_title
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            return dlg.selectedFiles()[0]
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def mayContinue(self):
        # return not (self.dirty and not self.discardChangesDialog())
        return True

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR_GROUND_TRUTH)
        if color:
            self.lineColor = color
            # Change the color for all shape lines:
            Shape.line_color = self.lineColor
            self.canvas.update()
            self.setDirty()

    def chooseColor2(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.fillColor = color
            Shape.fill_color = self.fillColor
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def deleteSelectedTarget(self):
        curr_shape = self.canvas.selectedShape
        try:
            _id = int(curr_shape.id_number)
            curr_index = self.mImgList.index(self.filePath)
            self.deleted_targets[curr_shape.id_number] = curr_index
            print('Target {:d} marked for deletion from frame {:d}'.format(
                _id, curr_index))
            self.deleteSelectedShape()
        except ValueError:
            print('Only targets with numeric IDs can be marked for deletion')

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.lablHist = [line]
                    else:
                        self.labelHist.append(line)

    def loadPascalXMLByFilename(self, xmlPath, curr_index):
        if self.filePath is not None and os.path.isfile(xmlPath):
            tVocParseReader = PascalVocReader(xmlPath)
            shapes = tVocParseReader.getShapes()
            if self.params.max_boxes < 0:
                shapes = shapes[self.params.max_boxes:]
            elif self.params.max_boxes > 0:
                shapes = shapes[:self.params.max_boxes]
        else:
            shapes = []

        if len(self.deleted_targets) > 0:
            shapes = [shape for shape in shapes if shape[6] not in self.deleted_targets or
                      curr_index < self.deleted_targets[shape[6]]]

        if len(shapes) == 0:
            return

        # print('Read {:d} shapes from xml {:s}'.format(
        #     len(shapes), xmlPath))
        self.loadLabels(shapes)

        self.canvas.verified = tVocParseReader.verified

    def readImage(self, image_name):
        if self.frames_reader is None:
            raise Exception("frame reader is None")
        else:
            img_array = self.frames_reader.get_frame_by_name(image_name)
            height, width, channel = img_array.shape
            self.image_np = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            bytesPerLine = 3 * width
            qImg = QImage(img_array.data, width, height, bytesPerLine, QImage.Format_RGB888)
            return qImg

    def clear_saved_settings(self):
        path = os.path.expanduser('./.labelImgSettings.pkl')
        if os.path.isfile(path):
            os.remove(path)

    def bring_window_to_front(self, window):
        window.setWindowState(window.windowState() & Qt.WindowMinimized | Qt.WindowActive)
        window.show()
        window.activateWindow()

    def save_map(self):
        # msg = QMessageBox()
        # msg.setIcon(QMessageBox.Information)
        # msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        # msg.setText("Press Save to start saving currently loaded boxes as detections and annotations in mAP format")
        # # msg.setInformativeText("This is additional information")
        # msg.setWindowTitle("Save as mAP")
        # msg.setDetailedText(
        #     "Both manually annotated and tracked boxes will be saved in a subfolder called mAP within the"
        #     " sequence directory."
        #     "Refer to the terminal output for detailed messages and errors from the saving process.")
        # retval = msg.exec_()
        # if retval == QMessageBox.Save:

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Please wait")

        msg.open()
        out_dir = self.saveBoxesTXT(0)
        msg.close()

        if out_dir is not None:
            print('Done saving boxes in mAP format')

    def save_yolo(self):
        # msg = QMessageBox()
        # msg.setIcon(QMessageBox.Information)
        # msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        # msg.setText("Press Save to start saving currently loaded boxes as detections and annotations in YOLO format")
        # # msg.setInformativeText("This is additional information")
        # msg.setWindowTitle("Save as YOLO")
        # msg.setDetailedText(
        #     "Both manually annotated and tracked boxes will be saved in a subfolder called YOLO within the"
        #     " sequence directory."
        #     "Refer to the terminal output for detailed messages and errors from the saving process.")
        # retval = msg.exec_()
        # if retval == QMessageBox.Save:

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Please wait")

        msg.open()
        out_dir = self.saveBoxesTXT(1)
        msg.close()

        if out_dir is not None:
            print('Done saving boxes in YOLO format')

    def save_csv(self):
        # msg = QMessageBox()
        # msg.setIcon(QMessageBox.Information)
        # msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        # msg.setText("Press Save to start saving currently loaded boxes as detections and annotations in CSV format")
        # # msg.setInformativeText("This is additional information")
        # msg.setWindowTitle("Save as CSV")
        # msg.setDetailedText(
        #     "Both manually annotated and tracked boxes will be saved in the sequence directory"
        #     " as a file called annotations.csv"
        #     "Refer to the terminal output for detailed messages and errors from the saving process.")
        # retval = msg.exec_()
        # if retval == QMessageBox.Save:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Please wait")

        msg.open()
        out_file_path = self.saveBoxesCSV()
        msg.close()

        if out_file_path is not None:
            print('Done saving boxes in csv format')

    def tracking_settings_dialog(self):
        window = self.other_windows.tracking_settings_dialog
        self.bring_window_to_front(window)

    def roi_button(self, roi_text, roi_dialog_box):
        self.roi_id = roi_text.text()
        if not self.roi_id:
            print('Must enter a valid ID')
            return
        print('Setting ROI ID to {:s}'.format(self.roi_id))
        roi_dialog_box.close()

    def createMask(self):
        # if self.roi is not None:
        #     print('ROI can only be set once per session')
        #     return
        self.saveSeqSettings()

        curr_shape = self.canvas.selectedShape
        curr_shape.addMask(self.image_np,
                           self.params.mask,
                           self.augment,
                           self.hed_net
                           )

        self.canvas.repaint()

    def saveROI(self):

        if self.roi is None:
            self.setROI()

        roi_save_dir = 'roi_{}_{}_{}_{}'.format(self.roi['xmin'], self.roi['ymin'],
                                                self.roi['xmax'], self.roi['ymax'])

        if isinstance(self.frames_reader, DirectoryReader):
            dir_path = self.dirname
            root_dir = os.path.dirname(dir_path)
            dir_name = os.path.splitext(os.path.basename(dir_path))[0]
            roi_save_path = os.path.join(root_dir, dir_name + '_' + roi_save_dir)
        elif isinstance(self.frames_reader, VideoReader):
            video_path = self.frames_reader.video_fn
            video_fname = os.path.splitext(os.path.basename(video_path))[0]
            video_dir = os.path.dirname(video_path)
            roi_save_path = os.path.join(video_dir, video_fname + '_' + roi_save_dir)
        n_frames = self.frames_reader.num_frames

        print('Saving ROI sequence with {} frames to {}'.format(n_frames, roi_save_path))

        if not os.path.isdir(roi_save_path):
            os.makedirs(roi_save_path)

        _pause = 0
        for i in range(n_frames):
            fname = 'frame{:06d}.png'.format(i + 1)
            roi_img = self.frames_reader.get_frame(i)
            out_fname = os.path.join(roi_save_path, fname)
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
            cv2.imshow(roi_save_dir, roi_img)
            k = cv2.waitKey(1 - _pause)
            if k == 32:
                _pause = 1 - _pause
            elif k == 27:
                break

            cv2.imwrite(out_fname, roi_img)

        cv2.destroyWindow(roi_save_dir)

    def setROI(self):

        self.saveSeqSettings()

        curr_shape = self.canvas.selectedShape
        try:
            xmin = int(curr_shape.points[0].x())
            ymin = int(curr_shape.points[0].y())
            xmax = int(curr_shape.points[2].x())
            ymax = int(curr_shape.points[2].y())
        except:
            return

        copy_existing_annotations = False
        rel_min_point = None
        if self.roi is not None:
            rel_min_point = [xmin, ymin]
            xmin += self.roi['xmin']
            xmax += self.roi['xmin']
            ymin += self.roi['ymin']
            ymax += self.roi['ymin']
            copy_existing_annotations = True

        self.roi = dict(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

        if copy_existing_annotations:
            new_save_dir = self.get_annotation_path()
            print('Copying all existing annotations to {:s}'.format(new_save_dir))
            files = glob.iglob(os.path.join(self.saveDir, "*.xml"))
            if isinstance(self.frames_reader, DirectoryReader):
                folder_name = os.path.dirname(self.filePath)
            else:
                folder_name = self.frames_reader.video_fn
            height = self.roi['ymax'] - self.roi['ymin']
            width = self.roi['xmax'] - self.roi['xmin']
            channel = 1 if self.imageData.isGrayscale() else 3
            imageShape = [height, width, channel]
            frame_box = np.array((0, 0, width, height)).reshape((1, 4))
            exit_threshold = 0.1
            file_id = 1
            for file in files:
                if os.path.isfile(file):
                    reader = PascalVocReader(file)
                    shapes = reader.getShapes()
                    filename = reader.filename
                    writer = PascalVocWriter(folder_name, filename, imageShape)
                    for label, points, line_color, fill_color, difficult, bbox_source, id_number, score, mask, \
                        mask_img in shapes:
                        # box location w.r.t. the new ROI
                        xmin = points[0][0] - rel_min_point[0]
                        xmax = points[1][0] - rel_min_point[0]
                        ymin = points[0][1] - rel_min_point[1]
                        ymax = points[2][1] - rel_min_point[1]
                        if (xmax < 0 and ymax < 0) or (xmin >= width and ymin >= height):
                            # box is entirely out of the scene
                            continue
                        ioa_1 = np.empty((1,))
                        computeOverlap(None, ioa_1, None,
                                       np.array((xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)),
                                       frame_box)
                        if ioa_1 < exit_threshold:
                            continue
                        shifted_points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                        bndbox = LabelFile.convertPoints2BndBox(shifted_points, label)
                        writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3],
                                         label, difficult, bbox_source, id_number, score, mask, mask_img)
                    filename = os.path.basename(file)
                    dst_file_path = os.path.join(new_save_dir, filename)
                    writer.save(targetFile=dst_file_path)
                    sys.stdout.write('\rDone {:d} files'.format(file_id))
                    sys.stdout.flush()
                    file_id += 1
                    # shutil.copy2(file, new_save_dir)
            sys.stdout.write('\n')
            sys.stdout.flush()

        if not self.frames_reader.setROI(self.roi):
            self.roi = None
        self.loadFile(self.filePath)
        self.max_id = 0
        self.loadSeqSettings()

        try:
            self.deleteSelectedShape()
        except ValueError:
            pass

    def loadROI(self):
        annotations_path = ustr(QFileDialog.getExistingDirectory(
            self, 'Choose ROI annotations directory', self.saveDir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))

        if annotations_path is None or len(annotations_path) <= 1:
            return
        roi = self.getROIfromAnnotationsPath(annotations_path)
        if roi is None:
            print('Invalid folder chosen')
        else:
            self.saveSeqSettings()

            self.roi = roi
            print('Set ROI to: ', self.roi)
            self.frames_reader.setROI(self.roi)
            self.loadFile(self.filePath)
            self.max_id = 0
            self.loadSeqSettings()

    def getROIfromAnnotationsPath(self, annotations_path):
        dir_name = os.path.basename(annotations_path)
        roi_extents = [int(s) for s in dir_name.split('_') if s.isdigit()]
        if len(roi_extents) != 4:
            print('annotations_path: ', annotations_path)
            print('dir_name: ', dir_name)
            print('roi_extents: ', roi_extents)
            return None
        xmin, ymin, xmax, ymax = roi_extents
        return dict(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

    def removeROI(self):
        if self.roi is None:
            print('No ROI to remove')
            return

        self.saveSeqSettings()

        self.roi = None
        self.frames_reader.setROI(self.roi)
        self.loadFile(self.filePath)
        self.max_id = 0
        self.loadSeqSettings()

    def trackButton(self, track_text, track_dialog_box):
        try:
            self.track_id = int(track_text.text())
        except ValueError:
            print('Must enter a valid numeric ID')
            return
        print('Setting target ID to {:d}'.format(self.track_id))
        track_dialog_box.close()

    def trackSelectedObject(self):
        curr_shape = self.canvas.selectedShape
        curr_index = self.mImgList.index(self.filePath)
        try:
            xmin = curr_shape.points[0].x()
            ymin = curr_shape.points[0].y()
            xmax = curr_shape.points[2].x()
            ymax = curr_shape.points[2].y()
        except:
            # shape is not a bbox
            return
        bbox = dict(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
        if curr_shape.id_number is not None and curr_shape.id_number >= 0:
            # track_text.setText(str(curr_shape.id_number))
            self.track_id = int(curr_shape.id_number)
        else:
            track_dialog_box = QDialog()
            track_dialog_box.resize(250, 100)
            track_text = QLineEdit(track_dialog_box)
            track_text.setText(str(self.max_id + 1))
            track_text.move(50, 10)
            track_button = QPushButton("OK", track_dialog_box)
            track_button.move(75, 50)
            track_button.clicked.connect(partial(self.trackButton, track_text, track_dialog_box))
            track_dialog_box.setWindowTitle("Enter a unique target ID")
            track_dialog_box.setWindowModality(Qt.ApplicationModal)
            track_dialog_box.exec_()

        curr_shape.id_number = self.track_id

        if self.track_id > self.max_id:
            self.max_id = self.track_id

        # if not self.other_windows.tracking_server_log.isVisible():
        #     self.view_tracking_server_log()

        self.send_patch_tracking_request(
            curr_index, bbox, curr_shape.label, self.track_id, 'ground_truth')

    def connectToPort(self, sock, port, name):
        print_msg = True
        while True:
            try:
                sock.connect(('localhost', port))
                break
            except ConnectionRefusedError:
                time.sleep(1)
                if print_msg:
                    sys.stdout.write('Waiting to connect to the {:s} port'.format(name))
                    sys.stdout.flush()
                    print_msg = False
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                continue
                # cmd = 'lsof -t -i tcp:{:d} | xargs kill -9'.format(self.params.port)
                # pid = subprocess.check_output(cmd, shell=True)
                # print('Process {} using port'.format(pid))
            except KeyboardInterrupt:
                return False
        if not print_msg:
            sys.stdout.write('Done\n')
            sys.stdout.flush()
        return True

    def send_stop_request(self):

        TRACKING_PORT = 3002
        requests = [
            dict(
                request_type="stop",
                port=3000,
            )
        ]
        # print('Sending request for frame {:d} with {:d} boxes'.format(
        #     frame_number, len(bboxes)))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if not self.connectToPort(sock, TRACKING_PORT, 'tracking'):
            return
        # try:
        #     sock.connect(('localhost', TRACKING_PORT))
        # except ConnectionRefusedError:
        #     return
        send_msg_to_connection(requests, sock)
        sock.close()

    def send_patch_tracking_request(self, frame_number, bbox, label,
                                    id_number, bbox_source):

        TRACKING_PORT = 3002
        # self.other_windows.tracking_server_log.start_server()
        # cmd_args = self.other_windows.tracking_server_log.getCommandLineArgs()
        num_frames = self.frames_reader.num_frames
        requests = [
            dict(
                request_type="patch_tracking",
                cmd_args='',
                path=self.frames_reader.get_path(),
                frame_number=frame_number,
                port=3000,
                trigger_tracking_request=False,
                bbox=bbox,
                label=label,
                id_number=id_number,
                roi=self.roi,
                bbox_source=bbox_source,
                num_frames=num_frames,
            )
        ]
        # print('Sending request for frame {:d} with {:d} boxes'.format(
        #     frame_number, len(bboxes)))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if not self.connectToPort(sock, TRACKING_PORT, 'tracking'):
            return
        # try:
        #     sock.connect(('localhost', TRACKING_PORT))
        # except ConnectionRefusedError:
        #     return
        send_msg_to_connection(requests, sock)
        sock.close()

    # def visualize(self):
    #
    #     msg = QMessageBox()
    #     msg.setIcon(QMessageBox.Information)
    #     msg.setWindowTitle("Please wait")
    #
    #     msg.open()
    #     out_file_path = self.saveBoxesCSV(verbose=False)
    #     msg.close()
    #
    #     if out_file_path is None:
    #         return
    #
    #     frame_number = self.mImgList.index(self.filePath)
    #     class_dict = dict((v, k) for k, v in enumerate(self.labelHist))
    #
    #     TRACKING_PORT = 3002
    #     # self.other_windows.tracking_server_log.start_server()
    #     # cmd_args = self.other_windows.tracking_server_log.getCommandLineArgs()
    #     num_frames = self.frames_reader.num_frames
    #     requests = [
    #         dict(
    #             request_type="visualize",
    #             cmd_args='',
    #             path=self.frames_reader.get_path(),
    #             csv_path=out_file_path,
    #             class_dict=class_dict,
    #             frame_number=frame_number,
    #             port=3000,
    #             trigger_tracking_request=False,
    #             roi=self.roi,
    #             num_frames=num_frames,
    #         )
    #     ]
    #     # print('Sending request for frame {:d} with {:d} boxes'.format(
    #     #     frame_number, len(bboxes)))
    #     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     self.connectToPort(sock, TRACKING_PORT, 'tracking')
    #     # try:
    #     #     sock.connect(('localhost', TRACKING_PORT))
    #     # except ConnectionRefusedError:
    #     #     return
    #     send_msg_to_connection(requests, sock)
    #     sock.close()

    def getBkgFile(self, use_prev=0):
        if self.lastBkgDir is not None and len(self.lastBkgDir) > 1:
            if use_prev:
                return True
            path = os.path.dirname(self.lastBkgDir)
        else:
            path = os.path.dirname(self.frames_reader.get_path()) if self.frames_reader else os.path.expanduser("~")

        formats = None
        filters = "Images (*.png *.xpm *.bmp *.jpg *.jpeg)"
        # filters = "Image Files (%s)" % ' '.join('*')
        bkg_filename = QFileDialog.getOpenFileName(
            self, '%s - Choose background image or folder with multiple images' % self.base_title, path, filters)
        if bkg_filename and len(bkg_filename) > 0:
            if isinstance(bkg_filename, (tuple, list)):
                bkg_filename = bkg_filename[0]
        if not bkg_filename:
            return False
        self.lastBkgDir = bkg_filename
        return True

    def augment(self, request, mask=None, use_prev=0, save_seq=1):

        curr_shape = self.canvas.selectedShape
        if curr_shape is None or not curr_shape:
            curr_shape = self.canvas.shapes[0]

        if mask is None:
            if not curr_shape.mask:
                print('No mask found')
                return
            mask = curr_shape.mask

        if mask is None or not mask:
            print('No mask found')
            return

        if not self.getBkgFile(use_prev):
            return

        bkg_filename = self.lastBkgDir

        bkg_data_dict, bkg_imgs, bkg_files = readBackgroundData([bkg_filename, ])

        src_file = self.frames_reader.get_file_path()
        # src_img = self.frames_reader.get_frame_by_name(src_file)
        src_img = cv2.imread(src_file)

        xmin = int(curr_shape.points[0].x())
        ymin = int(curr_shape.points[0].y())
        xmax = int(curr_shape.points[2].x())
        ymax = int(curr_shape.points[2].y())

        # print('xmin: {}'.format(xmin))
        # print('ymin: {}'.format(ymin))

        patch_img = src_img[ymin:ymax, xmin:xmax, :]
        _contour_pts = Shape.getContourPts(mask, verbose=0)

        _contour_pts = [[(x - xmin), (y - ymin)] for x, y in _contour_pts]

        # print('curr_shape.mask: ', curr_shape.mask)
        # print('_contour_pts: ', _contour_pts)

        # col = 1
        col = 255
        mask_img, _ = Shape.contour_pts_to_mask(_contour_pts, patch_img, col=(col, col, col))

        # cv2.imwrite('temp_mask_img.png', mask_img)
        # mask_img = cv2.imread('temp_mask_img.png')
        # mask_img = mask_img.astype(np.float32) / 255.0

        # cv2.imshow('augment :: src_img', src_img)

        src_data_dict = {
            src_file: {
                'img': src_img,
                'data': [],
            }}

        src_data_dict[src_file]['data'].append(
            {"class_name": curr_shape.label, "bbox": [xmin, ymin, xmax, ymax],
             'mask': mask_img, 'target_id': curr_shape.id_number,
             'seq_id': 0})

        generateAugmentedSequence([src_file, ], src_data_dict, bkg_files, bkg_data_dict, bkg_imgs,
                                  visualize=1, static_bkg=1, boxes_per_bkg=1, random_bkg_box=0,
                                  save_seq=save_seq, show_bbox=0, map_to_bbox=1, save_path='log')

    def visualize(self, request):

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Please wait")

        # msg.open()
        # df = self.saveBoxesCSV(verbose=False, write=False)
        # msg.close()
        #
        # if df is None:
        #     return

        frame_number = self.mImgList.index(self.filePath)
        class_dict = dict((v, k) for k, v in enumerate(self.labelHist))

        num_frames = self.frames_reader.num_frames

        request_path = self.frames_reader.get_path()
        request_roi = self.roi
        init_frame_id = frame_number

        save_fname_templ = os.path.splitext(os.path.basename(request_path))[0]

        class_labels = dict((v, k) for k, v in class_dict.items())

        # print('self.params.visualizer.save: ', self.params.visualizer.save)
        visualizer = Visualizer(self.params.visualizer, self.logger, class_labels)
        init_frame = self.frames_reader.get_frame(init_frame_id, convert_to_rgb=0)

        height, width, _ = init_frame.shape
        frame_size = width, height
        visualizer.initialize(save_fname_templ, frame_size)

        n_frames = self.frames_reader.num_frames
        for frame_id in range(init_frame_id, n_frames):
            try:
                curr_frame = self.frames_reader.get_frame(frame_id, convert_to_rgb=0)
            except IOError as e:
                print('{}'.format(e))
                break

            file_path = self.frames_reader.get_file_path()
            if file_path is None:
                print('Visualization is only supported on image sequence data')
                return

            filename = os.path.basename(file_path)

            xml_path = os.path.join(self.saveDir, os.path.splitext(filename)[0] + '.xml')
            if not os.path.isfile(xml_path):
                print('{} :: annotations xml file not found: {}'.format(filename, xml_path))
                continue

            xml_reader = PascalVocReader(xml_path)
            shapes = xml_reader.getShapes()
            frame_data = []
            masks = []

            for shape in shapes:
                label, points, _, _, difficult, bbox_source, id_number, score, mask, _ = shape

                if id_number is None:
                    id_number = -1

                xmin, ymin = points[0]
                xmax, ymax = points[2]

                target_id = id_number
                class_id = class_dict[label]

                width = xmax - xmin
                height = ymax - ymin

                frame_data.append([frame_id, target_id, xmin, ymin, width, height, class_id])
                if mask is not None:
                    _contour_pts = Shape.getContourPts(mask)
                    masks.append(_contour_pts)

            frame_data = np.asarray(frame_data)
            if not visualizer.update(frame_id, curr_frame, frame_data, masks):
                break

        visualizer.close()

    # def view_tracking_server_log(self):
    #     window = self.other_windows.tracking_server_log
    #     window.start_server()
    #     self.bring_window_to_front(window)

    def show_ground_truth_bboxes(self):
        self.show_ground_truth_bboxes_bool = not self.show_ground_truth_bboxes_bool
        self.hide_ground_truth_bboxes()
        self.canvas.repaint()

    def hide_ground_truth_bboxes(self, shapes=None):
        if shapes is None:
            shapes = self.canvas.shapes
        for shape in shapes:
            if shape.bbox_source == "ground_truth":
                shape.set_hidden(not self.show_ground_truth_bboxes_bool)
                self.shapesToItems[shape].setHidden(not self.show_ground_truth_bboxes_bool)

    def show_tracking_bboxes(self):
        self.show_tracker_bboxes_bool = not self.show_tracker_bboxes_bool
        self.hide_tracking_bboxes()
        self.canvas.repaint()

    def hide_tracking_bboxes(self, shapes=None):
        if shapes is None:
            shapes = self.canvas.shapes
        for shape in shapes:
            if shape.bbox_source == "single_object_tracker":
                shape.set_hidden(not self.show_tracker_bboxes_bool)
                self.shapesToItems[shape].setHidden(not self.show_tracker_bboxes_bool)

    def hide_all_bboxes_sources(self, shapes=None):
        self.hide_ground_truth_bboxes(shapes)
        self.hide_tracking_bboxes(shapes)

    def toggle_show_text(self):
        self.canvas.set_show_text(self.actions.toggle_show_text.isChecked())
        self.canvas.repaint()

    def toggle_delete_bin(self):
        self.delete_bin = self.actions.toggle_delete_bin.isChecked()

    def reload_current_image(self):
        currIndex = self.mImgList.index(self.filePath)
        filename = self.mImgList[currIndex]
        self.loadFile(filename)

    def update_current_image(self):
        self.reload_current_image()

    def set_as_ground_truth(self):
        if self.canvas.selectedShape:
            self.canvas.selectedShape.set_as_ground_truth()
            self.setDirty()

    def default_label_checkbox(self):
        self.defaultLabelComboBox.setEnabled(self.useDefaultLabelCheckbox.isChecked())

    def set_default_label(self):
        self.default_label = self.defaultLabelComboBox.currentText()

    def set_jump_interval(self):
        try:
            new_jump_interval = float(self.jump_interval_textbox.currentText())
        except BaseException as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("set_jump_interval failed: {}".format(str(e)))
            print(exc_type, fname, exc_tb.tb_lineno)
            self.jump_interval_textbox.setCurrentText(str(self.jump_interval))
            return False
        else:
            new_jump_interval = int(new_jump_interval)
            new_jump_interval = max(1, new_jump_interval)
            self.jump_interval = new_jump_interval
            self.jump_interval_textbox.setCurrentText(str(new_jump_interval))
            # if self.filePath is not None:
            #     currIndex = self.mImgList.index(self.filePath)
            #     start_index = currIndex % new_jump_interval
            return True

    def go_to_button_clicked(self):
        currIndex = self.mImgList.index(self.filePath)
        self.curr_index = currIndex
        try:
            img_number = int(self.go_to_textbox.text())
            if 0 <= img_number < self.mImgList.__len__():
                if self.autoSaving.isChecked():
                    self.saveFile()
                filename = self.mImgList[img_number]
                self.loadFile(filename)
                if self.filePath is not None:
                    currIndex = self.mImgList.index(self.filePath)
            else:
                self.go_to_textbox.setText(str(currIndex))
            self.curr_index = currIndex
        except:
            pass

    '''
    SERVER CODE
    '''

    def start_server_thread(self):
        def server_thread():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            bindToPort(sock, self.port, 'server')
            sock.listen(1)
            while True:
                try:
                    connection, addr = sock.accept()
                    connection.settimeout(None)
                    threading.Thread(target=self.run_dialog.handle_requests,
                                     args=[connection]).start()
                except (KeyboardInterrupt, SystemExit):
                    return

        threading.Thread(target=server_thread).start()

    def get_annotation_path(self, frame_number=None):
        xml_dir_name = 'annotations'
        if self.roi is not None:
            xml_dir_name = '{:s}_roi_{:d}_{:d}_{:d}_{:d}'.format(xml_dir_name,
                                                                 self.roi['xmin'], self.roi['ymin'],
                                                                 self.roi['xmax'], self.roi['ymax'])
        # Label xml file and show bound box according to its filename
        if isinstance(self.frames_reader, DirectoryReader):
            saveDir = os.path.join(self.dirname, xml_dir_name)
        else:

            dirname = os.path.splitext(os.path.basename(self.frames_reader.video_fn))[0]
            saveDir = os.path.join(os.path.dirname(self.frames_reader.video_fn), dirname, xml_dir_name)

        if not os.path.exists(saveDir):
            os.mkdir(saveDir)

        if frame_number is None:
            return saveDir

        if isinstance(self.frames_reader, DirectoryReader):
            file_path = self.mImgList[frame_number]
        else:
            file_path = "{}.xml".format(frame_number)
            # file_path =  "image{:06d}.xml".format(frame_number + 1)

        basename = os.path.basename(
            os.path.splitext(file_path)[0]) + XML_EXT
        xmlPath = os.path.join(saveDir, basename)
        return xmlPath

    def delete_all_in_current_image(self):
        reply = QMessageBox.warning(self, "Confirm", "Are you sure to delete all annotations for the current frames?"
                                                     " This cannot be undone", QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.No:
            return
        currIndex = self.mImgList.index(self.filePath)
        annotation_path = self.get_annotation_path(currIndex)
        os.remove(annotation_path)
        self.reload_current_image()

    def delete_all_annotations(self):
        reply = QMessageBox.warning(self, "Confirm", "Are you sure to delete annotations for ALL FRAMES?"
                                                     " This cannot be undone", QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.No:
            return
        annotation_path = self.get_annotation_path(0)
        annotation_dir = os.path.dirname(annotation_path)
        files = os.listdir(annotation_dir)
        for file in files:
            if file.endswith(".xml"):
                os.remove(os.path.join(annotation_dir, file))
        self.reload_current_image()

    def saveBoxesTXT(self, _type):

        if _type == 0:
            _type_str = 'mAP'
        else:
            _type_str = 'yolo'

        if not self.saveDir or not os.path.isdir(self.saveDir):
            print('Folder containing the loaded boxes does not exist')
            return None

        files = glob.glob(os.path.join(self.saveDir, '*.xml'))
        n_files = len(files)
        if n_files == 0:
            print('No loaded boxes found')
            return None

        def convert_to_yolo(size, box):
            dw = 1. / size[0]
            dh = 1. / size[1]
            x = (box[0] + box[1]) / 2.0
            y = (box[2] + box[3]) / 2.0
            w = box[1] - box[0]
            h = box[3] - box[2]
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            return (x, y, w, h)

        def getint(fn):
            basename = os.path.basename(fn)
            num = re.sub("\D", "", basename)
            try:
                return int(num)
            except:
                return 0

        if len(files) > 0:
            files = sorted(files, key=getint)

        out_dir = os.path.join(os.path.dirname(self.saveDir), _type_str)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        list_file = None
        if _type == 1:
            list_path = os.path.join(out_dir, 'list.txt')
            list_file = open(list_path, 'w')

        print('Loading annotations from {:d} files...'.format(n_files))
        print('Writing {} annotations to {:s}...'.format(_type_str, out_dir))

        file_id = 0
        n_boxes = 0
        for file in files:
            file_no_ext = os.path.splitext(os.path.basename(file))[0]

            out_file_path = os.path.join(out_dir, '{}.txt'.format(file_no_ext))
            out_file = open(out_file_path, 'w')

            xml_reader = PascalVocReader(file)
            shapes = xml_reader.getShapes()
            for shape in shapes:
                label, points, _, _, difficult, bbox_source, id_number, score, _, _ = shape

                xmin, ymin = points[0]
                xmax, ymax = points[2]

                if _type == 0:
                    out_file.write('{:s} {:d} {:d} {:d} {:d}\n'.format(label, xmin, ymin, xmax, ymax))
                else:
                    class_id = self.labelHist.index(label) + 1
                    bb = convert_to_yolo((xml_reader.width, xml_reader.height), [xmin, xmax, ymin, ymax])
                    out_file.write('{:d} {:f} {:f} {:f} {:f}\n'.format(class_id, bb[0], bb[1], bb[2], bb[3]))
                if _type == 1:
                    list_file.write('{:s}\n'.format(xml_reader.filename))
                n_boxes += 1

            file_id += 1
            sys.stdout.write('\rDone {:d}/{:d} files with {:d} boxes'.format(
                file_id, n_files, n_boxes))
            sys.stdout.flush()

            out_file.close()
        if _type == 1:
            list_file.close()

        sys.stdout.write('\n')
        sys.stdout.flush()

        return out_dir

    def saveMasksButton(self, w_text, h_text, b_text, mask_dialog_box, button_pushed):
        try:
            w = int(w_text.text())
            h = int(h_text.text())
        except ValueError:
            print('Both width and height must be integers')
            return

        button_pushed[0] = 1

        print('Setting output size to {:d} x {:d}'.format(w, h))
        mask_dialog_box.close()

        self.out_mask_size = (w, h)

        b_text = b_text.text()
        if '.' in b_text:
            self.out_mask_border[0] = float(b_text)
        else:
            self.out_mask_border[0] = int(b_text)

    def saveMasksDialogue(self):
        mask_dialog_box = QDialog()

        pos = dotdict()
        size = dotdict()

        pos.w_label = 20, 30
        size.w_label = 80
        w_label = QLabel(mask_dialog_box)
        w_label.setText('width/min_dim ')
        w_label.move(*pos.w_label)

        pos.w_text = pos.w_label[0] + size.w_label, 25
        size.w_text = 40, 20
        w_text = QLineEdit(mask_dialog_box)
        w_text.setText(str(self.out_mask_size[0]))
        w_text.move(*pos.w_text)
        w_text.resize(*size.w_text)

        pos.h_label = pos.w_text[0] + size.w_text[0] + 20, 30
        size.h_label = 80
        h_label = QLabel(mask_dialog_box)
        h_label.setText('height/max_dim ')
        h_label.move(*pos.h_label)

        pos.h_text = pos.h_label[0] + size.h_label, 25
        size.h_text = 40, 20
        h_text = QLineEdit(mask_dialog_box)
        h_text.setText(str(self.out_mask_size[1]))
        h_text.move(*pos.h_text)
        h_text.resize(*size.h_text)

        pos.b_label = pos.h_text[0] + size.h_text[0] + 20, 30
        size.b_label = 50
        b_label = QLabel(mask_dialog_box)
        b_label.setText('border ')
        b_label.move(*pos.b_label)

        pos.b_text = pos.b_label[0] + size.b_label, 25
        size.b_text = 40, 20
        b_text = QLineEdit(mask_dialog_box)
        b_text.setText(str(self.out_mask_border[0]))
        b_text.move(*pos.b_text)
        b_text.resize(*size.b_text)

        mask_button = QPushButton("OK", mask_dialog_box)
        mask_button.move(pos.b_text[0] / 2, 65)

        button_pushed = [0, ]
        mask_button.clicked.connect(
            partial(self.saveMasksButton, w_text, h_text, b_text, mask_dialog_box, button_pushed))

        size.mask_dialog_box = pos.b_text[0] + size.b_text[0] + 20, 100

        mask_dialog_box.resize(*size.mask_dialog_box)
        mask_dialog_box.setWhatsThis('both 0 or -1 to disable resizing / '
                                     'one 0 to infer it from the other using the original aspect ratio / '
                                     'one -1 to use the other as maximum or minimum dimension / '
                                     'floating point value for border specifies a fixed aspect ratio that will be '
                                     'achieved by adding the appropriate horizontal or vertical border'
                                     )

        mask_dialog_box.setWindowTitle("Mask Output Size")
        mask_dialog_box.setWindowModality(Qt.ApplicationModal)
        mask_dialog_box.exec_()

        return button_pushed[0]

    def save_masks(self):
        if not self.saveMasksDialogue():
            return

        out_border = self.out_mask_border[0]
        fixed_ar = 0
        if isinstance(out_border, float):
            fixed_ar = out_border

        save_masks(os.path.dirname(self.saveDir), self.saveDir, self.out_mask_size,
                   self.out_mask_border[0], fixed_ar, save_raw_mask=self.params.mask.save_raw,
                   save_test=self.params.mask.save_test,
                   show_img=1, frames_reader=self.frames_reader)

    def saveBoxesCSV(self, verbose=True, write=True):
        if not self.saveDir or not os.path.isdir(self.saveDir):
            print('Folder containing the loaded boxes does not exist')
            return None

        files = glob.glob(os.path.join(self.saveDir, '*.xml'))
        n_files = len(files)
        if n_files == 0:
            print('No loaded boxes found')
            return None

        def getint(fn):
            basename = os.path.basename(fn)
            num = re.sub("\D", "", basename)
            try:
                return int(num)
            except:
                return 0

        if len(files) > 0:
            files = sorted(files, key=getint)

        print('Loading annotations from {:d} files...'.format(n_files))

        file_id = 0
        n_boxes = 0
        csv_raw = []

        for file in files:
            xml_reader = PascalVocReader(file)
            shapes = xml_reader.getShapes()
            for shape in shapes:
                label, points, _, _, difficult, bbox_source, id_number, score, _, _ = shape

                if id_number is None:
                    id_number = -1

                xmin, ymin = points[0]
                xmax, ymax = points[2]

                raw_data = {
                    'target_id': int(id_number),
                    'filename': xml_reader.filename,
                    'width': xml_reader.width,
                    'height': xml_reader.height,
                    'class': label,
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax)
                }
                csv_raw.append(raw_data)
                n_boxes += 1

            file_id += 1
            if verbose:
                sys.stdout.write('\rDone {:d}/{:d} files with {:d} boxes'.format(
                    file_id, n_files, n_boxes))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        df = pd.DataFrame(csv_raw)
        if write:
            out_dir = os.path.dirname(self.saveDir)
            out_file_path = os.path.join(out_dir, 'annotations.csv')
            df.to_csv(out_file_path)
            return out_file_path
        else:
            return df


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


str_to_list = lambda _str, _type, _sep: list(map(_type, _str.split(_sep)))
list_int_x = lambda _str: str_to_list(_str, int, 'x')
list_int_comma = lambda _str: str_to_list(_str, int, ',')
list_float_comma = lambda _str: str_to_list(_str, float, ',')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Labeling Tool")
    parser.add_argument("--load_prev", type=int, default=0,
                        help="Load previously opened file or directory")
    parser.add_argument("--file_name", type=str, default="",
                        help="File or folder to load")
    parser.add_argument("--predef_class_file", type=str, default="",
                        help="Path to the file containing the list of predefined classes")

    # mask GUI parameters
    parser.add_argument("--mask_disp_size", type=list_int_comma, default=(1000, 1000),
                        help="Size of the window shown for drawing the mask")
    parser.add_argument("--mask_border_size", type=list_int_comma, default=(3, 3),
                        help="Size of border around the bounding box to include in the mask window")
    parser.add_argument("--mask_del_thresh", type=int, default=15,
                        help="Distance threshold for deleting the existing mask points")
    parser.add_argument("--mask_magnified_window", type=int, default=1,
                        help="Show magnified window around the cursor location")
    parser.add_argument("--mask_show_pts", type=int, default=1,
                        help="Show individual points")
    return parser.parse_args()


# def get_main_app(argv, params):
#     app = QApplication(argv)
#     app.setApplicationName(__appname__)
#     app.setWindowIcon(newIcon("app"))
#
#     return app, win
# def main(argv, params, logger):
#     '''construct main app and run it'''
#     app, _win = get_main_app(argv, params)
#     return app.exec_()

def logging_profile(self, message, *args, **kws):
    if self.isEnabledFor(PROFILE_LEVEL_NUM):
        self._log(PROFILE_LEVEL_NUM, message, args, **kws)


if __name__ == '__main__':
    _params = LabelingParams()
    processArguments(_params)

    # args = get_arguments()
    # if not args.predef_class_file:
    #     args.predef_class_file = os.path.join(os.path.dirname(sys.argv[0]), 'data', 'predefined_classes.txt')

    # setup logger
    PROFILE_LEVEL_NUM = 9
    logging.addLevelName(PROFILE_LEVEL_NUM, "PROFILE")
    logging.Logger.profile = logging_profile

    logging_fmt = '%(levelname)s::%(module)s::%(funcName)s::%(lineno)s :  %(message)s'
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level, format=logging_fmt)
    _logger = logging.getLogger()

    app = QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))

    win = MainWindow(_params, _logger)
    win.show()

    sys.exit(app.exec_())
