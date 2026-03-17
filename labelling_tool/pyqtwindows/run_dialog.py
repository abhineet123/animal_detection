import os
import sys
import cv2
import numpy as np
import socket
import time
from libs.netio import send_msg_to_connection, recv_from_connection

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
from pyqtwindows.WindowMixin import WindowMixin
from libs.lib import newAction, struct, addActions
from functools import partial
from libs.pascal_voc_io import PascalVocReader, PascalVocWriter
from libs.labelFile import LabelFile
from libs.frames_readers import DirectoryReader, VideoReader


class RunDialog:
    def __init__(self, main_window):
        self.main_window = main_window
        self.CWD = os.path.abspath(os.getcwd())

        self.save_boxes = True
        self.max_frame = 0
        self.add_bboxes_threads = []
        self.currIndex = None

    def handle_add_bboxes_request(self, msg, connection):
        try:
            if self.currIndex is None:
                self.currIndex = self.main_window.mImgList.index(self.main_window.filePath)

            add_bbox_thread = AddBBoxesThread(run_window=self, main_window=self.main_window,connection=connection,
                                              msg=msg, currIndex=self.currIndex, save_boxes=self.save_boxes)
            add_bbox_thread.reload_signal.connect(self.main_window.update_current_image)
            # add_bbox_thread.update_status_signal.connect(self.update_status)
            # add_bbox_thread.finish_signal.connect(self.finish_running)
            while len(self.add_bboxes_threads) > 0:
                old_thread = self.add_bboxes_threads[0]
                if old_thread.isFinished():
                    _thread = self.add_bboxes_threads.pop(0)
                    # del _thread
                else:
                    break
            self.add_bboxes_threads.append(add_bbox_thread)
            add_bbox_thread.start()
            # send_msg_to_connection(dict(status='success'), connection)
        except (KeyboardInterrupt, SystemExit):
            return

    # def finish_running(self, trigger_batch_message=True):
    #     print("Running finished!")
    #     if trigger_batch_message:
    #         QMessageBox.information(self, "Finished", "Batch running finished!", QMessageBox.Ok)

    def handle_requests(self, connection):
        msg = recv_from_connection(connection)
        if msg['action'] == 'add_bboxes':
            # print(msg)
            '''
            syntax: dictionary that has:
            path (str), frame_no (int), bboxes (list of lists of 4 ints), src (str)
            return: dictionary
            status=success/failure, error (str)
            '''
            try:
                self.handle_add_bboxes_request(msg, connection)
            except BaseException as e:
                print("Adding failed: {}".format(str(e)))
                send_msg_to_connection(dict(status='failure', error=str(e)), connection)
        else:
            raise NotImplementedError("Wrong type of message!")

    # def update_status(self, msg):
    #     self.status_label.setText(msg)


class AddBBoxesThread(QThread):
    reload_signal = pyqtSignal()
    update_status_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(bool)

    def __init__(self, run_window, main_window, connection, msg, currIndex, save_boxes):
        super(AddBBoxesThread, self).__init__()
        self.connection = connection
        self.msg = msg
        self.run_window = run_window
        self.main_window = main_window
        self.currIndex = currIndex
        self.save_boxes = save_boxes

    def run(self):
        path = self.msg['path']
        frame_number = self.msg['frame_number']
        width = self.msg['width']
        height = self.msg['height']
        channel = self.msg['channel']
        bboxes = self.msg['bboxes']
        scores = self.msg['scores']
        labels = self.msg['labels']
        bbox_source = self.msg['bbox_source']
        id_numbers = self.msg['id_numbers']
        last_frame_number = self.msg['last_frame_number']
        trigger_tracking_request = self.msg['trigger_tracking_request']
        num_frames = self.msg['num_frames']
        new_box_from_pt = (bbox_source == 'single_object_tracker')
        num_new_bboxes = len(bboxes)

        if 'masks' in self.msg.keys():
            masks = self.msg['masks']
        else:
            masks = None
        trigger_batch_message = num_frames > 1
        if not self.main_window.frames_reader or os.path.abspath(path) != self.main_window.frames_reader.get_path():
            print("Incorrect path!")
            # send_self.msg_to_connection(dict(status="failure", error="Incorrect path!"), connection)
        else:
            if new_box_from_pt or self.save_boxes:
                xml_path = self.main_window.get_annotation_path(frame_number)
                if os.path.exists(xml_path):
                    try:
                        tVocParseReader = PascalVocReader(xml_path)
                        shapes = tVocParseReader.getShapes()
                    except:
                        shapes = []
                else:
                    shapes = []
                imageShape = [height, width, channel]
                if isinstance(self.main_window.frames_reader, DirectoryReader):
                    img_full_path = self.main_window.mImgList[frame_number]
                    folder_name = os.path.dirname(img_full_path)
                    file_name = os.path.basename(img_full_path)
                else:
                    folder_name = self.main_window.frames_reader.video_fn
                    file_name = str(frame_number)

                tVocWriter = PascalVocWriter(folder_name, file_name, imageShape)            #
                for label, points, _, _, difficult, _bbox_source, id_number, score, mask, mask_img in shapes:
                    existing_box_from_gt = _bbox_source == "ground_truth"
                    if new_box_from_pt and existing_box_from_gt and id_number == id_numbers[0]:
                        print('Received duplicale target {:d} bbox for frame {:d} for which GT exists'.format(
                            id_numbers[0], frame_number))
                        return
                    bndbox = LabelFile.convertPoints2BndBox(points, label)
                    if existing_box_from_gt or _bbox_source != bbox_source or label == 'gate' or \
                            (_bbox_source == "single_object_tracker" and id_number != id_numbers[0]):
                        # Override previous bboxes from the same source
                        tVocWriter.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label,
                                             difficult, _bbox_source, id_number, score, mask, mask_img)
                for j in range(num_new_bboxes):
                    bbox = bboxes[j]
                    xmin = bbox['xmin']
                    ymin = bbox['ymin']
                    xmax = bbox['xmax']
                    ymax = bbox['ymax']
                    label = labels[j]
                    score = scores[j]
                    difficulty = False
                    id_number = id_numbers[j]
                    if masks is not None:
                        # box_h = ymax - ymin
                        # box_w = xmax - xmin
                        # mask_img = np.array(masks[j]).reshape((box_h, box_w))
                        # mask_img = np.array(masks[j])
                        print('Loading mask from {:s}'.format(masks[j]))
                        # mask_img = cv2.imread(masks[j])
                        mask_img = np.load(masks[j])

                        if mask_img is None:
                            print('mask image could not be read')
                    else:
                        mask_img = None

                    tVocWriter.addBndBox(xmin, ymin, xmax, ymax, label, difficulty, bbox_source,
                                         id_number, score, None, mask_img)
                    # print('bbox: ', bbox)
                tVocWriter.save(targetFile=xml_path)

            if self.currIndex == frame_number:
                self.reload_signal.emit()
            if new_box_from_pt:
                print('Received target {:d} bbox for frame {:d}'.format(id_numbers[0], frame_number))

            else:
                self.update_status_signal.emit("Received {:d} {:s} bboxes for frame {:d}.".format(
                    num_new_bboxes, bbox_source, frame_number))
            if last_frame_number == frame_number and not trigger_tracking_request:
                self.update_status_signal.emit("Running finished for {:d} frames.".format(num_frames))
                self.finish_signal.emit(trigger_batch_message)

# if __name__ == '__main__':
#     app = QApplication([])
#     win = RunDialog(None)
#     win.show()
#     app.exec_()
