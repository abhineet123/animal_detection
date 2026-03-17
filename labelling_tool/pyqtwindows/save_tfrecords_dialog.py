import os, sys
import _pickle
from glob import glob

from PyQt5.QtWidgets import QMessageBox

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
from object_detection_module.data_processing.annotation_parsing import parse_PASCAL_VOC_groundtruth
from pyqtwindows.dataset_details_dialog import DatasetDetailsDialog
from libs.frames_readers import get_frames_reader


class SaveTFRecordsDialog(QMainWindow, WindowMixin):

    DELIMITER = " // "

    def __init__(self, main_window):
        super(SaveTFRecordsDialog, self).__init__()

        self.main_window = main_window
        self.CWD = os.path.abspath(os.getcwd())
        self.DEFAULT_TFRECORDS_SAVE_DIR = self.main_window.DEFAULT_TFRECORDS_SAVE_DIR
        if not os.path.exists(self.DEFAULT_TFRECORDS_SAVE_DIR):
            os.makedirs(self.DEFAULT_TFRECORDS_SAVE_DIR)

        self.setWindowTitle('Save TFRecords')
        self.setMinimumSize(750, 500)
        wid = QWidget(self)
        self.setCentralWidget(wid)
        layout = QVBoxLayout()
        wid.setLayout(layout)

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self.edit_dataset)
        layout.addWidget(self.list)

        self.use_ground_truth_bboxes = QCheckBox("Use ground truth bounding boxes")
        self.use_ground_truth_bboxes.setChecked(True)
        layout.addWidget(self.use_ground_truth_bboxes)
        self.use_detection_bboxes = QCheckBox("Use detection bounding boxes")
        self.use_detection_bboxes.setChecked(False)
        layout.addWidget(self.use_detection_bboxes)
        self.use_tracking_bboxes = QCheckBox("Use tracking bounding boxes")
        self.use_tracking_bboxes.setChecked(True)
        layout.addWidget(self.use_tracking_bboxes)

        pbtxt_label = QLabel("You can load a pbtxt file for label list here or leave it empty for automatic generation."
                             " Automatically generated file will be put in the same dir as .tfrecords file.")
        layout.addWidget(pbtxt_label)

        self.pbtxt_text = QLineEdit()
        layout.addWidget(self.pbtxt_text)

        self.pbtxt_button = QPushButton("Browse .pbtxt label file")
        layout.addWidget(self.pbtxt_button)
        self.pbtxt_button.clicked.connect(self.pbtxt_button_clicked)

        self.output = QTextEdit()
        # self.output.setDisabled(True)
        layout.addWidget(self.output)


        action = partial(newAction, self)

        save = action("&Save", self.save, "Ctrl+S")
        add_files = action("Add Video &Files", self.add_files, "Ctrl+F")
        add_dirs = action("Add Directo&ry", self.add_dirs, "Ctrl+R")
        edit_dataset = action("Edit Dataset", self.edit_dataset, "Ctrl+E")
        delete = action("&Delete", self.delete, "Ctrl+D")
        cancel = action("&Cancel", self.cancel, "Ctrl+C", enabled=False)
        hide = action("&Hide Window", self.hide, "Ctrl+H")

        self.actions = struct(save=save, add_files=add_files, add_dirs=add_dirs,
                              edit_dataset=edit_dataset, delete=delete, cancel=cancel, hide=hide,
                              all_actions=(save, add_files, add_dirs, edit_dataset, delete, cancel, hide))

        self.tools = self.toolbar("Tools", position=Qt.TopToolBarArea)
        self.tools.clear()
        addActions(self.tools, self.actions.all_actions)

    def save_curr_frames_list(self):
        _pickle.dump(self.curr_frames_list, open(self.pkl_path, 'wb'))

    def load_curr_frames_list(self):
        if os.path.exists(self.pkl_path):
            self.curr_frames_list = _pickle.load(open(self.pkl_path, 'rb'))
        else:
            self.curr_frames_list = []

    def reset_curr_frames_list(self, curr_annotation_path):
        self.curr_annotation_path = curr_annotation_path
        self.pkl_path = os.path.join(self.curr_annotation_path, '.frames_list.pkl')

    def update_frames_list(self, opened_frame, new_jump_interval):
        self.load_curr_frames_list()
        if len(self.curr_frames_list) == 0:
            # newly opened
            curr_start_frame = opened_frame
            curr_end_frame = opened_frame
            curr_jump_interval = new_jump_interval
            self.curr_frames_list.append([curr_start_frame, curr_end_frame, curr_jump_interval])
        else:
            # update
            curr_start_frame, curr_end_frame, curr_jump_interval = self.curr_frames_list[-1]
            create_new_list = False
            if (opened_frame - curr_start_frame) % curr_jump_interval == 0:
                if curr_start_frame <= opened_frame <= curr_end_frame:
                    pass
                else:
                    if opened_frame == curr_start_frame - curr_jump_interval:
                        self.curr_frames_list[-1] = [opened_frame, curr_end_frame, curr_jump_interval]
                    elif opened_frame == curr_end_frame + curr_jump_interval:
                        self.curr_frames_list[-1] = [curr_start_frame, opened_frame, curr_jump_interval]
                    else:
                        create_new_list = True
            else:
                create_new_list = True
            if create_new_list:
                self.curr_frames_list.append([opened_frame, opened_frame, new_jump_interval])
        self.save_curr_frames_list()

    def save(self):
        save_path = QFileDialog.getSaveFileName(self, "Save as", self.DEFAULT_TFRECORDS_SAVE_DIR, "TFRecords (*.tfrecords)")
        save_path = os.path.splitext(save_path[0])[0]
        if not save_path:
            return
        config_file = save_path + ".pkl"
        pbtxt_text = self.pbtxt_text.text()
        if pbtxt_text and os.path.isfile(pbtxt_text):
            self.pbtxt_path = pbtxt_text
        else:
            self.pbtxt_path = save_path + '.pbtxt'
            label_list = self.get_label_list()
            with open(self.pbtxt_path, 'w') as f:
                for index, label in enumerate(label_list):
                    f.writelines("item {\n  id: %s\n  name: \'%s\'\n}\n\n\n" % (index + 1, label))
        itemsTextList = [str(self.list.item(i).text().split(self.DELIMITER)[0]) for i in range(self.list.count())]
        valid_items = 0
        dataset_configs = []
        for item in itemsTextList:
            path = str(item)
            if os.path.exists(path):
                annotation_path = self.get_roi_annotation_dir(path)
                roi = DatasetDetailsDialog.getROIfromAnnotationsPath(annotation_path)
                if roi is None:
                    roi = [-1, -1, -1, -1]
                else:
                    roi = [roi['xmin'], roi['ymin'], roi['xmax'], roi['ymax']]
                frames_list_path = os.path.join(annotation_path, '.frames_lists.pkl')
                if os.path.exists(frames_list_path):
                    frames_list = set()
                    frames_list_rules = _pickle.load(open(frames_list_path, 'rb'))
                    for start_frame, end_frame, interval in frames_list_rules:
                        frames_list.update(list(range(start_frame, end_frame + 1, interval)))
                    frames_list = sorted(list(frames_list))
                else:
                    frames_reader = get_frames_reader(path)
                    frames_list = list(range(0, frames_reader.num_frames, 1))
                    del frames_reader
                bbox_source_filter = []
                if self.use_ground_truth_bboxes.isChecked():
                  bbox_source_filter += ["ground_truth"]
                if self.use_detection_bboxes.isChecked():
                  bbox_source_filter += ["object_detector"]
                if self.use_tracking_bboxes.isChecked():
                  bbox_source_filter += ["tracker"]
                dataset_configs.append(dict(
                    frames_source=path,
                    annotation_source=annotation_path,
                    annotation_type='pascal_voc',
                    label_map_file=self.pbtxt_path,
                    mot_class_name='object',
                    frames_list=frames_list,
                    bbox_source_filter=bbox_source_filter,
                    roi=roi,
                ))
                valid_items += 1
            _pickle.dump(dataset_configs, open(config_file, 'wb'))
        if valid_items:
            self.save_path = save_path
            self.call_program(config_file, save_path + '.tfrecords')

    def call_program(self, config_file, output_path):
        statement = ['./data_processing/convert_datasets_to_tfrecords.py',
                     '--config_file', config_file,
                     '--output_path', output_path,
                     '--tmp_dir', os.path.join(self.CWD, 'tmp'),
                     '--create_fake_negatives_prob', '0.00',
                     ]
        os.chdir(os.path.join(self.CWD, './object_detection_module'))
        self.write_output(" ".join(['python3', " ".join(statement)]))
        self.process = QProcess(self)
        self.process.readyRead.connect(self.data_ready)
        self.process.readyReadStandardError.connect(self.error_ready)
        self.process.started.connect(self.proc_start)
        self.process.finished.connect(self.proc_finish)
        self.process.start('python3', statement)
        os.chdir(self.CWD)

    def get_label_list(self):
        itemsTextList = [str(self.list.item(i).text()).split(self.DELIMITER)[0] for i in range(self.list.count())]
        all_names = set()
        for item in itemsTextList:
            path = str(item)
            annotation_path = self.get_roi_annotation_dir(path)
            if os.path.exists(annotation_path):
                if os.path.isdir(annotation_path):
                    _, all_names_ = parse_PASCAL_VOC_groundtruth(annotation_path, return_names=True)
                    all_names.update(all_names_)
        all_names = sorted(list(all_names))
        return all_names

    def get_roi_annotation_dir(self, path, as_base_name=False):
        if os.path.isfile(path):
            path = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0], "annotations")
        else:
            path = os.path.join(path, "annotations")
        config_path = os.path.join(path, '.config.pkl')
        if os.path.exists(config_path):
            saved_config = _pickle.load(open(config_path, 'rb'))
            roi_path = saved_config['roi_path']
            roi_path = roi_path if roi_path else path
        else:
            roi_path = path
        if as_base_name:
            roi_path = os.path.basename(roi_path)
        return roi_path

    def addItem(self, path):
        item = QListWidgetItem("{}{}ROI: {}".format(path, self.DELIMITER,
                                                    self.get_roi_annotation_dir(path, as_base_name=True)))
        self.list.addItem(item)

    def add_files(self):
        path = os.path.expanduser('~')
        opened_files = QFileDialog.getOpenFileNames(self, 'Add files', path, 'Video files (*)')
        for file in opened_files[0]:
            frames_reader = get_frames_reader(file)
            if frames_reader.num_frames > 0:
                self.addItem(file)
            del frames_reader

    def add_dirs(self):
        path = os.path.expanduser('~')
        opened_directory = QFileDialog.getExistingDirectory(self,
                                         'Add Directory', path, QFileDialog.ShowDirsOnly
                                         | QFileDialog.DontResolveSymlinks)
        frames_reader = get_frames_reader(opened_directory)
        if frames_reader.num_frames > 0:
            self.addItem(opened_directory)
        del frames_reader

    def edit_dataset(self):
        selected_dataset = self.list.currentItem().text().split(self.DELIMITER)[0]
        self.currently_editing_dataset = selected_dataset
        self.currently_editing_index = self.list.currentRow()
        frames_reader = get_frames_reader(selected_dataset)
        num_frames = frames_reader.num_frames
        del frames_reader
        dataset_frame_selection_dialog = \
            DatasetDetailsDialog(parent=self, parent_window=self,
                                 num_frames=num_frames, dataset_path=selected_dataset)
        dataset_frame_selection_dialog.show()

    def update_selected_item(self):
        self.list.item(self.currently_editing_index).setText(
            "{}{}ROI: {}".format(self.currently_editing_dataset, self.DELIMITER,
                                 self.get_roi_annotation_dir(self.currently_editing_dataset, as_base_name=True)))

    def delete(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def cancel(self):
        if self._is_running():
            reply = QMessageBox.question(self, 'Confirm Stop',
                         "Are you sure you want to stop? "
                         "You can hide this window to allow background running.",
                                         QMessageBox.Yes, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.process.kill()
                return True
            else:
                return False
        else:
            return True

    def closeEvent(self, QCloseEvent):
        QCloseEvent.ignore()
        self.hide()

    def pbtxt_button_clicked(self):
        path = self.pbtxt_text.text()
        if not (path and os.path.isfile(path)):
            path = os.path.expanduser('./tmp')
        label_file = QFileDialog.getOpenFileName(self, "Open label file", path,
                                                 "Protobuf txt (*.pbtxt)")
        label_file = label_file[0]
        if label_file:
            self.pbtxt_text.setText(label_file)

    def _is_running(self):
        return not self.actions.save.isEnabled()

    def write_output(self, text):
        cursor = self.output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.output.ensureCursorVisible()

    def data_ready(self):
        msg = str(self.process.readAll().data(), encoding='utf-8').replace('\\n', '\n')
        self.write_output(msg)

    def error_ready(self):
        msg = str(self.process.readAllStandardError().data(), encoding='utf-8').replace('\\n', '\n')
        self.write_output(msg)

    def proc_start(self):
        self.actions.save.setEnabled(False)
        self.actions.cancel.setEnabled(True)
        self.actions.add_files.setEnabled(False)
        self.actions.add_dirs.setEnabled(False)
        self.write_output("\nTFRecords Process started\n")

    def proc_finish(self):
        self.actions.save.setEnabled(True)
        self.actions.cancel.setEnabled(False)
        self.actions.add_files.setEnabled(True)
        self.actions.add_dirs.setEnabled(True)
        if self.main_window.other_windows.detection_settings_dialog.get_setting_choice() == "custom":
            self.main_window.other_windows.detection_settings_dialog.pbtxt_text.setCurrentText(self.pbtxt_path)
            self.main_window.other_windows.detection_settings_dialog.tfrecords_text.setText(self.save_path
                                                                                            + '.tfrecords')
            self.main_window.other_windows.detection_settings_dialog.save_settings()
        self.write_output("\nTFRecords Process finished\n")
        QMessageBox.information(self, "Finished!", "Creating tfrecords file finished!", QMessageBox.Ok)

    def stop(self):
        if self._is_running():
            reply = QMessageBox.question(self, 'Confirm Stop',
                         "Are you sure you want to stop?"
                         " You may lose or even corrupt your data!", QMessageBox.Yes, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.process.kill()


if __name__ == '__main__':
    app = QApplication([])
    win = SaveTFRecordsDialog()
    win.show()
    app.exec_()
