import os, sys
from glob import glob

from PyQt5.QtWidgets import QMessageBox
import _pickle

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


class DatasetDetailsDialog(QMainWindow, QDialog, WindowMixin):

    def __init__(self, parent_window, num_frames, dataset_path, **kwargs):
        super(DatasetDetailsDialog, self).__init__(**kwargs)

        self.CWD = os.path.abspath(os.getcwd())

        self.dataset_path = dataset_path
        self.parent_window = parent_window

        self.setWindowTitle('Dataset Details')
        self.setMinimumSize(400, 300)
        wid = QWidget(self)
        self.setCentralWidget(wid)
        layout = QVBoxLayout()
        wid.setLayout(layout)

        self.filepath_label = QLabel("Path: {}".format(dataset_path))
        layout.addWidget(self.filepath_label)

        set_roi_dir_widget = QWidget()
        set_roi_dir_layout = QHBoxLayout()
        set_roi_dir_widget.setLayout(set_roi_dir_layout)
        roi_dir_label = QLabel("ROI dir: ")
        self.roi_dir_text = QLineEdit()
        self.roi_dir_button = QPushButton("Select")
        self.roi_dir_button.clicked.connect(self.roi_dir_button_clicked)
        set_roi_dir_layout.addWidget(roi_dir_label)
        set_roi_dir_layout.addWidget(self.roi_dir_text)
        set_roi_dir_layout.addWidget(self.roi_dir_button)
        layout.addWidget(set_roi_dir_widget)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Start frame", "End frame", "Jump Interval"])
        self.table.resize(400, 300)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.table)

        action = partial(newAction, self)

        add = action("&Add", self.add, "Ctrl+A")
        delete = action("&Delete", self.delete, "Ctrl+D")
        clear = action("Clear", self.clear)
        ok = action("&OK", self.ok, "Ctrl+O")
        cancel = action("&Close", self.close, "Ctrl+C")

        self.actions = struct(add=add, delete=delete, clear=clear, cancel=cancel, ok=ok,
                              all_actions=(add, delete, clear, ok, cancel))

        self.tools = self.toolbar("Tools", position=Qt.BottomToolBarArea)
        self.tools.clear()
        addActions(self.tools, self.actions.all_actions)

        if os.path.isdir(dataset_path):
            tentative_annotations_path = os.path.join(dataset_path, 'annotations')
        else:
            tentative_annotations_path = os.path.join(os.path.dirname(dataset_path),
                                                      os.path.splitext(os.path.basename(dataset_path))[0],
                                                      'annotations')
        if not os.path.exists(tentative_annotations_path):
            os.makedirs(tentative_annotations_path)
        self.config_path = os.path.join(tentative_annotations_path, '.config.pkl')

        self.num_frames = num_frames

        self.load_roi_path()

        self.load_and_populate_table()

    def load_roi_path(self):
        if not os.path.exists(self.config_path):
            _pickle.dump(dict(roi_path=None), open(self.config_path, 'wb'))
        saved_config = _pickle.load(open(self.config_path, 'rb'))
        roi_path = saved_config['roi_path']
        roi_path = roi_path if roi_path else None
        if roi_path is not None:
            self.roi_dir_text.setText(roi_path)

    def get_frames_lists_path(self):
        roi_path = self.roi_dir_text.text()
        roi_path = roi_path if roi_path else None
        frames_lists_dir = roi_path if roi_path else os.path.dirname(self.config_path)
        frames_lists_path = os.path.join(frames_lists_dir, '.frames_lists.pkl')
        return frames_lists_path

    def load_and_populate_table(self):
        frames_lists_path = self.get_frames_lists_path()
        self.clear()
        if os.path.isfile(frames_lists_path):
            frames_lists = _pickle.load(open(frames_lists_path, 'rb'))
            for start_frame, end_frame, interval in frames_lists:
                self.add(start_frame=str(start_frame), end_frame=str(end_frame), interval=str(interval))

    def save_frames_lists_to_file(self):
        frames_lists_path = self.get_frames_lists_path()
        list_to_save = []
        for j in range(self.table.rowCount()):
            start_frame = self.table.item(j, 0).text()
            end_frame = self.table.item(j, 1).text()
            interval = self.table.item(j, 2).text()
            list_to_save.append([int(start_frame), int(end_frame), int(interval)])
        _pickle.dump(list_to_save, open(frames_lists_path, 'wb'))

    def add(self, _value=None, start_frame=0, end_frame=None, interval=1):
        if not end_frame:
            end_frame = self.num_frames
        num_row = self.table.rowCount()
        self.table.insertRow(num_row)
        self.table.setItem(num_row, 0, QTableWidgetItem(str(start_frame)))
        self.table.setItem(num_row, 1, QTableWidgetItem(str(end_frame)))
        self.table.setItem(num_row, 2, QTableWidgetItem(str(interval)))

    def delete(self):
        index = list(set([item.row() for item in self.table.selectedItems()]))[0]
        self.table.removeRow(index)

    def closeEvent(self, QCloseEvent):
        self.parent_window.update_selected_item()
        QCloseEvent.accept()

    def ok(self):
        roi_path = self.roi_dir_text.text()
        roi_path = roi_path if roi_path else None
        if roi_path is not None:
            _pickle.dump(dict(roi_path=roi_path), open(self.config_path, 'wb'))
            self.save_frames_lists_to_file()
        else:
            self.save_frames_lists_to_file()
        self.close()

    def clear(self):
        self.table.setRowCount(0)

    def roi_dir_button_clicked(self):
        if os.path.isdir(self.dataset_path):
            path = self.dataset_path
        elif os.path.isfile(self.dataset_path):
            path = os.path.dirname(self.dataset_path)
        else:
            return
        roi_path = QFileDialog.getExistingDirectory(self,
                                         'Add Directory', path, QFileDialog.ShowDirsOnly
                                         | QFileDialog.DontResolveSymlinks)

        roi_path = roi_path if roi_path else None
        if roi_path is not None:
            self.roi_dir_text.setText(roi_path)
            self.load_and_populate_table()

    @classmethod
    def getROIfromAnnotationsPath(cls, annotations_path):
        dir_name = os.path.basename(annotations_path)
        roi_extents = [int(s) for s in dir_name.split('_') if s.isdigit()]
        if len(roi_extents) != 4:
            return None
        xmin, ymin, xmax, ymax = roi_extents
        return dict(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

if __name__ == '__main__':
    app = QApplication([])
    win = DatasetDetailsDialog()
    win.show()
    app.exec_()