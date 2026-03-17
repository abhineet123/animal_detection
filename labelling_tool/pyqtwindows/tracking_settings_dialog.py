import os, sys

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    if sys.version_info.major >= 3:
        import sip

        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
from pyqtwindows.WindowMixin import WindowMixin
from libs.lib import newAction, struct, addActions
from functools import partial


class TrackingSettingDialog(QMainWindow, WindowMixin):
    def __init__(self, main_window):
        super(TrackingSettingDialog, self).__init__()

        self.main_window = main_window
        if self.main_window is None:
            default_cfg_path = "../tracking/cfg/params.cfg"
            default_mtf_cfg_path = "../tracking/mtf/"
        else:
            try:
                default_cfg_path = self.main_window.tracking_settings.cfg_path
                default_mtf_cfg_path = self.main_window.tracking_settings.mtf_cfg_path
            except AttributeError:
                default_cfg_path = "../tracking/cfg/params.cfg"
                default_mtf_cfg_path = "../tracking/cfg/mtf/"

        self.CWD = os.path.abspath(os.getcwd())

        self.setWindowTitle('Tracker Settings')
        self.setMinimumSize(800, 300)
        wid = QWidget(self)
        self.setCentralWidget(wid)
        layout = QVBoxLayout()
        wid.setLayout(layout)

        self.custom_settings_group = QGroupBox("Server Settings:")
        layout.addWidget(self.custom_settings_group)
        custom_settings_layout = QVBoxLayout()
        self.custom_settings_group.setLayout(custom_settings_layout)

        cfg_layout = QHBoxLayout()
        cfg_label = QLabel("Config file:")
        cfg_layout.addWidget(cfg_label)
        self.cfg_text = QLineEdit()
        self.cfg_text.setText(default_cfg_path)
        cfg_layout.addWidget(self.cfg_text)
        self.cfg_button = QPushButton("Browse")
        cfg_layout.addWidget(self.cfg_button)
        self.cfg_button.clicked.connect(self.cfg_button_clicked)
        custom_settings_layout.addLayout(cfg_layout)

        self.mtf_settings_group = QGroupBox("MTF settings:")
        layout.addWidget(self.mtf_settings_group)
        mtf_settings_layout = QVBoxLayout()
        self.mtf_settings_group.setLayout(mtf_settings_layout)

        mtf_cfg_label = QLabel("Select directory containing MTF configuration files:")
        mtf_settings_layout.addWidget(mtf_cfg_label)
        self.mtf_cfg_text = QLineEdit()
        self.mtf_cfg_text.setText(default_mtf_cfg_path)
        mtf_settings_layout.addWidget(self.mtf_cfg_text)
        self.mtf_cfg_button = QPushButton("Choose directory")
        mtf_settings_layout.addWidget(self.mtf_cfg_button)
        self.mtf_cfg_button.clicked.connect(self.mtf_cfg_button_clicked)

        # tfrecords_label = QLabel("Select dataset .tfrecords file:")
        # custom_settings_layout.addWidget(tfrecords_label)
        # self.tfrecords_text = QLineEdit(default_tfrecords_path)
        # custom_settings_layout.addWidget(self.tfrecords_text)
        # self.tfrecords_button = QPushButton("Browse .tfrecords file")
        # custom_settings_layout.addWidget(self.tfrecords_button)
        # self.tfrecords_button.clicked.connect(self.tfrecords_button_clicked)

        action = partial(newAction, self)

        save_settings = action("&Save Settings", self.save_settings, "Ctrl+S")
        close = action("&Close", self.close, "Ctrl+C")

        self.actions = struct(save_settings=save_settings, close=close,
                              all_actions=(save_settings, close))

        self.tools = self.toolbar("Tools", position=Qt.BottomToolBarArea)
        self.tools.clear()
        addActions(self.tools, self.actions.all_actions)

    def cfg_button_clicked(self):
        path = self.cfg_text.text()
        if not os.path.isfile(path):
            path = os.path.expanduser(os.path.join(self.CWD, '../tracking/cfg'))
        if not os.path.exists(path):
            path = os.path.expanduser("~")
        label_file = QFileDialog.getOpenFileName(self, "Open cfg file", path,
                                                 "Configuration files (*.cfg)")
        label_file = label_file[0]
        if label_file:
            self.cfg_text.setText(label_file)

    def mtf_cfg_button_clicked(self):
        path = self.mtf_cfg_text.text()
        if not os.path.isdir(path):
            path = os.path.expanduser(os.path.join(self.CWD, '../tracking/mtf'))
        if not os.path.exists(path):
            path = os.path.expanduser("~")
        opened_directory = QFileDialog.getExistingDirectory(
            self, 'Add Directory', path, QFileDialog.ShowDirsOnly
                                         | QFileDialog.DontResolveSymlinks)
        if opened_directory:
            self.mtf_cfg_text.setText(opened_directory)

    def save_settings(self):
        if self.main_window is not None:
            self.main_window.tracking_settings = struct(
                cfg_path=self.cfg_text.text(),
                mtf_cfg_path=self.mtf_cfg_text.text()
            )
            self.main_window.settings.save()
            self.close()

    def _is_running(self):
        return False

if __name__ == '__main__':
    app = QApplication([])
    win = TrackingSettingDialog(None)
    win.show()
    app.exec_()
