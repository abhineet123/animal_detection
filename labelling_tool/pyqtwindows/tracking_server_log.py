import os, sys, time
import socket

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

TRACKING_PORT = 3002


class TrackingServerLog(QMainWindow, WindowMixin):
    def __init__(self, main_window, settings_dialog):
        super(TrackingServerLog, self).__init__()

        self.main_window = main_window
        self.settings_dialog = settings_dialog
        self.CWD = os.path.abspath(os.getcwd())

        self.setWindowTitle('Object Tracker Server Log')
        self.setMinimumSize(800, 600)
        wid = QWidget(self)
        self.setCentralWidget(wid)
        layout = QVBoxLayout()
        wid.setLayout(layout)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        # self.output.setTextBackgroundColor(Qt.black)
        # self.output.setTextColor(Qt.white)
        layout.addWidget(self.output)

        action = partial(newAction, self)

        hide = action("&Hide Window", self.hide, "Ctrl+H")
        stop = action("&Stop Tracker", self.stop, "Ctrl+S", enabled=True)

        self.actions = struct(hide=hide,stop=stop,
                              all_actions=(hide,stop))

        self.tools = self.toolbar("Tools", position=Qt.TopToolBarArea)
        self.tools.clear()
        addActions(self.tools, self.actions.all_actions)

        self.process = None

        self.__is_running = False

        self.cfg_text = self.settings_dialog.cfg_text
        self.mtf_cfg_text = self.settings_dialog.mtf_cfg_text

        self.start_server()

    def _is_running(self):
        return self.__is_running

    def stop(self):
        if self.process:
            self.process.kill()
            self.process = None
            self.__is_running = False

    def getCommandLineArgs(self):
        statement = ['--cfg={:s}'.format(self.cfg_text.text())]
        statement.append('--server.patch_tracker.mtf_cfg_dir={:s}'.format(self.mtf_cfg_text.text()))

        return statement

    def start_server(self):

        return

        if self.__is_running:
            return

        statement = ['main.py']
        statement.extend(self.getCommandLineArgs())

        os.chdir(os.path.join(self.CWD, './tracking'))
        env = QProcessEnvironment.systemEnvironment()
        self.write_output(" ".join(['python2', " ".join(statement)]) + "\n")
        self.process = QProcess(self)
        self.process.readyRead.connect(self.data_ready)
        self.process.readyReadStandardError.connect(self.error_ready)
        self.process.started.connect(self.proc_start)
        self.process.finished.connect(self.proc_finish)
        self.process.setProcessEnvironment(env)
        self.process.start('python2', statement)

        os.chdir(self.CWD)

    def stop_server(self):
        return
        if self.__is_running:
            self.process.kill()

            # while True:
            #     # wait till the server has stopped accepting connections
            #     try:
            #         socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
            #             ('localhost', TRACKING_PORT))
            #         time.sleep(0.5)
            #     except ConnectionRefusedError:
            #         break
            self.__is_running = False


    def proc_start(self):
        self.write_output("Server started\n")
        self.__is_running = True

    def proc_finish(self):
        self.write_output("Server stopped\n")
        self.__is_running = False

    def closeEvent(self, QCloseEvent):
        QCloseEvent.ignore()
        self.hide()

    def write_output(self, text):
        cursor = self.output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.output.ensureCursorVisible()
        # self.output.insertPlainText(text)

    def data_ready(self):
        msg = str(self.process.readAll().data(), encoding='utf-8').replace('\\n', '\n')
        self.write_output(msg)

    def error_ready(self):
        msg = str(self.process.readAllStandardError().data(), encoding='utf-8').replace('\\n', '\n')
        self.write_output(msg)


if __name__ == '__main__':
    app = QApplication([])
    win = TrackingServerLog(None)
    win.show()
    app.exec_()
