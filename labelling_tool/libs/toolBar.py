try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *


class PrettyToolBar(QToolBar):

    def __init__(self, title):
        super(PrettyToolBar, self).__init__(title)
        layout = self.layout()
        m = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*m)
        self.setContentsMargins(*m)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

    def addAction(self, action):
        if isinstance(action, QWidgetAction):
            return super(PrettyToolBar, self).addAction(action)
        btn = PrettyToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        self.addWidget(btn)


class ToolBar(QToolBar):

    def __init__(self, title):
        super(ToolBar, self).__init__(title)
        layout = self.layout()
        m = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*m)
        self.setContentsMargins(*m)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

    def addAction(self, action):
        if isinstance(action, QWidgetAction):
            return super(ToolBar, self).addAction(action)
        btn = ToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        self.addWidget(btn)


class PrettyToolButton(QToolButton):
    """ToolBar companion class which ensures all buttons have the same size."""
    minSize = (40, 40)

    def minimumSizeHint(self):
        ms = super(PrettyToolButton, self).minimumSizeHint()
        w1, h1 = ms.width(), ms.height()
        w2, h2 = self.minSize
        PrettyToolButton.minSize = max(w1, w2), max(h1, h2)
        # ToolButton.minSize = 40, 40
        return QSize(*PrettyToolButton.minSize)


class ToolButton(QToolButton):
    """ToolBar companion class which ensures all buttons have the same size."""
    minSize = (40, 40)

    def minimumSizeHint(self):
        ms = super(ToolButton, self).minimumSizeHint()
        w1, h1 = ms.width(), ms.height()
        w2, h2 = self.minSize
        ToolButton.minSize = 40, 40
        return QSize(*ToolButton.minSize)
