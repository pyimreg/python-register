# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'editor.ui'
#
# Created: Thu Mar  8 07:24:07 2012
#      by: PyQt4 UI code generator 4.8.5
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


class Editor(QtGui.QMainWindow):
    
    def __init__(self):
        super(Editor, self).__init__()
        self.initUI()
        
    def initUI(self):        
        self.setObjectName(_fromUtf8("MainWindow"))
        self.resize(823, 663)
        self.setWindowTitle(QtGui.QApplication.translate("MainWindow", "Python-Register Editor", None, QtGui.QApplication.UnicodeUTF8))
        self.centralwidget = QtGui.QWidget(self)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.widget = QtGui.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 20, 791, 571))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.widget)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.imageLeft = QtGui.QGraphicsView(self.widget)
        self.imageLeft.setObjectName(_fromUtf8("imageLeft"))
        self.horizontalLayout.addWidget(self.imageLeft)
        self.imageRight = QtGui.QGraphicsView(self.widget)
        self.imageRight.setObjectName(_fromUtf8("imageRight"))
        self.horizontalLayout.addWidget(self.imageRight)
        self.horizontalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(261, 20, 291, 151))
        self.horizontalLayoutWidget.setObjectName(_fromUtf8("horizontalLayoutWidget"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setMargin(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.imageZoomRight = QtGui.QGraphicsView(self.horizontalLayoutWidget)
        self.imageZoomRight.setObjectName(_fromUtf8("imageZoomRight"))
        self.horizontalLayout_2.addWidget(self.imageZoomRight)
        self.imageZoomLeft = QtGui.QGraphicsView(self.horizontalLayoutWidget)
        self.imageZoomLeft.setObjectName(_fromUtf8("imageZoomLeft"))
        self.horizontalLayout_2.addWidget(self.imageZoomLeft)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 823, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setTitle(QtGui.QApplication.translate("MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuAbout = QtGui.QMenu(self.menubar)
        self.menuAbout.setTitle(QtGui.QApplication.translate("MainWindow", "About", None, QtGui.QApplication.UnicodeUTF8))
        self.menuAbout.setObjectName(_fromUtf8("menuAbout"))
        self.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(self)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        self.setStatusBar(self.statusbar)
        self.actionOpen_left = QtGui.QAction(self)
        self.actionOpen_left.setText(QtGui.QApplication.translate("MainWindow", "Open left...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOpen_left.setObjectName(_fromUtf8("actionOpen_left"))
        self.actionOpen_right = QtGui.QAction(self)
        self.actionOpen_right.setText(QtGui.QApplication.translate("MainWindow", "Open right...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOpen_right.setObjectName(_fromUtf8("actionOpen_right"))
        self.actionSave_features = QtGui.QAction(self)
        self.actionSave_features.setText(QtGui.QApplication.translate("MainWindow", "Save features...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave_features.setObjectName(_fromUtf8("actionSave_features"))
        self.actionExit = QtGui.QAction(self)
        self.actionExit.setText(QtGui.QApplication.translate("MainWindow", "Exit", None, QtGui.QApplication.UnicodeUTF8))
        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.menuFile.addAction(self.actionOpen_left)
        self.menuFile.addAction(self.actionOpen_right)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave_features)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())
        
        self.actionOpen_left.triggered.connect(self.openLeft)
        self.actionExit.triggered.connect(self.exxit)


    def openLeft(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open image file', '~')
        pixmap = QtGui.QPixmap(filename)
        scene = QtGui.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.imageLeft.setScene(scene)
        
    def exxit(self):
        self.close()
        