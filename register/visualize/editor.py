# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'editor.ui'
#
# Created: Wed Mar  7 16:06:43 2012
#      by: PyQt4 UI code generator 4.8.5
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(823, 663)
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "Python-Register Editor", None, QtGui.QApplication.UnicodeUTF8))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.widget = QtGui.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 20, 791, 571))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.widget)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.graphicsView = QtGui.QGraphicsView(self.widget)
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.horizontalLayout.addWidget(self.graphicsView)
        self.graphicsView_2 = QtGui.QGraphicsView(self.widget)
        self.graphicsView_2.setObjectName(_fromUtf8("graphicsView_2"))
        self.horizontalLayout.addWidget(self.graphicsView_2)
        self.horizontalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(261, 20, 291, 151))
        self.horizontalLayoutWidget.setObjectName(_fromUtf8("horizontalLayoutWidget"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setMargin(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.graphicsView_4 = QtGui.QGraphicsView(self.horizontalLayoutWidget)
        self.graphicsView_4.setObjectName(_fromUtf8("graphicsView_4"))
        self.horizontalLayout_2.addWidget(self.graphicsView_4)
        self.graphicsView_3 = QtGui.QGraphicsView(self.horizontalLayoutWidget)
        self.graphicsView_3.setObjectName(_fromUtf8("graphicsView_3"))
        self.horizontalLayout_2.addWidget(self.graphicsView_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 823, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setTitle(QtGui.QApplication.translate("MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuAbout = QtGui.QMenu(self.menubar)
        self.menuAbout.setTitle(QtGui.QApplication.translate("MainWindow", "About", None, QtGui.QApplication.UnicodeUTF8))
        self.menuAbout.setObjectName(_fromUtf8("menuAbout"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_left = QtGui.QAction(MainWindow)
        self.actionOpen_left.setText(QtGui.QApplication.translate("MainWindow", "Open left...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOpen_left.setObjectName(_fromUtf8("actionOpen_left"))
        self.actionOpen_right = QtGui.QAction(MainWindow)
        self.actionOpen_right.setText(QtGui.QApplication.translate("MainWindow", "Open right...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionOpen_right.setObjectName(_fromUtf8("actionOpen_right"))
        self.actionSave_features = QtGui.QAction(MainWindow)
        self.actionSave_features.setText(QtGui.QApplication.translate("MainWindow", "Save features...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave_features.setObjectName(_fromUtf8("actionSave_features"))
        self.actionExit = QtGui.QAction(MainWindow)
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

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        pass

