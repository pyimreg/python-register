# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'graphics.ui'
#
# Created: Fri Mar 16 16:59:34 2012
#      by: PyQt4 UI code generator 4.7
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(648, 612)
        Dialog.setMouseTracking(True)
        Dialog.setSizeGripEnabled(False)
        Dialog.setModal(False)
        
        self.scene = QtGui.QGraphicsScene()
        
        self.graphicsView = QtGui.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(10, 10, 631, 591))
        self.graphicsView.setObjectName("graphicsView")
        
        self.graphicsView.setScene(self.scene)
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "View", None, QtGui.QApplication.UnicodeUTF8))
    
    
    def pointView(self, model):
        """
        Views a model as a point.
        """
        self.scene.addRect(
            QtCore.QRect(model.x, model.y, 10, 10),
            pen=QtGui.QPen()
            )
    
        
        