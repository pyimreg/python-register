""" Model view controller """

import sys
import graphics

from PyQt4 import QtCore, QtGui


class PointModel():
    """
    Representation of things, like points and polygons.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Controller(QtGui.QDialog):
    """ Control of view and model """

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.view = graphics.Ui_Dialog()
        self.view.setupUi(self)

        self.model = PointModel(50, 50)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    application = Controller()
    application.show()
    sys.exit(app.exec_())
