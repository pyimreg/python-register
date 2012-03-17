import sys
import os

from PyQt4.QtCore import QDateTime, QObject, QUrl, pyqtSignal
from PyQt4.QtGui import QApplication
from PyQt4.QtDeclarative import QDeclarativeView


app = QApplication(sys.argv)

# Create the QML user interface.
view = QDeclarativeView()
view.setSource(QUrl(os.path.join(os.path.dirname(__file__), 'editor.qml')))
view.setResizeMode(QDeclarativeView.SizeRootObjectToView)

# Get the root object of the user interface.  
rootObject = view.rootObject()

# Display the user interface and allow the user to interact with it.
view.setGeometry(100, 100, 1000, 500)
view.show()

app.exec_()
