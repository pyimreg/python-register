""" Digressions in QT """

def searchInspector(search):
    import sys
    from PyQt4.QtGui import QApplication, QDialog
    from dialog import Ui_Dialog

    app = QApplication(sys.argv)
    window = QDialog()
    ui = Ui_Dialog()
    ui.setupUi(window)
    ui.updateList(search)
    window.show()
    sys.exit(app.exec_())

