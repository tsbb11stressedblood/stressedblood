from main_view import Ui_MainWindow
from PyQt4 import QtCore, QtGui



def testtest():
    print "HEJHEJHEJ HEHEHEHE"
    print ui.imageLabel.squares

#if __name__ == "__main__":

import sys
app = QtGui.QApplication(sys.argv)
#app.setOverrideCursor(QtGui.QCursor(2))
MainWindow = QtGui.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow, testtest)
MainWindow.show()
sys.exit(app.exec_())

#print ui.imageLabel.squares


