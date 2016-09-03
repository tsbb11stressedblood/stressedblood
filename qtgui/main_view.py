# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_view.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

import matplotlib.image as mpimg

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)



class Ui_MainWindow(object):

    def exitEverything(self):
        print "HEHE"

    def showDialog(self):
        fname = QtGui.QFileDialog.getOpenFileName(MainWindow, 'Open image...', '')
        print "fname:", fname


        if '.png' in fname:
            self.imagePixmap = QtGui.QPixmap(fname)
            self.imageLabel = ImageArea(fname)

            self.imageLabel.setPixmap(self.imagePixmap)
            self.scrollArea.setWidget(self.imageLabel)
            #self.scrollArea.verticalScrollBar().setValue(200)

            #self.imageLabel.resize(4000, 4000)

    def checkPlus(self):
        self.toolButton_5.setChecked(False)
        self.pushButton_4.setChecked(False)
        self.pushButton_5.setChecked(False)
        self.imageLabel.drawingROI = False
        self.imageLabel.drawingZoom = True
        self.scrollArea.verticalScrollBar().setValue(self.imageLabel.scaleY*self.imageLabel.scalePosY)
        self.scrollArea.horizontalScrollBar().setValue(self.imageLabel.scaleX * self.imageLabel.scalePosX)
        #self.imageLabel.updateGeometry()
        #self.imageLabel.setGeometry(QtCore.QRect(0,0,4000,4000))
        #self.scrollArea.updateGeometry()
        #self.scrollArea.setGeometry(QtCore.QRect(0,0,4000,4000))
        #self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0,0,4000,4000))

    def checkMinus(self):
        self.toolButton_6.setChecked(False)
        self.pushButton_4.setChecked(False)
        self.pushButton_5.setChecked(False)
        self.imageLabel.updateGeometry()

    def uncheckPlusAndMinus(self):
        self.toolButton_5.setChecked(False)
        self.toolButton_6.setChecked(False)

    def drawROI(self):
        self.pushButton_5.setChecked(False)
        self.uncheckPlusAndMinus()
        self.imageLabel.drawingROI = True

    def clearAllROI(self):
        self.pushButton_5.setChecked(False)
        self.imageLabel.squares = []
        self.imageLabel.repaint()

    def clearROI(self):
        self.pushButton_4.setChecked(False)
        self.uncheckPlusAndMinus()

    def resetZoom(self):
        self.imageLabel.scaleX = 1
        self.imageLabel.scaleY = 1
        self.imageLabel.scalePosX = 0
        self.imageLabel.scalePosY = 0
        self.imageLabel.setFixedSize(self.imageLabel.width(), self.imageLabel.height())
        self.scrollArea.updateGeometry()
        self.scrollArea.verticalScrollBar().setValue(0)
        self.scrollArea.horizontalScrollBar().setValue(0)
        self.imageLabel.updateGeometry()
        self.imageLabel.repaint()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1024, 768)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))

        self.scrollArea = QtGui.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        #self.scrollArea.setGeometry(QtCore.QRect(0,0,4000,4000))
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))

        #self.scrollAreaWidgetContents = QtGui.QWidget()
        #self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 861, 487))
        #self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 4000, 4000))
        #self.scrollAreaWidgetContents.setObjectName(_fromUtf8("scrollAreaWidgetContents"))
        #self.scrollArea.setWidget(self.scrollAreaWidgetContents)


        self.verticalLayout_3.addWidget(self.scrollArea)
        self.verticalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setEnabled(True)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton.clicked.connect(self.showDialog)
        self.textEdit = QtGui.QTextEdit()

        self.horizontalLayout.addWidget(self.pushButton)
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.toolButton_6 = QtGui.QToolButton(self.centralwidget)
        self.toolButton_6.setAutoRaise(False)
        self.toolButton_6.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_6.setObjectName(_fromUtf8("toolButton_6"))
        self.toolButton_6.setCheckable(True)
        self.toolButton_6.clicked.connect(self.checkPlus)

        self.horizontalLayout_4.addWidget(self.toolButton_6)
        self.toolButton_5 = QtGui.QToolButton(self.centralwidget)
        self.toolButton_5.setObjectName(_fromUtf8("toolButton_5"))
        self.toolButton_5.setCheckable(True)
        self.toolButton_5.clicked.connect(self.checkMinus)

        self.horizontalLayout_4.addWidget(self.toolButton_5)
        self.pushButton_2 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_2.clicked.connect(self.resetZoom)

        self.horizontalLayout_4.addWidget(self.pushButton_2)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.ROI_layout = QtGui.QHBoxLayout()
        self.ROI_layout.setObjectName(_fromUtf8("ROI_layout"))
        self.pushButton_4 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_4.setCheckable(True)
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.pushButton_4.clicked.connect(self.drawROI)

        self.ROI_layout.addWidget(self.pushButton_4)
        self.pushButton_5 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_5"))
        self.pushButton_5.setCheckable(True)
        self.pushButton_5.clicked.connect(self.clearROI)

        self.ROI_layout.addWidget(self.pushButton_5)
        self.pushButton_21 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_21.setObjectName(_fromUtf8("pushButton_21"))
        self.ROI_layout.addWidget(self.pushButton_21)
        self.pushButton_21.clicked.connect(self.clearAllROI)

        self.verticalLayout_4.addLayout(self.ROI_layout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem)
        self.pushButton_3 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.verticalLayout.addWidget(self.pushButton_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        #########self.menubar.setGeometry(QtCore.QRect(0, 0, 885, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName(_fromUtf8("menuHelp"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_image = QtGui.QAction(MainWindow)
        self.actionLoad_image.setCheckable(False)
        self.actionLoad_image.setObjectName(_fromUtf8("actionLoad_image"))
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))

        self.actionAbout = QtGui.QAction(MainWindow)
        self.actionAbout.setObjectName(_fromUtf8("actionAbout"))
        self.menuFile.addAction(self.actionLoad_image)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuFile.addSeparator()
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.pushButton.setText(_translate("MainWindow", "Open image...", None))
        self.toolButton_6.setText(_translate("MainWindow", "+", None))
        self.toolButton_5.setText(_translate("MainWindow", "-", None))
        self.pushButton_2.setText(_translate("MainWindow", "Reset zoom", None))
        self.pushButton_4.setText(_translate("MainWindow", "Draw ROI", None))
        self.pushButton_5.setText(_translate("MainWindow", "Clear ROI", None))
        self.pushButton_21.setText(_translate("MainWindow", "Clear all ROI", None))
        self.pushButton_3.setText(_translate("MainWindow", "Run", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuHelp.setTitle(_translate("MainWindow", "Help", None))
        self.actionLoad_image.setText(_translate("MainWindow", "Open image...", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))
        self.actionAbout.setText(_translate("MainWindow", "About...", None))

class ImageArea(QtGui.QLabel):

    def __init__(self, img):
        QtGui.QLabel.__init__(self)
        self.image = QtGui.QImage(img)
        self.pimg = QtGui.QPixmap(img)
        #self.image = self.image.scaled(2000, 2000, 1)


        self.squares = []
        self.x = 0
        self.y = 0
        self.origX = 0
        self.origY = 0
        self.mouseHeldDown = False
        self.scalePosX = 0
        self.scalePosY = 0
        self.scaleX = 1
        self.scaleY = 1
        self.drawingROI = False
        self.drawingZoom = False

    def paintEvent(self, event):
        self.painter = QtGui.QPainter()
        self.painter.begin(self)
        self.painter.scale(self.scaleX, self.scaleY)

        #self.painter.drawImage(self.image.rect(), self.image)
        #rect = self.painter.matrix().inverted().mapRect(event.rect()).adjusted(-1, -1, 1, 1)

        #self.painter.matrix()
        #self.painter.drawImage(rect, self.image)

        #self.painter.drawImage(QtCore.QPointF(0, 0), self.image)
        self.painter.drawPixmap(0, 0, self.pimg)
        #self.painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
        if self.drawingROI:
            self.painter.setPen(QtGui.QPen(QtGui.QColor(255,0,0)))
        if self.drawingZoom:
            self.painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))

        if self.mouseHeldDown:
            self.painter.drawRect(self.origX, self.origY, self.x-self.origX, self.y-self.origY)
        for s in self.squares:
            self.painter.setPen(QtGui.QPen(QtGui.QColor(255,0,0)))
            self.painter.drawRect(s[0], s[1], s[2]-s[0], s[3]-s[1])
        self.painter.end()


    def mousePressEvent(self, QMouseEvent):
        print "mouse!", QMouseEvent
        self.origX = QMouseEvent.x()
        self.origY = QMouseEvent.y()


    def mouseReleaseEvent(self, QMouseEvent):
        self.x = QMouseEvent.x()
        self.y = QMouseEvent.y()
        if self.drawingROI:
            self.squares.append([self.origX, self.origY, self.x, self.y])
        self.mouseHeldDown = False

        if self.x != self.origX:
            self.scaleX = self.pimg.width() / (self.x - self.origX)
            if self.drawingZoom:
                self.scalePosX = self.origX
        if self.y != self.origY:
            self.scaleY = self.pimg.height() / (self.y - self.origY)
            if self.drawingZoom:
                self.scalePosY = self.origY

        self.setFixedSize(self.width()*self.scaleX, self.height()*self.scaleY)
        self.repaint()


    def mouseMoveEvent(self, QMouseEvent):
        self.mouseHeldDown = True
        self.x = QMouseEvent.x()
        self.y = QMouseEvent.y()
        self.repaint()

    def resizeEvent(self, *args, **kwargs):
        pass
        #self.updateGeometry()
        #self.setGeometry(QtCore.QRect(0,0,4000,4000))

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    #app.setOverrideCursor(QtGui.QCursor(2))
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



