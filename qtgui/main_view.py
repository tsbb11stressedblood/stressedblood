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
            self.imageLabel.origWidth = self.imagePixmap.width()
            self.imageLabel.origHeight = self.imagePixmap.height()
            print self.imageLabel.origWidth
            print self.imageLabel.origHeight

            self.scrollArea.setWidget(self.imageLabel)

            QtCore.QObject.connect(self.imageLabel, QtCore.SIGNAL('clicked()'), self.scrollClick)

            #self.imageLabel.connect(self.scrollClick)

            #self.scrollArea.verticalScrollBar().setValue(200)

            #self.imageLabel.resize(4000, 4000)

    def checkPlus(self):
        self.toolButton_5.setChecked(False)
        self.pushButton_4.setChecked(False)
        self.pushButton_5.setChecked(False)
        self.imageLabel.drawingROI = False
        self.imageLabel.drawingZoom = True
        self.imageLabel.zoomingOut = False
        #self.scrollArea.verticalScrollBar().setValue(self.imageLabel.scaleY*self.imageLabel.scalePosY)
        #self.scrollArea.horizontalScrollBar().setValue(self.imageLabel.scaleX * self.imageLabel.scalePosX)
        self.imageLabel.repaint()
        #self.imageLabel.updateGeometry()
        #self.imageLabel.setGeometry(QtCore.QRect(0,0,4000,4000))
        #self.scrollArea.updateGeometry()
        #self.scrollArea.setGeometry(QtCore.QRect(0,0,4000,4000))
        #self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0,0,4000,4000))

    def checkMinus(self):
        self.toolButton_6.setChecked(False)
        self.pushButton_4.setChecked(False)
        self.pushButton_5.setChecked(False)

        self.imageLabel.zoomingOut = True
        self.imageLabel.drawingZoom = False
        self.imageLabel.drawingROI = False


    def uncheckPlusAndMinus(self):
        self.toolButton_5.setChecked(False)
        self.toolButton_6.setChecked(False)

    def drawROI(self):
        self.pushButton_5.setChecked(False)
        self.uncheckPlusAndMinus()
        self.imageLabel.drawingROI = True
        self.imageLabel.drawingZoom = False
        self.imageLabel.zoomingOut = False

    def clearAllROI(self):
        self.pushButton_5.setChecked(False)
        self.imageLabel.squares = []
        self.imageLabel.repaint()

    def clearROI(self):
        self.pushButton_4.setChecked(False)
        self.uncheckPlusAndMinus()

    def resetZoom(self):
        self.imageLabel.scaleX = 1.0
        self.imageLabel.scaleY = 1.0
        self.imageLabel.scalePosX = 0
        self.imageLabel.scalePosY = 0
        self.imageLabel.setFixedSize(self.imageLabel.origWidth, self.imageLabel.origHeight)
        self.scrollArea.updateGeometry()
        self.scrollArea.verticalScrollBar().setValue(0)
        self.scrollArea.horizontalScrollBar().setValue(0)
        self.imageLabel.updateGeometry()
        self.imageLabel.repaint()

    def scrollClick(self):
        #print self.imageLabel.scaleX, self.imageLabel.scaleY, self.imageLabel.scalePosX, self.imageLabel.scalePosY
        self.scrollArea.verticalScrollBar().setValue(self.imageLabel.scaleX * self.imageLabel.scalePosY)
        self.scrollArea.horizontalScrollBar().setValue(self.imageLabel.scaleX * self.imageLabel.scalePosX)
        self.scrollArea.updateGeometry()

        self.imageLabel.repaint()


    def openImageView(self):
        print "hmm"
        #window2 = QtGui.QMainWindow()
        #window.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        #ui = Ui_SecondWindow()

        #ui.setupUi(self.window2)

        #self.window2.setEnabled(True)
        self.window2.resize(1024, 768)
        self.window2.show()
        print "hmmmmm???"

    def callb(self):
        print "YEY"
        self.pushButton_4.setCursor(QtGui.QCursor(2))

    def setupUi(self, MainWindow):
        self.MainWindow = MainWindow
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1024, 768)

        ############################################################### WINDOW 2 ###################################

        self.window2 = QtGui.QMainWindow(MainWindow)

        self.window2.setTabShape(QtGui.QTabWidget.Rounded)
        self.centralwidget2 = QtGui.QWidget(self.window2)
        self.centralwidget2.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_2 = QtGui.QGridLayout(self.centralwidget2)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtGui.QLayout.SetMaximumSize)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.ROI_label = QtGui.QLabel(self.centralwidget2)
        self.ROI_label.setObjectName(_fromUtf8("ROI_label"))
        self.horizontalLayout.addWidget(self.ROI_label)
        self.line_2 = QtGui.QFrame(self.centralwidget2)
        self.line_2.setFrameShape(QtGui.QFrame.VLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.horizontalLayout.addWidget(self.line_2)
        self.select_ROI_label = QtGui.QLabel(self.centralwidget2)
        self.select_ROI_label.setObjectName(_fromUtf8("select_ROI_label"))
        self.horizontalLayout.addWidget(self.select_ROI_label, QtCore.Qt.AlignRight)
        self.comboBox_2 = QtGui.QComboBox(self.centralwidget2)
        self.comboBox_2.setObjectName(_fromUtf8("comboBox_2"))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.horizontalLayout.addWidget(self.comboBox_2)
        self.pushButton_4 = QtGui.QPushButton(self.centralwidget2)
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.pushButton_4.clicked.connect(self.callb)

        self.horizontalLayout.addWidget(self.pushButton_4)

        self.line_3 = QtGui.QFrame(self.centralwidget2)
        self.line_3.setFrameShape(QtGui.QFrame.VLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.horizontalLayout.addWidget(self.line_3)
        self.label_show = QtGui.QLabel(self.centralwidget2)
        self.label_show.setObjectName(_fromUtf8("label_show"))
        self.horizontalLayout.addWidget(self.label_show, QtCore.Qt.AlignRight)
        self.comboBox = QtGui.QComboBox(self.centralwidget2)
        self.comboBox.setEditable(False)
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.horizontalLayout.addWidget(self.comboBox)
        self.pushButton = QtGui.QPushButton(self.centralwidget2)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.horizontalLayout.addWidget(self.pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.line = QtGui.QFrame(self.centralwidget2)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.verticalLayout.addWidget(self.line)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setSizeConstraint(QtGui.QLayout.SetMaximumSize)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.image_1 = QtGui.QLabel(self.centralwidget2)
        self.image_1.setMaximumSize(QtCore.QSize(46, 44))
        self.image_1.setObjectName(_fromUtf8("image_1"))
        self.gridLayout.addWidget(self.image_1, 2, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.image_3 = QtGui.QLabel(self.centralwidget2)
        self.image_3.setObjectName(_fromUtf8("image_3"))
        self.gridLayout.addWidget(self.image_3, 2, 2, 1, 1, QtCore.Qt.AlignHCenter)
        self.image_6 = QtGui.QLabel(self.centralwidget2)
        self.image_6.setObjectName(_fromUtf8("image_6"))
        self.gridLayout.addWidget(self.image_6, 3, 2, 1, 1, QtCore.Qt.AlignHCenter)
        self.image_8 = QtGui.QLabel(self.centralwidget2)
        self.image_8.setObjectName(_fromUtf8("image_8"))
        self.gridLayout.addWidget(self.image_8, 4, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.image_9 = QtGui.QLabel(self.centralwidget2)
        self.image_9.setObjectName(_fromUtf8("image_9"))
        self.gridLayout.addWidget(self.image_9, 4, 2, 1, 1, QtCore.Qt.AlignHCenter)
        self.image_2 = QtGui.QLabel(self.centralwidget2)
        self.image_2.setObjectName(_fromUtf8("image_2"))
        self.gridLayout.addWidget(self.image_2, 2, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.image_4 = QtGui.QLabel(self.centralwidget2)
        self.image_4.setObjectName(_fromUtf8("image_4"))
        self.gridLayout.addWidget(self.image_4, 3, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.image_5 = QtGui.QLabel(self.centralwidget2)
        self.image_5.setObjectName(_fromUtf8("image_5"))
        self.gridLayout.addWidget(self.image_5, 3, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.image_7 = QtGui.QLabel(self.centralwidget2)
        self.image_7.setObjectName(_fromUtf8("image_7"))
        self.gridLayout.addWidget(self.image_7, 4, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addLayout(self.gridLayout)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.pushButton_prev = QtGui.QPushButton(self.centralwidget2)
        self.pushButton_prev.setObjectName(_fromUtf8("pushButton_prev"))
        self.horizontalLayout_2.addWidget(self.pushButton_prev)
        self.label_images = QtGui.QLabel(self.centralwidget2)
        self.label_images.setObjectName(_fromUtf8("label_images"))
        self.horizontalLayout_2.addWidget(self.label_images)
        # self.label_page = QtGui.QLabel(self.centralwidget2)
        # self.label_page.setObjectName(_fromUtf8("label_page"))
        # self.horizontalLayout_2.addWidget(self.label_page, QtCore.Qt.AlignVCenter)
        self.pushButton_next = QtGui.QPushButton(self.centralwidget2)
        self.pushButton_next.setObjectName(_fromUtf8("pushButton_next"))
        self.horizontalLayout_2.addWidget(self.pushButton_next)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.window2.setCentralWidget(self.centralwidget2)
        self.statusbar = QtGui.QStatusBar(self.window2)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        self.window2.setStatusBar(self.statusbar)
        self.menubar = QtGui.QMenuBar(self.window2)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 827, 18))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuSettings = QtGui.QMenu(self.menubar)
        self.menuSettings.setObjectName(_fromUtf8("menuSettings"))
        self.window2.setMenuBar(self.menubar)
        self.actionShow_number_of_cells = QtGui.QAction(self.window2)
        self.actionShow_number_of_cells.setObjectName(_fromUtf8("actionShow_number_of_cells"))
        self.menuSettings.addAction(self.actionShow_number_of_cells)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())

        self.retranslateUi2(self.window2)
        QtCore.QMetaObject.connectSlotsByName(self.window2)




        ############################################################### WINDOW 2 ###################################



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



        #self.scrollArea.connect(self.scrollClick)


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
        self.pushButton_3.clicked.connect(self.openImageView)

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

    def retranslateUi2(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.ROI_label.setText(_translate("MainWindow",
                                          "<html><head/><body><p>HL Ratio for selected ROI: <span style=\" font-weight:600;\">1.0</span></p></body></html>",
                                          None))
        self.select_ROI_label.setText(_translate("MainWindow", "Select ROI:", None))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "All", None))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "ROI 1", None))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "ROI 2", None))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "...", None))
        self.pushButton_4.setText(_translate("MainWindow", "OK", None))

        self.label_show.setText(_translate("MainWindow", "Show type of cells:", None))
        self.comboBox.setItemText(0, _translate("MainWindow", "All", None))
        self.comboBox.setItemText(1, _translate("MainWindow", "Heterophils", None))
        self.comboBox.setItemText(2, _translate("MainWindow", "Lymphocytes", None))
        self.comboBox.setItemText(3, _translate("MainWindow", "Monocytes", None))
        self.comboBox.setItemText(4, _translate("MainWindow", "Discarded", None))
        self.pushButton.setText(_translate("MainWindow", "OK", None))
        self.image_1.setText(_translate("MainWindow", "TextLabel", None))
        self.image_3.setText(_translate("MainWindow", "TextLabel", None))
        self.image_6.setText(_translate("MainWindow", "TextLabel", None))
        self.image_8.setText(_translate("MainWindow", "TextLabel", None))
        self.image_9.setText(_translate("MainWindow", "TextLabel", None))
        self.image_2.setText(_translate("MainWindow", "TextLabel", None))
        self.image_4.setText(_translate("MainWindow", "TextLabel", None))
        self.image_5.setText(_translate("MainWindow", "TextLabel", None))
        self.image_7.setText(_translate("MainWindow", "TextLabel", None))
        self.pushButton_prev.setText(_translate("MainWindow", "<-", None))
        self.label_images.setText(_translate("MainWindow", "1-9 of X (Page 1 of Y)", None))
        # self.label_page.setText(_translate("MainWindow", "Page 1 of Y", None))
        self.pushButton_next.setText(_translate("MainWindow", "->", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuSettings.setTitle(_translate("MainWindow", "Settings", None))
        self.actionShow_number_of_cells.setText(_translate("MainWindow", "Show number of cells", None))

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
        self.scaleX = 1.0
        self.scaleY = 1.0
        self.drawingROI = False
        self.drawingZoom = False
        self.zoomingOut = False
        self.origWidth = self.width()
        self.origHeight = self.height()

    def paintEvent(self, event):
        self.setCursor(QtGui.QCursor(QtGui.QCursor(2)))
        self.painter = QtGui.QPainter()
        self.painter.begin(self)
        self.painter.scale(self.scaleX, self.scaleX)

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

        if self.mouseHeldDown and (self.drawingROI or self.drawingZoom):
            self.painter.drawRect(self.origX/self.scaleX, self.origY/self.scaleX, (self.x-self.origX)/self.scaleX, (self.y-self.origY)/self.scaleX)
        for s in self.squares:
            self.painter.setPen(QtGui.QPen(QtGui.QColor(255,0,0)))
            self.painter.drawRect(s[0], s[1], (s[2]-s[0]), (s[3]-s[1]))
        self.painter.end()


    def mousePressEvent(self, QMouseEvent):
        print "mouse!", QMouseEvent
        self.origX = QMouseEvent.x()
        self.origY = QMouseEvent.y()

    def mouseReleaseEvent(self, QMouseEvent):
        self.mouseHeldDown = False
        self.x = QMouseEvent.x()
        self.y = QMouseEvent.y()
        if self.drawingROI:
            self.squares.append([self.origX, self.origY, self.x, self.y])


        if self.drawingZoom:
            self.scalePosX = self.origX / self.scaleX
            self.scalePosY = self.origY / self.scaleX

            if self.x != self.origX:
                self.scaleX *= self.pimg.width() / (self.x - self.origX)
                #if self.drawingZoom:
                    #self.scalePosX = self.origX/self.scaleX
            if self.y != self.origY:
                self.scaleY *= self.pimg.height() / (self.y - self.origY)
                #if self.drawingZoom:
                #    self.scalePosY = self.origY/self.scaleX


            #self.scrollArea.verticalScrollBar().setValue(self.scaleY * self.scalePosY)
            #self.scrollArea.horizontalScrollBar().setValue(self.scaleX * self.scalePosX)

            self.setFixedSize(self.origWidth*self.scaleX, self.origHeight*self.scaleX)
            self.repaint()
            self.emit(QtCore.SIGNAL('clicked()'))

        elif self.zoomingOut:


            self.scaleX /= 2.0
            self.scaleY /= 2.0

            self.setFixedSize(self.origWidth * self.scaleX, self.origHeight * self.scaleX)


            self.repaint()

            #self.scrollArea.verticalScrollBar().setValue(self.scaleY * self.scalePosY / 2.0)
            #self.scrollArea.horizontalScrollBar().setValue(self.scaleX * self.scalePosX / 2.0)

            self.emit(QtCore.SIGNAL('clicked()'))

            self.updateGeometry()
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


class MyWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MyWindow,self).__init__(parent)

        self.win2 = QtGui.QMainWindow()

    def createWin(self):
        self.win2.show()


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    #app.setOverrideCursor(QtGui.QCursor(2))
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

