# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_view.ui'
#
# Created by: PyQt4 UI code generator 4.11.4

#import imp
#print imp.find_module("cv2")
#import openslide

from PyQt4 import QtCore, QtGui

#from cnn import sliding_window
from cnn import sliding_window
from cnn import extract_cells
#from extract_cells import extract_cells

import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from scipy.misc import imresize
import scipy
import scipy.ndimage

from matplotlib import pyplot as plt

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





#squares = []


class Ui_MainWindow(object):
    def __init__(self):
        self.ratio = 1.0

    def exitEverything(self):
        print "HEHE"

    def showDialog(self):
        fname = QtGui.QFileDialog.getOpenFileName(self.MainWindow, 'Open image...', '')
        #fname = QtGui.QFileDialog.getOpenFileName(self, 'Open image...', '')
        print "fname:", fname

        if '.png' in fname or '.tif' in fname:
            self.image = cv2.imread(str(fname), 1)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGBA)

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
        self.imageLabel.clearingROI = False


    def uncheckPlusAndMinus(self):
        self.toolButton_5.setChecked(False)
        self.toolButton_6.setChecked(False)

    def drawROI(self):
        self.pushButton_5.setChecked(False)
        self.uncheckPlusAndMinus()
        self.imageLabel.drawingROI = True
        self.imageLabel.drawingZoom = False
        self.imageLabel.zoomingOut = False
        self.imageLabel.clearingROI = False

    def clearAllROI(self):
        self.pushButton_5.setChecked(False)
        del self.imageLabel.squares[:]
        #del squares[:]
        self.imageLabel.repaint()

    def clearROI(self):
        self.pushButton_4.setChecked(False)
        self.imageLabel.clearingROI = True
        self.imageLabel.drawingROI = False
        self.imageLabel.drawingZoom = False
        self.imageLabel.zoomingOut = False
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
        self.pageNum = 0
        self.numPages = 0
        print "hmm"
        #window2 = QtGui.QMainWindow()
        #window.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        #ui = Ui_SecondWindow()

        #ui.setupUi(self.window2)

        #s = squares[0]

        s = self.imageLabel.squares[0]

        #ROI_image = self.image[s[1]:s[1]+s[3], s[0]:s[0]+s[2]]

        #ROI_image = self.image[0:200, 0:200]

        ROI_image = self.image[int(s[1]):int(s[3]), int(s[0]):int(s[2])]

        print "roi image shape:", ROI_image.shape

        w = ROI_image.shape[0]
        h = ROI_image.shape[1]

        numx = 1
        numy = 1

        if w > 512:
            numx = w/512 + 1
        if h > 512:
            numy = h/512 + 1

        print "numx numy:", numx, numy

        #self.painter.drawRect(s[0], s[1], (s[2] - s[0]), (s[3] - s[1]))

        #heat_map = sliding_window.get_heatmap(image=np.transpose(ROI_image/np.float(256.0), axes=(2, 1, 0)), stride=4)

        stitched_heatmap = np.zeros((64 * numx, 64 * numy, 4))

        for i in range(numx):
            for j in range(numy):
                heat_map = sliding_window.get_heatmapp(image=np.transpose(ROI_image[i*512:i*512+512, j*512:j*512+512, :] / np.float(256.0), axes=(2, 1, 0)))

                stitched_heatmap[i*64:i*64+64, j*64:j*64+64, :] = heat_map

                #heat_map = sliding_window.get_heatmap(image=ROI_image, stride=8)
                #heat_map = sliding_window.get_heatmap(image=self.image, stride=4)

                # print(np.max(heat_map))
                print "heatmap done?", ROI_image.shape
                #plt.figure("untouched heatmap")

        plt.figure('stitched heatmap')
        plt.imshow(stitched_heatmap[:,:,0:3])


        #scaled_heatmap = imresize(stitched_heatmap, 10.0)
        #scaled_heatmap = imresize(stitched_heatmap*255.0, 900)
        #scaled_heatmap = scipy.ndimage.interpolation.zoom(stitched_heatmap, 9.0)
        scaled_heatmap = cv2.resize(stitched_heatmap*255.0, (int(stitched_heatmap.shape[1]*8), int(stitched_heatmap.shape[0]*8)))

        #non = lambda s: s if s < 0 else None
        #mom = lambda s: max(0, s)
        #shifted = np.zeros_like(scaled_heatmap)
        #shifted[mom(32):non(32), mom(32):non(32)] = scaled_heatmap[mom(-32):non(-32), mom(-32):non(-32)]

        #plt.figure('offset stitched and scaled heatmap')
        #plt.imshow(shifted[:, :, 0:3])

        plt.figure('resized heatmap')
        plt.imshow(np.uint8(scaled_heatmap[:, :, 0:3]))
        plt.show()

        print "scaled heatmap: ", scaled_heatmap.shape
        print "orig image: ", ROI_image.shape

        scaled_and_cropped_heatmap = np.zeros_like(ROI_image)
        scaled_and_cropped_heatmap = scaled_heatmap[0:ROI_image.shape[0], 0:ROI_image.shape[1], :]

        plt.figure('scaled, resized and cropped heatmap')
        plt.imshow(np.uint8(scaled_and_cropped_heatmap[:, :, 0:3]))
        plt.show()


        self.red_cells, self.red_cells_confidence, self.green_cells, self.green_cells_confidence =\
            extract_cells.extract_cells(ROI_image, scaled_and_cropped_heatmap)

        self.red_cells_confidence = sorted(self.red_cells_confidence)
        self.green_cells_confidence = sorted(self.green_cells_confidence)

        print ("red conf,", self.red_cells_confidence)
        print ("green conf", self.green_cells_confidence)

        #sort after confidence
        self.red_cells = [x for (y, x) in sorted(zip(self.red_cells_confidence, self.red_cells))]
        self.green_cells = [x for (y, x) in sorted(zip(self.green_cells_confidence, self.green_cells))]

        self.totcells = self.red_cells + self.green_cells

        self.ratio = 0.0

        if len(self.red_cells) > 0:
            self.ratio = float(len(self.green_cells)) / len(self.red_cells)

        self.set_ratio_text()

        self.numcells = len(self.totcells)
        self.numPages = int(math.ceil(self.numcells / 9.0))

        self.updateLabel()

        #for i,c in enumerate(self.totcells):
        #    if i < 9:
        #        cc = cv2.cvtColor(c, cv2.COLOR_BGRA2RGBA)
        #        height, width, channel = cc.shape  # BGR RGB
        #        qImg = QtGui.QImage(cc.tostring(), width, height, QtGui.QImage.Format_ARGB32)
        #        self.images[i].setPixmap(QtGui.QPixmap(qImg))

        self.updateImages(self.totcells)


        #self.window2.setEnabled(True)
        self.window2.resize(1024, 768)
        self.window2.show()
        print "hmmmmm???"


    def callb(self):
        print "YEY"
        self.pushButton_4.setCursor(QtGui.QCursor(2))

    def nextPage(self):
        currtext = self.comboBox.currentText()
        if currtext == "Heterophils":
            numcells = len(self.green_cells)
        elif currtext == "Lymphocytes":
            numcells = len(self.red_cells)
        else:
            numcells = self.numcells

        if self.pageNum*9 + 9 < numcells:
            self.pageNum += 1
        self.updateAllImages()
        self.updateLabel()

    def updateAllImages(self):
        currtext = self.comboBox.currentText()
        if currtext == "Heterophils":
            self.updateImages(self.green_cells)
        elif currtext == "Lymphocytes":
            self.updateImages(self.red_cells)
        else:
            self.updateImages(self.totcells)


    def updateLabel(self):
        currtext = self.comboBox.currentText()
        if currtext == "Heterophils":
            numPages = int(math.ceil(len(self.green_cells) / 9.0))
            numcells = len(self.green_cells)
        elif currtext == "Lymphocytes":
            numPages = int(math.ceil(len(self.red_cells) / 9.0))
            numcells = len(self.red_cells)
        else:
            numcells = len(self.totcells)
            numPages = int(math.ceil(len(self.totcells) / 9.0))


        fromm = 9*self.pageNum+1
        to = 9*self.pageNum+9

        if to > numcells:
            to = numcells

        self.label_images.setText(_translate("MainWindow", str(fromm) + " to " + str(to) + " of " +
                                             str(numcells) + " (Page " + str(self.pageNum+1) + " of " +
                                             str(numPages) + ")", None))

    def updateImages(self, list):

        for i in range(9):
            self.images[i].clear()

        for i, c in enumerate(list):
            #if i < 9 + 9 * self.pageNum and i >= 9 * self.pageNum and i < self.numcells:
            cc = cv2.cvtColor(c, cv2.COLOR_BGRA2RGBA)
            #cc = cv2.rectangle(cc, (0, 0), (5, 5), (0, 0, 255, 200), -1)

            height, width, channel = cc.shape  # BGR RGB
            qImg = QtGui.QImage(cc.tostring(), width, height, QtGui.QImage.Format_ARGB32)
            #qImg = QtGui.QImage(cc.tostring(), height, width, QtGui.QImage.Format_ARGB32)

            print "i:", i
            print "pageNum", self.pageNum

            if i-9*self.pageNum < 9 and i-9*self.pageNum >= 0:
                self.images[i - 9 * self.pageNum].setPixmap(QtGui.QPixmap(qImg))


    def prevPage(self):
        if self.pageNum > 0:
            self.pageNum -= 1
        self.updateAllImages()
        self.updateLabel()

    def on_context_0(self, point):
        self.popMenu[0].exec_(self.images[0].mapToGlobal(point))

    def on_context_1(self, point):
        self.popMenu[1].exec_(self.images[1].mapToGlobal(point))

    def on_context_2(self, point):
        self.popMenu[2].exec_(self.images[2].mapToGlobal(point))

    def on_context_3(self, point):
        self.popMenu[3].exec_(self.images[3].mapToGlobal(point))

    def on_context_4(self, point):
        self.popMenu[4].exec_(self.images[4].mapToGlobal(point))

    def on_context_5(self, point):
        self.popMenu[5].exec_(self.images[5].mapToGlobal(point))

    def on_context_6(self, point):
        self.popMenu[6].exec_(self.images[6].mapToGlobal(point))

    def on_context_7(self, point):
        self.popMenu[7].exec_(self.images[7].mapToGlobal(point))

    def on_context_8(self, point):
        self.popMenu[8].exec_(self.images[8].mapToGlobal(point))

    def testtest(self, num):
        #self.images = []
        print "SUCCESS!", num

    def setupUi(self, MainWindow, runCallback):
        self.MainWindow = MainWindow
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1024, 768)
        self.numcells = 0

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
        self.comboBox.addItem(_fromUtf8("All"))
        self.comboBox.addItem(_fromUtf8("Heterophils"))
        self.comboBox.addItem(_fromUtf8("Lymphocytes"))
        #self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8("Discarded"))
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

        self.images = []
        self.images.append(QtGui.QLabel(self.centralwidget2))
        self.images[0].setMaximumSize(QtCore.QSize(46, 44))
        self.images[0].setObjectName(_fromUtf8("image_1"))
        self.gridLayout.addWidget(self.images[0], 2, 0, 1, 1, QtCore.Qt.AlignHCenter)



        #self.tableWidget = QtGui.QTableWidget()
        #self.tableWidget.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        #self.tableWidget.show()


        #self.images[0]


        self.images.append(QtGui.QLabel(self.centralwidget2))
        self.images[1].setObjectName(_fromUtf8("image_2"))
        self.gridLayout.addWidget(self.images[1], 2, 1, 1, 1, QtCore.Qt.AlignHCenter)

        self.images.append(QtGui.QLabel(self.centralwidget2))
        self.images[2].setObjectName(_fromUtf8("image_3"))
        self.gridLayout.addWidget(self.images[2], 2, 2, 1, 1, QtCore.Qt.AlignHCenter)

        self.images.append(QtGui.QLabel(self.centralwidget2))
        self.images[3].setObjectName(_fromUtf8("image_4"))
        self.gridLayout.addWidget(self.images[3], 3, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.images.append(QtGui.QLabel(self.centralwidget2))
        self.images[4].setObjectName(_fromUtf8("image_5"))
        self.gridLayout.addWidget(self.images[4], 3, 1, 1, 1, QtCore.Qt.AlignHCenter)

        self.images.append(QtGui.QLabel(self.centralwidget2))
        self.images[5].setObjectName(_fromUtf8("image_6"))
        self.gridLayout.addWidget(self.images[5], 3, 2, 1, 1, QtCore.Qt.AlignHCenter)

        self.images.append(QtGui.QLabel(self.centralwidget2))
        self.images[6].setObjectName(_fromUtf8("image_7"))
        self.gridLayout.addWidget(self.images[6], 4, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.images.append(QtGui.QLabel(self.centralwidget2))
        self.images[7].setObjectName(_fromUtf8("image_8"))
        self.gridLayout.addWidget(self.images[7], 4, 1, 1, 1, QtCore.Qt.AlignHCenter)

        self.images.append(QtGui.QLabel(self.centralwidget2))
        self.images[8].setObjectName(_fromUtf8("image_9"))
        self.gridLayout.addWidget(self.images[8], 4, 2, 1, 1, QtCore.Qt.AlignHCenter)


        self.popMenu = []
        point = QtCore.QPoint()

        for i in range(9):
            self.images[i].setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

            tmp = QtGui.QMenu(self.window2)
            tmp.addAction(QtGui.QAction('Move to Lymphocytes', self.window2))
            tmp.addAction(QtGui.QAction('Move to Heterophils', self.window2))
            tmp.addSeparator()
            remaction = QtGui.QAction('Remove', self.window2)
            #remaction.triggered.connect(lambda num='test': self.testtest(num))
            remaction.connect(remaction, QtCore.SIGNAL('triggered()'), lambda num=i: self.testtest(num))
            tmp.addAction(remaction)


            self.popMenu.append(tmp)

        self.images[0].customContextMenuRequested.connect(self.on_context_0)
        self.images[1].customContextMenuRequested.connect(self.on_context_1)
        self.images[2].customContextMenuRequested.connect(self.on_context_2)
        self.images[3].customContextMenuRequested.connect(self.on_context_3)
        self.images[4].customContextMenuRequested.connect(self.on_context_4)
        self.images[5].customContextMenuRequested.connect(self.on_context_5)
        self.images[6].customContextMenuRequested.connect(self.on_context_6)
        self.images[7].customContextMenuRequested.connect(self.on_context_7)
        self.images[8].customContextMenuRequested.connect(self.on_context_8)




        self.verticalLayout.addLayout(self.gridLayout)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.pushButton_prev = QtGui.QPushButton(self.centralwidget2)
        self.pushButton_prev.setObjectName(_fromUtf8("pushButton_prev"))
        self.pushButton_prev.clicked.connect(self.prevPage)

        self.horizontalLayout_2.addWidget(self.pushButton_prev)
        self.label_images = QtGui.QLabel(self.centralwidget2)
        self.label_images.setObjectName(_fromUtf8("label_images"))
        self.horizontalLayout_2.addWidget(self.label_images)
        # self.label_page = QtGui.QLabel(self.centralwidget2)
        # self.label_page.setObjectName(_fromUtf8("label_page"))
        # self.horizontalLayout_2.addWidget(self.label_page, QtCore.Qt.AlignVCenter)
        self.pushButton_next = QtGui.QPushButton(self.centralwidget2)
        self.pushButton_next.setObjectName(_fromUtf8("pushButton_next"))
        self.pushButton_next.clicked.connect(self.nextPage)


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
        if runCallback:
            self.pushButton_3.clicked.connect(runCallback)
        else:
            self.pushButton_3.clicked.connect(self.openImageView)
        #if external:
        #    self.pushButton_3.clicked.connect(self.returnROIs)
        #else:
        #    self.pushButton_3.clicked.connect(self.openImageView)


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


    def returnROIs(self):
        #print self.imageLabel.squares
        return self.imageLabel.squares


    def showCells(self):
        #self.comboBox.update()
        print "hehe, ", self.comboBox.currentText()
        currtext = self.comboBox.currentText()
        if currtext == "Heterophils":
            self.updateImages(self.green_cells)
        elif currtext == "Lymphocytes":
            self.updateImages(self.red_cells)
        else:
            self.updateImages(self.totcells)
        self.updateLabel()
        self.pageNum = 0


    def updateComboBox(self):
        pass
        #print "HEHEHEHEHEHEHEHEHE!!!!!!", self.comboBox.currentText()
        #self.comboBox.update()
        #self.pushButton.emit(QtCore.SIGNAL('clicked()'))
        #self.pushButton.connect(self.pushButton, QtCore.SIGNAL('clicked()'),
        #                        lambda num=self.comboBox.currentText(): self.showCells(num))


    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("Select ROI(s)", "Select ROI(s)", None))
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
        MainWindow.setWindowTitle(_translate("MainWindow", "Results", None))
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

        #self.comboBox.addItem("All")
        #self.comboBox.addItem("Heterophils")

        #self.comboBox.addAction(QtGui.QAction('All', self.comboBox))
        #self.comboBox.addAction(QtGui.QAction('Heterophils', self.comboBox))
        #self.comboBox.setItemText(0, _translate("MainWindow", "All", None))
        #self.comboBox.setItemText(1, _translate("MainWindow", "Heterophils", None))

        #self.comboBox.setItemText(2, _translate("MainWindow", "Lymphocytes", None))
        #self.comboBox.setItemText(3, _translate("MainWindow", "Monocytes", None))
        #self.comboBox.setItemText(4, _translate("MainWindow", "Discarded", None))
        self.pushButton.setText(_translate("MainWindow", "OK", None))
        #self.comboBox.currentIndex()

        #self.pushButton.connect(self.pushButton, QtCore.SIGNAL('clicked()'),
        #                        lambda num=self.comboBox.currentText(): self.showCells(num))

        self.pushButton.connect(self.pushButton, QtCore.SIGNAL('clicked()'),
                                self.showCells)

        #self.comboBox.blockSignals(False)

        #self.comboBox.connect(self.comboBox, QtCore.SIGNAL('currentIndexChanged(const QString&)'), #also activated()?
        #                      self.updateComboBox)

        self.images[0].setText(_translate("MainWindow", "", None))
        self.images[1].setText(_translate("MainWindow", "", None))
        self.images[2].setText(_translate("MainWindow", "", None))
        self.images[3].setText(_translate("MainWindow", "", None))
        self.images[4].setText(_translate("MainWindow", "", None))
        self.images[5].setText(_translate("MainWindow", "", None))
        self.images[6].setText(_translate("MainWindow", "", None))
        self.images[7].setText(_translate("MainWindow", "", None))
        self.images[8].setText(_translate("MainWindow", "", None))
        self.pushButton_prev.setText(_translate("MainWindow", "<-", None))

        self.label_images.setText(_translate("MainWindow", "1-9 of X (Page 1 of " + str(self.numcells) + ")", None))
        # self.label_page.setText(_translate("MainWindow", "Page 1 of Y", None))
        self.pushButton_next.setText(_translate("MainWindow", "->", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuSettings.setTitle(_translate("MainWindow", "Settings", None))
        self.actionShow_number_of_cells.setText(_translate("MainWindow", "Show number of cells", None))


    def set_ratio_text(self):
        self.ROI_label.setText(_translate("MainWindow",
                                              "<html><head/><body><p>HL Ratio for selected ROI: <span style=\" font-weight:600;\">" + str(self.ratio) + "</span></p></body></html>",
                                              None))

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
        self.clearingROI = False
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
        self.scalePosX = self.origX / self.scaleX
        self.scalePosY = self.origY / self.scaleX

    def mouseReleaseEvent(self, QMouseEvent):
        self.mouseHeldDown = False
        self.x = QMouseEvent.x()
        self.y = QMouseEvent.y()
        print "mouse release!"

        self.scalePosX = self.origX / self.scaleX
        self.scalePosY = self.origY / self.scaleX

        if self.drawingROI:
            self.squares.append([self.origX/self.scaleX, self.origY/self.scaleX, self.x/self.scaleX, self.y/self.scaleX])


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

        elif self.clearingROI:
            self.squares = [s for s in self.squares if not (self.scalePosX > s[0] and self.scalePosY > s[1]
                    and self.scalePosX < s[2] and self.scalePosY < s[3])]
            #self.squares = []
            print self.squares
            #print tokeep
            #self.squares = tokeep
            self.updateGeometry()
            self.repaint()
            #self.clearingROI = False

            #for i, s in enumerate(self.squares):
            #    if self.scalePosX > s[0] and self.scalePosX < s[1]
            #        and self.scalePosY > s[3] and self.scalePosY < s[4]:
            #        del



    def mouseMoveEvent(self, QMouseEvent):
        self.mouseHeldDown = True
        self.x = QMouseEvent.x()
        self.y = QMouseEvent.y()
        self.repaint()
        print(self.origX-self.x, self.origY-self.y)

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
    ui.setupUi(MainWindow, None)
    MainWindow.show()
    sys.exit(app.exec_())

