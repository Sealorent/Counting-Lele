import numpy as np
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap, QImage
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt


import cv2
fname = ""


class UI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("init.ui", self)
        self.setWindowTitle("Image Thresholding")
        self.btnOpenImage.clicked.connect(self.open_file)
        self.btnProses.clicked.connect(self.thres)
        self.btnThreshold.clicked.connect(self.th)
        self.btnRgb.clicked.connect(self.rgb)
        self.btnThin.clicked.connect(self.thin)
        self.labelOpenImage = self.findChild(QLabel, "lbOpenImage")
        self.lbhslImage = self.findChild(QLabel, "lbHasil")
        self.lbJumlah = self.findChild(QLabel, "lbJumlah")
        self.lbBlue = self.findChild(QLabel, "lb_hasil_blue")
        self.lbRed = self.findChild(QLabel, "lb_hasil_red")
        self.lbGreen = self.findChild(QLabel, "lb_hasil_green")
        self.addToolBar(NavigationToolbar(self.Histogram.canvas, self))

    def open_file(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'd:\\Project\\PCV\\Foto', "Image Files (*.jpg *.gif *.bmp *.png *.jpeg *.jfif )")
        pixmap = QPixmap(fname[0]).scaled(300, 300)
        self.labelOpenImage.setPixmap(pixmap)
        self.labelOpenImage.setScaledContents(True)
        self.Histogram.canvas.sumbu1.clear()
        read_img = cv2.imread(fname[0], cv2.IMREAD_COLOR)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([read_img], [i], None, [256], [0, 256])
            self.Histogram.canvas.sumbu1.plot(histr, color=col, linewidth=3.0)
            self.Histogram.canvas.sumbu1.set_ylabel('Y', color='blue')
            self.Histogram.canvas.sumbu1.set_xlabel('X', color='blue')
            self.Histogram.canvas.sumbu1.set_title('Histogram')
            self.Histogram.canvas.sumbu1.set_facecolor('xkcd:wheat')
            self.Histogram.canvas.sumbu1.grid()
        self.Histogram.canvas.draw()

    def thres(self):
        # rgb_image = cv2.imread(fname[0], cv2.COLOR_BGR2RGB)
        # h, w, ch = rgb_image.shape
        # bytes_per_line = ch * w
        # convert_to_Qt_format = QImage(
        #     rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # p = convert_to_Qt_format
        # q = QPixmap.fromImage(p).scaled(300, 300)
        # # self.lbhslImage.setPixmap(q)
        img = cv2.imread(fname[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
        kernal = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(thresh, kernal, iterations=1)
        contours, hierarchy = cv2.findContours(
            dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.lbJumlah.setText("%i" % len(contours))

    def rgb(self):
        img1 = cv2.imread(fname[0])
        img_array = np.array(img1)
        r, g, b = img_array.shape
        self.lbBlue.setText(str(b))
        self.lbGreen.setText(str(g))
        self.lbRed.setText(str(r))

    def th(self):
        image1 = cv2.imread(fname[0])
        img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        _, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
        qmap = QImage(thresh2.data, thresh2.shape[1],
                      thresh2.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qmap).scaled(340, 328)
        self.lbhslImage.setPixmap(pixmap)

    def thin(self):
        image1 = cv2.imread(fname[0], cv2.IMREAD_GRAYSCALE)


        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(image1, cv2.MORPH_OPEN, kernel)
        blur = cv2.GaussianBlur(opening, (1, 1), 0)
        ret, th4 = cv2.threshold(
            blur, 120, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        skel = (skeletonize(th4//255) * 255).astype(np.uint8)
        qmap = QImage(skel.data, skel.shape[1],
                      skel.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qmap).scaled(350,350)
        self.lbhslImage.setPixmap(pixmap)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = UI()
    w.show()
    sys.exit(app.exec_())
