
from PyQt5.QtWidgets import QApplication,QDialog,QWidget
from PyQt5.uic import loadUi
from PyQt5 import *
import numpy as np
import sys,os,cv2,datetime
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal

class Window(QDialog):
    def __init__(self):
        """-----------------------------"""
        super(Window,self).__init__()
        loadUi('gui.ui',self)
        self.image=None
        self.startButton.clicked.connect(self.startdetect)
        
    def startdetect(self):
        self.capture=cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.timer=QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(4)
    def update_frame(self):
        ret,self.image=self.capture.read()
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('Cascade/haarcascade-frontalface-default.xml')
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.2,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(self.image,(x,y),(x+w,y+h),(255,0,0),2)
            color_face = gray[y:y+h,x:x+w]
            cropped=cv2.imwrite('picture/example.jpg',color_face)
            new_model=load_model('models/male_female_model.h5')
            _file='picture/example.jpg'
            
            _image=image.load_img(_file,target_size=(150,150))
            
            _image=image.img_to_array(_image)
            
            _image=np.expand_dims(_image,axis=0)
            
            _image=_image/255
            
            
            prediction=new_model.predict_classes(_image)
            if prediction ==0:
                cv2.putText(self.image, "Woman", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                self.listWidget.addItem('Woman')
            elif prediction == 1:
                cv2.putText(self.image, "Man", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                self.listWidget.addItem('Man')
            else:
                cv2.putText(self.image, "Face detect", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                self.listWidget.addItem('Face detect')
            
        
        self.displayImage(self.image,1)
    def displayImage(self,img,windows=1):
        qformat=QtGui.QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QtGui.QImage.Format_RGBA8888
            else:
                qformat=QtGui.QImage.Format_RGB888
        outimage=QtGui.QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outimage=outimage.rgbSwapped()
        if windows==1:
            self.imglabel.setPixmap(QtGui.QPixmap.fromImage(outimage))
            


if __name__=='__main__':
    app=QApplication(sys.argv)
    window=Window()
    window.setWindowTitle("LabVision")
    window.show()
    sys.exit(app.exec_())
