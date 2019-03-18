#Gender detection
Alt-H2

Download python3.x.x  [https://www.python.org/downloads/].

Download tensorflow  [https://www.tensorflow.org/].


> pip install tensorflow_gpu==1.9.0.
> pip install matplotlib.
> pip install opencv-contrib-python.
> pip install PyQt5.
- 
----------

| Female        |Male           | 
| ------------- |:-------------:| 
| class 0       | class 1       | 


```import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Conv2D,MaxPool2D,Dense
from keras.preprocessing import image
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
import os,sys,cv2,time

#create data from ImageDataGenerator
image_gen=ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            rescale=1/255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')
                            
                            
after resize all image in subfolder

python resize.py --input folder -o output folder



##test model
img=cv2.imread('picture/woman1.jpg',1)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
_file="picture/woman1.jpg"
images=image.load_img(_file,target_size=(150,150))
images=image.img_to_array(images)
images=np.expand_dims(images,axis=0)
images=images/255
test_model=load_model('models/male_female_model.h5')
prediction=test_model.predict_classes(images)
if prediction==0:
    img =cv2.putText(img=np.copy(img), text="Woman", org=(10,50),fontFace=2, fontScale=0.75, color=(0,0,255), thickness=1)
elif prediction==1:
    img =cv2.putText(img=np.copy(img), text="Man", org=(10,50),fontFace=2, fontScale=0.75, color=(0,0,255), thickness=1)
plt.imshow(img)
plt.show()

result: 
[logo]:[https://github.com/RashadGarayev/genderDetectionKeras/blob/master/picture/Screenshot%2003-18-2019%2013.11.41.jpg]




```



## train model ##

`python train.py`

----------

## Real-time gender classification ##

`python deep.py`



