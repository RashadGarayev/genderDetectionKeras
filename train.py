
# coding: utf-8

# In[1]:


import os,sys,cv2,time


# In[2]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Conv2D,MaxPool2D,Dense
from keras.preprocessing import image
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from PIL import Image

image_gen=ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            rescale=1/255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')

image_gen.flow_from_directory('data/train/')

model=Sequential()

#model 

input_shape=(150,150,3)

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(150,150,3),
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(150,150,3),
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(150,150,3),
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

batch_size=16
train_image_gen=image_gen.flow_from_directory('data/train/',
                                             target_size=input_shape[:2],
                                             batch_size=batch_size,
                                             class_mode='binary')
test_image_gen=image_gen.flow_from_directory('data/test/',
                                             target_size=input_shape[:2],
                                             batch_size=batch_size,
                                             class_mode='binary')


print(train_image_gen.class_indices #test class)

#training model
# epochs and step_per epoch number
result=model.fit_generator(train_image_gen,epochs=10,
                            steps_per_epoch=2000,
                           validation_data=test_image_gen,validation_steps=12)

model.save_weights("models/male_female_weights.h5") #save model weight
model.save("models/male_female_model.h5") #save model
print("Model Saved")

result.history['acc']
plt.plot(result.history['acc'])

new_model=load_model('models/male_female_model.h5')
man_file='picture/man3.jpg'
woman_file='picture/woman3.jpg'
man_image=image.load_img(man_file,target_size=(150,150))
woman_image=image.load_img(woman_file,target_size=(150,150))
man_image=image.img_to_array(man_image)
woman_image=image.img_to_array(woman_image)
#-------------------------------------------
man_image=np.expand_dims(man_image,axis=0)
woman_image=np.expand_dims(woman_image,axis=0)
man_image=man_image/255
woman_image=woman_image/255
print(man_image.shape)
print(woman_image.shape)
#--------------------------------------------------------------------
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




