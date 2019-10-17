#!/usr/bin/env python
# coding: utf-8

# # CIFAR-10 Image Classification Using Keras

# In[9]:


from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 as cv


# In[10]:


(x_train, y_train), (x_test, y_test)= cifar10.load_data()


# In[11]:


print("X_train: {}".format(x_train.shape))
print("Y_train: {}".format(y_train.shape))
print("X_test: {}".format(x_test.shape))
print("Y_test: {}".format(y_test.shape))


# In[12]:


Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)


# In[13]:



print("Y_train: {}".format(Y_train.shape))
print("Y_test: {}".format(Y_test.shape))


# # Convert RGB to Grayscale

# In[14]:


X_train = np.zeros((x_train.shape[0],32,32,1))
X_test = np.zeros(((x_test.shape[0],32,32,1)))
for i in range(x_train.shape[0]):
    img = x_train[i]
    gray_img = np.reshape(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (32,32,1))
    X_train[i] = gray_img
for i in range(x_test.shape[0]):
    img = x_test[i]
    gray_img = np.reshape(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (32,32,1))
    X_test[i] = gray_img

X_train /= 255
X_test /= 255


# # Apply Convolution

# In[15]:


model = Sequential()
#(32,32,1)
model.add(Convolution2D(32,3,3, input_shape = (32,32,1)))
model.add(Activation('relu'))
#(30,30,32)
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
#(28,28,64)
model.add(MaxPooling2D(pool_size = (2,2)))
#(14,14,64)
model.add(Convolution2D(24,4,4))
model.add(Activation('relu'))
#(11,11,24)
model.add(Flatten())

model.add(Dropout(0.40))
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Activation('relu'))


model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()


# In[16]:


model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy'])
model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs=50, verbose=1, batch_size =100)


# In[ ]:





# In[ ]:





# In[ ]:




