# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import pickle
import random
import cv2


np.random.seed(0)
cols = 5
num_classes = 43

# Unpickling the Datasets
with open('train.p', 'rb') as f:
    training_set = pickle.load(f)
    
with open('test.p', 'rb') as f:
    test_set = pickle.load(f)
    
with open('valid.p', 'rb') as f:
    validation_set = pickle.load(f)


# Get the Training and Test Sets
X_train = training_set['features']
y_train = training_set['labels']
X_test = test_set['features']
y_test = test_set['labels']
X_val  = validation_set['features']
y_val  = validation_set['labels']


def get_and_visualize_data():
    
    assert(X_train.shape[0] == y_train.shape[0]), "No. of images in X_train and y_train are unequal"
    assert(X_test.shape[0] == y_test.shape[0]), "No. of images in X_test and y_test are unequal"
    assert(X_val.shape[0] == y_val.shape[0]), "No. of images in X_val and y_val are unequal"
    assert(X_test.shape[1:] == (32, 32, 3)), "Images are not equal to 32 x 32 x 3"
    assert(X_train.shape[1:] == (32, 32, 3)), "Images are not equal to 32 x 32 x 3"
    assert(X_val.shape[1:] == (32, 32, 3)), "Images are not equal to 32 x 32 x 3"
    
    dataset = pd.read_csv('signnames.csv')
    
    num_of_samples = []
     
    fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 50))
    fig.tight_layout()
    for i in range(cols):
        for j, row in dataset.iterrows():
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j))
                num_of_samples.append(len(x_selected))

           
def grayscale(image):
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2GRAY)
    # print(X_train[1000].shape)
    # print(y_train[1000])
    return gray_image


# image = grayscale(X_train[1000])
# plt.imshow(image)
# plt.axis("off")


# HIstogram Equalization - To get normalized lighting effects

def histogram_equalize(image):
    image = cv2.equalizeHist(image)
    return image

# image = histogram_equalize(image)
# plt.imshow(image)
# plt.axis("off")



def preprocessing(image):
    image = grayscale(image)
    image = histogram_equalize(image)
    image = image/255
    return image


X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

# fig2 = plt.figure(2)
# plt.imshow(X_train[random.randint(0, len(X_train)-1)])
# plt.axis('off')

X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)
#print(X_train.shape)


# ImageData Augmentation Technique using Fit Generator
def fit_generation():
    global dataGeneration
    dataGeneration = ImageDataGenerator(width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        shear_range=0.1,
                                        rotation_range=10 # Degrees
                                        )
   
    dataGeneration.fit(X_train)
    batches = dataGeneration.flow(X_train, y_train, batch_size=20)
    X_batch, y_batch = next(batches)
    
    # Plot the Transformed Image
    fig, axis = plt.subplots(1, 15, figsize=(20,5))
    fig.tight_layout()
    for i in range(15):
        axis[i].imshow(X_batch[i].reshape(32,32))
        axis[i].axis('off')
    
    return dataGeneration

fit_generation()

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)



def leNet_model():
    classifier = Sequential()
    classifier.add(Conv2D(50, (5,5), input_shape=(32,32,1), activation='relu'))
    classifier.add(Conv2D(50, (5,5), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    classifier.add(Conv2D(30, (3,3), activation='relu'))
    classifier.add(Conv2D(30, (3,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    # classifier.add(Dropout(rate=0.5))
    
    classifier.add(Flatten())
    classifier.add(Dense(units=500, activation='relu'))
    classifier.add(Dropout(rate=0.5))
    classifier.add(Dense(units=num_classes, activation='softmax'))
    classifier.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return classifier


model = leNet_model()
history = model.fit_generator(dataGeneration.flow(X_train, y_train, batch_size=50), epochs=15,
                                                  validation_data=(X_val, y_val), shuffle=1)

fig1 = plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'implementation'])
plt.xlabel('epoch')
plt.title('loss')


fig2 = plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.xlabel('epoch')
plt.title('acc')


# Test the Data
score = model.evaluate(X_test, y_test, verbose = 0)
print('Test Score', score[0])
print('Test Accuracy', score[1])









