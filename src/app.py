# run only onces
#import zipfile as zf
#files = zf.ZipFile("../data/train.zip", 'r')
#files.extractall('../data/raw/images')
#files.close()

import keras,os
import pandas as pd
from keras.models import Sequential  #as all the layers of the model will be arranged in sequence
from keras.layers import Dense, Conv2D, MaxPooling2D , Flatten
from keras.preprocessing.image import ImageDataGenerator # as it imports data with labels easily into the model. It has functions to rescale, rotate, zoom, etc. This class alters the data on the go while passing it to the model.
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.model_selection import train_test_split

IMAGE_FOLDER_PATH = "../data/raw/images/train"
FILE_NAMES = os.listdir(IMAGE_FOLDER_PATH)
WIDTH = 200
HEIGHT = 200

labels = []
for i in os.listdir(IMAGE_FOLDER_PATH):
    labels+=[i]

targets = list()
full_paths = list()
train_cats_dir = list()
train_dogs_dir = list()

# finding each file's target
for file_name in FILE_NAMES:
    target = file_name.split(".")[0] # target name
    full_path = os.path.join(IMAGE_FOLDER_PATH, file_name)
    
    if(target == "dog"):
        train_dogs_dir.append(full_path)
    if(target == "cat"):
        train_cats_dir.append(full_path)
    
    full_paths.append(full_path)
    targets.append(target)

dataset = pd.DataFrame() # make dataframe
dataset['image_path'] = full_paths # file path
dataset['target'] = targets # file's target

print("total data counts:", dataset['target'].count())
counts = dataset['target'].value_counts()
print(counts)

fig = go.Figure(go.Bar(
            x= counts.values,
            y=counts.index,
            orientation='h'))

fig.update_layout(title='Data Distribution in Bars',font_size=15,title_x=0.45)
fig.show()

rows = 4
cols = 4
axes = []
fig=plt.figure(figsize=(10,10))
i = 0

for a in range(rows*cols):
    b = img.imread(train_cats_dir[i])
    axes.append(fig.add_subplot(rows,cols,a+1))
    plt.imshow(b)
    i+=1
fig.tight_layout()
plt.show()

rows = 4
cols = 4
axes = []
fig=plt.figure(figsize=(10,10))
i = 0

for a in range(rows*cols):
    b = img.imread(train_dogs_dir[i])
    axes.append(fig.add_subplot(rows,cols,a+1))
    plt.imshow(b)
    i+=1
fig.tight_layout()
plt.show()

dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1)

train_datagenerator = train_datagen.flow_from_dataframe(
    dataframe=dataset_train,
    x_col="image_path",
    y_col="target",
    target_size=(WIDTH, HEIGHT),
    class_mode="binary",
    batch_size=150)  

test_datagen = ImageDataGenerator(rescale=1./255)
test_datagenerator = test_datagen.flow_from_dataframe(
    dataframe=dataset_test,
    x_col="image_path",
    y_col="target",
    target_size=(WIDTH, HEIGHT),
    class_mode="binary",
    batch_size=150)

model = Sequential()
#→ 2 x convolution layer of 64 channel of 3x3 kernal and same padding
model.add(Conv2D(64, kernel_size=(3,3), input_shape=(WIDTH, HEIGHT, 3), padding="same", activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', padding="same"))
#→ 1 x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#→ 2 x convolution layer of 128 channel of 3x3 kernal and same padding
model.add(Conv2D(128, kernel_size=(3,3), input_shape=(WIDTH, HEIGHT, 3), padding="same", activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), activation = 'relu', padding="same"))
#→ 1 x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#→ 3 x convolution layer of 256 channel of 3x3 kernal and same padding
model.add(Conv2D(256, kernel_size=(3,3), input_shape=(WIDTH, HEIGHT, 3), padding="same", activation='relu'))
model.add(Conv2D(256, kernel_size=(3,3), activation = 'relu', padding="same"))
model.add(Conv2D(256, kernel_size=(3,3), activation = 'relu', padding="same"))
#→ 1 x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#→ 3 x convolution layer of 512 channel of 3x3 kernal and same padding
model.add(Conv2D(512, kernel_size=(3,3), input_shape=(WIDTH, HEIGHT, 3), padding="same", activation='relu'))
model.add(Conv2D(512, kernel_size=(3,3), activation = 'relu', padding="same"))
model.add(Conv2D(512, kernel_size=(3,3), activation = 'relu', padding="same"))
#→ 1 x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#→ 3 x convolution layer of 512 channel of 3x3 kernal and same padding
model.add(Conv2D(512, kernel_size=(3,3), input_shape=(WIDTH, HEIGHT, 3), padding="same", activation='relu'))
model.add(Conv2D(512, kernel_size=(3,3), activation = 'relu', padding="same"))
model.add(Conv2D(512, kernel_size=(3,3), activation = 'relu', padding="same"))
#→ 1 x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#1 x Dense layer of 4096 units
model.add(Dense(4096, activation='relu'))
#→ 1 x Dense layer of 4096 units
model.add(Dense(4096, activation='relu'))
#→ 1 x Dense Softmax layer of 2 units
model.add(Dense(2, activation='softmax'))

from keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(
    "vgg16_1.h5", 
    monitor='val_acc', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False, 
    mode='auto', 
    period=1)

early = EarlyStopping(
    monitor='val_acc', 
    min_delta=0, 
    patience=20, 
    verbose=1, 
    mode='auto')

hist = model.fit_generator(
    steps_per_epoch=100,
    generator=dataset_train, 
    validation_data=dataset_test, 
    validation_steps=10,
    epochs=100,
    callbacks=[checkpoint,early])

import matplotlib.pyplot as plt
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

from keras.preprocessing import image
img = image.load_img("../data/raw/images/cat.0.jpg",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

from keras.models import load_model
saved_model = load_model("vgg16_1.h5")
output = saved_model.predict(img)
if output[0][0] > output[0][1]:
    print("cat")
else:
    print('dog')
