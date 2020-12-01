import tensorflow as tf
import os
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from adabelief_tf import AdaBeliefOptimizer

image_size = 224

# Generate training data by augmentation
train_augmentator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

# Generate validation data by augmentation
validation_augmentator = ImageDataGenerator()


# read the label for images
def img_path(path):
    for dirname, _, filenames in os.walk(path):
        l = []
        y = []
        for filename in filenames:
            l.append(os.path.join(dirname, filename))
            y.append(os.path.join(dirname, filename).split("/")[-2])
        return (l, y)

    # resize and prepare input images


def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return (output)

    # define training  validation data


l_img_path, out_y = img_path(
    "./observations-master/experiements/dest_folder/train/with_mask/")
x, y = img_path(
    "./observations-master/experiements/dest_folder/train/without_mask/")
l_img_path = l_img_path + x
out_y = out_y + y
x, y = img_path(
    "./observations-master/experiements/dest_folder/test/with_mask/")
l_img_path = l_img_path + x
out_y = out_y + y
x, y = img_path(
    "./observations-master/experiements/dest_folder/test/without_mask/")
l_img_path = l_img_path + x
out_y = out_y + y
out_x = read_and_prep_images(l_img_path)
xval1, yval1 = img_path(
    "./observations-master/experiements/dest_folder/val/with_mask/")
xvali, yval = img_path(
    "./observations-master/experiements/dest_folder/val/without_mask/")
yval = yval + yval1
xvali = xvali + xval1
xval = read_and_prep_images(xvali)

# labelize output
for i in range(len(out_y)):
    if out_y[i] == "with_mask":
        out_y[i] = 1
    else:
        out_y[i] = 0

for i in range(len(yval)):
    if yval[i] == "with_mask":
        yval[i] = 1
    else:
        yval[i] = 0

    # convert to categorical in form [1, 0] or [0, 1]
out_y = keras.utils.to_categorical(out_y, 2)
yval = keras.utils.to_categorical(yval, 2)

model = tf.keras.Sequential([
    Conv2D(16, kernel_size=3, strides=(1, 1), input_shape=(224, 224, 3)),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(32, kernel_size=3, strides=(1, 1)),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(64, kernel_size=3, strides=(1, 1)),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, kernel_size=3, strides=(1, 1)),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, kernel_size=3, strides=(1, 1)),
    BatchNormalization(),
    ReLU(),
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Flatten(name="flatten"),
    Dense(512),
    Dropout(0.2),
    Dense(256),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

optimizer = AdaBeliefOptimizer(learning_rate=1e-3, epsilon=1e-6, rectify=False)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

train = model.fit_generator(train_augmentator.flow(out_x, out_y, batch_size=64), epochs=40,
                            validation_data=validation_augmentator.flow(xval, yval), callbacks=[checkpoint])

img_paths = [
    './person-wearing-a-mask.jpg',  # true out = [0 1]
    './5f120523682a4.image.jpg']  # true out = [1 0]

test_img = read_and_prep_images(img_paths)
preds = model.predict(test_img)
print(preds)

path = 'ourOwnModel381adam.h5'

tf.keras.models.save_model(filepath=path, model=model)
