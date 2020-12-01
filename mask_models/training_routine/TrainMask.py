import tensorflow as tf
import os
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from keras.layers import Dense, Flatten, Dropout, AveragePooling2D, Input
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
    "./observations-master/experiements/dest_folder/val/without_mask/")
xvali, yval = img_path(
    "./observations-master/experiements/dest_folder/val/with_mask/")
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

baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(image_size, image_size, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.65)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Do not train mobileNetV2 network
for layer in baseModel.layers:
    layer.trainable = False

optimizer = AdaBeliefOptimizer(learning_rate=1e-3, epsilon=1e-6, rectify=False)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

train = model.fit_generator(train_augmentator.flow(out_x, out_y, batch_size=64), epochs=10,
                            validation_data=validation_augmentator.flow(xval, yval), callbacks=[checkpoint])

img_paths = [
    './5f120523682a4.image.jpg',  # true out = 1 0
    './person-wearing-a-mask.jpg']  # true out = 0 1

test_img = read_and_prep_images(img_paths)
prediction = model.predict(test_img)
print(prediction)

path = 'mobileNetModel381adam.h5'

tf.keras.models.save_model(filepath=path, model=model)
