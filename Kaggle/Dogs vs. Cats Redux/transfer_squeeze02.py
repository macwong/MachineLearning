from keras_squeezenet import SqueezeNet
from keras.layers import Dense, Flatten, Convolution2D, Activation, GlobalAveragePooling2D, concatenate, MaxPooling2D, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import backend as K
from keras.utils import get_file
from keras.applications import VGG16


path = ""
sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x

img_input = Input(shape=(227, 227, 3))
x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
x = Activation('relu', name='relu_conv1')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

x = fire_module(x, fire_id=2, squeeze=16, expand=64)
x = fire_module(x, fire_id=3, squeeze=16, expand=64)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

x = fire_module(x, fire_id=4, squeeze=32, expand=128)
x = fire_module(x, fire_id=5, squeeze=32, expand=128)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

x = fire_module(x, fire_id=6, squeeze=48, expand=192)
x = fire_module(x, fire_id=7, squeeze=48, expand=192)
x = fire_module(x, fire_id=8, squeeze=64, expand=256)
x = fire_module(x, fire_id=9, squeeze=64, expand=256)
x = Dropout(0.5, name='drop9')(x)

x = Convolution2D(2, (1, 1), padding='valid', name='conv10')(x)
x = Activation('relu', name='relu_conv10')(x)
x = GlobalAveragePooling2D()(x)
out = Activation('softmax', name='loss')(x)

model = Model(img_input, out, name='squeezenet')

weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5', '', cache_subdir='models')
#model.load_weights(weights_path)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        path + 'train',  # this is the target directory
        target_size=(227, 227),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        path + 'valid',
        target_size=(227, 227),
        batch_size=batch_size,
        class_mode='binary')

steps = train_generator.samples // batch_size

if steps == 0:
    steps = 1
    
val_steps = validation_generator.samples // batch_size

if val_steps == 0:
    val_steps = 1

model.fit_generator(
        train_generator,
        steps_per_epoch=steps,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=val_steps)

model.save_weights('squeezecatsdogs.h5')  # always save your weights after training or during training


    