from keras_squeezenet import SqueezeNet
from keras.layers import Dense, Flatten, Convolution2D, Activation, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

path = ""

model = SqueezeNet()

# Remove the last classification layer, as it's not necessary for our solution
#test_model.layers.pop()
#model.layers.pop() # Get rid of the dropout layer
#model.outputs = [model.layers[-1].output]
#model.layers[-1].outbound_nodes = []
#model.output_layers.clear()
#for layer in test_model.layers:
#    layer.trainable = False

#output = Dense(2, activation = 'softmax')
#model.layers.append(Flatten())
#model.layers.append(Dense(300, activation = 'relu'))
#model.layers.append(Dense(2, activation = 'softmax'))

model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()

x = model.layers[len(model.layers) - 1]
x = Convolution2D(2, (1, 1), padding='valid', name='conv10')(x)
x = Activation('relu', name='relu_conv10')(x)
x = GlobalAveragePooling2D()(x)
out = Activation('softmax', name='loss')(x)

model = Model(model.inputs, out, name='squeezenet')

#x = test_model.layers[65]
#output = Dense(2, activation = 'softmax')(x)

#model = Model(test_model.input, output = output)

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


    