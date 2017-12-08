from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.applications import VGG16, VGG19
from keras.preprocessing.image import ImageDataGenerator
import abc
from matplotlib import pyplot as plt
import time
import datetime
import pandas as pd

class DaveBaseModel:
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, ids = None):
        self.batch_size = 32
        self.epochs = 50
        self.ids = ids
        
        self.model = self.get_model()
        
    def get_generator(self, Xtr):
        data_gen = ImageDataGenerator(
                    shear_range=0.1,
                    zoom_range=0.1,
                    rotation_range=10,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    vertical_flip=True,
                    horizontal_flip=True)

        data_gen.fit(Xtr)

        return data_gen
    
    def get_generator_validation(self, Xv):
        return ImageDataGenerator()

    def train(self, X_train, y_train, X_val = None, y_val = None, batch_size = -1, epochs = -1, saveModel = False):
        if batch_size > 0:
            self.batch_size = batch_size

        if epochs > 0:
            self.epochs = epochs

        data_gen = self.get_generator(X_train)
            
        val_data = None
        val_steps = None
            
        if (X_val is not None and y_val is not None):
            val_gen = self.get_generator_validation(X_val)
            val_data = val_gen.flow(X_val, y_val, batch_size=self.batch_size, shuffle=False)
            val_steps = len(X_val) / self.batch_size
        
        print("\n")
        print("================================================")
        print("Model:", self.get_name())
        print("Batch Size:", self.batch_size)
        print("Epochs:", self.epochs)

        history = self.model.fit_generator(data_gen.flow(X_train, y_train, batch_size=self.batch_size),
                         steps_per_epoch=len(X_train) / self.batch_size,
                         epochs=self.epochs,
                         validation_data = val_data,
                         validation_steps = val_steps)
        
        self.history = history.history
        
        if (saveModel):
            self.save_model()

    def predict(self, X_test, submit = True):
        pred_gen = ImageDataGenerator()
        predict = self.model.predict_generator(pred_gen.flow(X_test, batch_size=self.batch_size, shuffle = False), len(X_test) / self.batch_size)
        
        if submit and self.ids is not None:
            self.create_submission(predict)
        
        return predict
    
    def plot_results(self):
        plt.plot(self.history['acc'])
        plt.plot(self.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
    def save_model(self):
        name = self.get_name() + datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S') + ".h5"
        self.model.save_weights(name)
        
    def create_submission(self, predict):
        submission = pd.DataFrame(self.ids, columns=["id"])
        
        submission["is_iceberg"] = predict[:, 1]

        test_func = lambda p: round(p["is_iceberg"], 4)
        submission["is_iceberg"] = test_func(submission)
        submission["is_iceberg"] = submission["is_iceberg"].round(4)
        submission.to_csv("submission-" + self.get_name() + ".csv", float_format='%g', index = False)
    
    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

class DaveModel(DaveBaseModel):
    def ConvBlock(model, layers, filters):
        '''Create [layers] layers consisting of zero padding, a convolution with [filters] 3x3 filters and batch normalization. Perform max pooling after the last layer.'''
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(filters, (3, 3), activation='relu'))
            model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def get_model(self):
        '''Create the FCN and return a keras model.'''

        model = Sequential()

        # Input image: 75x75x3
        model.add(Lambda(lambda x: x, input_shape=(75, 75, 3)))
        DaveModel.ConvBlock(model, 1, 32)
        # 37x37x32
        DaveModel.ConvBlock(model, 1, 64)
        # 18x18x64
        DaveModel.ConvBlock(model, 1, 128)
        # 9x9x128
        DaveModel.ConvBlock(model, 1, 128)
        # 4x4x128
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(2, (3, 3), activation='relu'))
        model.add(GlobalAveragePooling2D())
        # 4x4x2
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        return model
    
    def get_name(self):
        return "davemodel"

    
class DaveVGG(DaveBaseModel):
    def get_model(self):
        vgg_model = VGG16(include_top=False, weights=None, input_shape=(75, 75, 3))

        #top_model = Sequential()
        x = vgg_model.output
        x = Flatten()(x)
        #x = Dense(512, activation='relu')(x)
        #x = Dense(512, activation='relu')(x)
        #x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        predictions = Dense(2, activation='softmax')(x)
        # top_model.load_weights(top_model_weights_path)

        model = Model(inputs=vgg_model.input, outputs=predictions)
        #model = Model(inputs= vgg_model.input, outputs= top_model(vgg_model.output))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        # print(self.model.summary())
        return model

    def get_name(self):
        return "vgg"
    

class DaveVGG19(DaveBaseModel):
    def get_model(self):
        vgg_model = VGG19(include_top=False, weights=None, input_shape=(75, 75, 3))

        x = vgg_model.output
        x = Flatten()(x)
        predictions = Dense(2, activation='softmax')(x)

        model = Model(inputs=vgg_model.input, outputs=predictions)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        return model

    def get_name(self):
        return "vgg19"


class SimpleModel(DaveBaseModel):
    def get_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: x, input_shape=(75, 75, 3)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        
        return model
        
    def get_name(self):
        return "simple"

class LeNetModel(DaveBaseModel):
    def get_model(self):
        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=(75, 75, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(2))
        model.add(Activation("softmax"))
        
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        
        return model
        
    def get_name(self):
        return "lenet"
    
    
    
    