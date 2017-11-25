import helpers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.applications import VGG16
import abc
from matplotlib import pyplot as plt
import time
import datetime

class DaveBaseModel:
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, Xtr, ytr, Xv, yv):
        self.Xtr = Xtr
        self.ytr = ytr
        self.Xv = Xv
        self.yv = yv
        self.batch_size = 32
        self.epochs = 50
        self.data_gen, self.val_gen = helpers.get_generator(Xtr, Xv)
        
        self.model = self.get_model()
        
    def train(self, batch_size = -1, epochs = -1, saveModel = False):
        if batch_size > 0:
            self.batch_size = batch_size
            
        if epochs > 0:
            self.epochs = epochs
        
        print("Batch Size:", self.batch_size)
        print("Epochs:", self.epochs)

        history = self.model.fit_generator(self.data_gen.flow(self.Xtr, self.ytr, batch_size=self.batch_size),
                         steps_per_epoch=len(self.Xtr) / self.batch_size,
                         epochs=self.epochs,
                         validation_data=self.val_gen.flow(self.Xv, self.yv, batch_size=self.batch_size, shuffle=False),
                         validation_steps = len(self.Xv) / self.batch_size)
        
        self.history = history.history
        
        if (saveModel):
            self.save_model()

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
        predictions = Dense(2, activation='softmax')(x)
        # top_model.load_weights(top_model_weights_path)

        model = Model(inputs=vgg_model.input, outputs=predictions)
        #model = Model(inputs= vgg_model.input, outputs= top_model(vgg_model.output))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        # print(self.model.summary())
        return model

    def get_name(self):
        return "vgg"

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
