from __future__ import division,print_function
import keras.backend as K

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)

from matplotlib import pyplot as plt
from imp import reload
import utils; reload(utils)
from utils import plots
import vgg16; reload(vgg16)
from vgg16 import Vgg16



path = "data/dogscats/"
#path = "data/dogscats/blah/"

# As large as you can, but no larger than 64 is recommended. 
# If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
batch_size=32

# Import our class, and instantiate
vgg = Vgg16()

# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=3)

vgg.model.save_weights('ft3.h5')



#vgg.model.load_weights('ft2.h5')
#
#



test_path = "data/dogscats/testing"

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

gen = ImageDataGenerator()
images = gen.flow_from_directory(test_path, target_size=(224,224), class_mode='categorical', shuffle = False)
predict = vgg.model.predict_generator(images, images.samples // images.batch_size + 1)

submission = pd.DataFrame()
submission['label'] = predict[0:, 1]

test_list = []

for fn in range(len(images.filenames)):
    test_list.append(images.filenames[fn].replace('test1\\', '').replace('.jpg', ''))
    
submission['id'] = pd.to_numeric(test_list)

submission.sort_values(["id"], inplace = True)

submission['label'][submission['label'] < 0.05] = 0.05
submission['label'][submission['label'] > 0.95] = 0.95

submission.to_csv('submission.csv', columns = ["id", "label"], index = False)
