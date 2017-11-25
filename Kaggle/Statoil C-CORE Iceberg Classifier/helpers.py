import numpy as np # linear algebra
from keras.preprocessing.image import ImageDataGenerator



def get_images(df):
    '''Create 3-channel 'images'. Return rescale-normalised images.'''
    images = []
    for i, row in df.iterrows():
        # Formulate the bands as 75x75 arrays
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        # Rescale
        r = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
        g = (band_2 - band_2.min()) / (band_2.max() - band_2.min())
        b = (band_3 - band_3.min()) / (band_3.max() - band_3.min())

        rgb = np.dstack((r, g, b))
        images.append(rgb)
        
    return np.array(images)

def get_generator(Xtr, Xv):
    data_gen = ImageDataGenerator(
                shear_range=0.1,
                zoom_range=0.1,
                rotation_range=10,
                width_shift_range=0.05,
                height_shift_range=0.05,
                vertical_flip=True,
                horizontal_flip=True)

    data_gen.fit(Xtr)

    val_gen = ImageDataGenerator()
    val_gen.fit(Xv)
    
    return data_gen, val_gen

