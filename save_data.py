import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
import pandas as pd

train_data, test_data, validation_data = tfds.load(
                                                    'celeb_a',
                                                    split=['train', 'test', 'validation'],
                                                    data_dir='./data',
                                                    download=True
                                                )

train_paths = []
i = 0
for ex in train_data:
    image_array = ex["image"]
    image_array = image_array.numpy()
    image = Image.fromarray(image_array)

    path = './data/celeb_a_png/train/' + str(i)

    image.save(path, format='PNG')
    train_paths.append(path)

test_paths = []
i = 0
for ex in test_data:
    image_array = ex["image"]
    image_array = image_array.numpy()
    image = Image.fromarray(image_array)

    path = './data/celeb_a_png/test/' + str(i)

    image.save(path, format='PNG')
    test_paths.append(path)


validation_paths = []
i = 0
for ex in validation_data:
    image_array = ex["image"]
    image_array = image_array.numpy()
    image = Image.fromarray(image_array)

    path = './data/celeb_a_png/train/' + str(i)

    image.save(path, format='PNG')
    validation_paths.append(path)

train_df = pd.DataFrame(data=train_paths, columns='path', key='df')
train_hdf = train_df.to_hdf('./data/celeb_paths/train_paths.h5')

test_df = pd.DataFrame(data=test_paths, columns='path', key='df')
test_hdf = test_df.to_hdf('./data/celeb_paths/test_paths.h5')

validation_df = pd.DataFrame(data=validation_paths, columns='path', key='df')
validation_hdf = validation_df.to_hdf('./data/celeb_paths/validation_paths.h5')

