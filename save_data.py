import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
import pandas as pd

# train_data, test_data, validation_data = tfds.load(
#                                                     'celeb_a',
#                                                     split=['train', 'test', 'validation'],
#                                                     data_dir='./data',
#                                                     download=False,

#                                                 )

train_data = tfds.load('celeb_a', split='train', data_dir='./data', download=False)
print('done')

train_paths = []
i = 0
for ex in train_data:
    image_array = ex["image"]
    image_array = image_array.numpy()
    image = Image.fromarray(image_array)

    path = './data/celeb_a_png/train/pic_' + str(i)

    image.save(path, format='PNG')
    train_paths.append(path)

    if i%10000==0: print('train conversion ', i, '/162770')
    i+=1

# test_paths = []
# i = 0
# for ex in test_data:
#     image_array = ex["image"]
#     image_array = image_array.numpy()
#     image = Image.fromarray(image_array)

#     path = './data/celeb_a_png/test/pic_' + str(i)

#     image.save(path, format='PNG')
#     test_paths.append(path)

#     if i%5000==0: print('train conversion ', i, '/19962')
#     i+=1


# validation_paths = []
# i = 0
# for ex in validation_data:
#     image_array = ex["image"]
#     image_array = image_array.numpy()
#     image = Image.fromarray(image_array)

#     path = './data/celeb_a_png/validation/pic_' + str(i)

#     image.save(path, format='PNG')
#     validation_paths.append(path)

#     if i%5000==0: print('train conversion ', i, '/19867')
#     i+=1

# train_paths = ['./data/celeb_a_png/train/pic_' + str(i) for i in range(162770)]
# test_paths = ['./data/celeb_a_png/test/pic_' + str(i) for i in range(19962)]
# validation_paths = ['./data/celeb_a_png/validation/pic_' + str(i) for i in range(19867)]

train_df = pd.DataFrame(data=train_paths, columns=['path'])
train_hdf = train_df.to_hdf('./data/celeb_paths/train_paths.h5', key='df')

# test_df = pd.DataFrame(data=test_paths, columns=['path'])
# test_hdf = test_df.to_hdf('./data/celeb_paths/test_paths.h5', key='df')

# validation_df = pd.DataFrame(data=validation_paths, columns=['path'])
# validation_hdf = validation_df.to_hdf('./data/celeb_paths/validation_paths.h5', key='df')

