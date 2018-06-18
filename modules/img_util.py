from PIL import Image
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pickle
from keras.utils import np_utils


def norm_image_generator(norm_folder):
    tt_splitter = float(
        input('Enter the ratio to split the training and test images (number between 0 and 1):\n--> '))

    # images list will contain face image data. i.e. pixel intensities
    images = []
    # labels list will contain the label that is assigned to the image
    labels = []
    # dictionary/map with label to subject name
    name_map = {}

    # Append all the absolute image paths in a list image_paths
    image_dir_paths = [os.path.join(norm_folder, d) for d in os.listdir(norm_folder)]
    label_idx = 0
    for image_dir_path in image_dir_paths:
        image_paths = [os.path.join(image_dir_path, f) for f in os.listdir(image_dir_path)]
        for image_path in image_paths:
            # Read the image and convert to grayscale
            image_pil = Image.open(image_path).convert('L')
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')

            # Get the subject name for the label
            name = os.path.basename(os.path.normpath(image_dir_path))
            name_map[label_idx] = name

            images.append(image)
            labels.append(label_idx)

        label_idx += 1

    with open('class_dict.pkl', 'wb') as f:
        pickle.dump(name_map, f, pickle.HIGHEST_PROTOCOL)

    number_classes = label_idx
    print("{} classes in dataset".format(number_classes))

    (train_data, test_data, train_labels, test_labels) = train_test_split(images, labels, train_size=tt_splitter)

    train_labels = np_utils.to_categorical(train_labels, number_classes)
    test_labels = np_utils.to_categorical(test_labels, number_classes)

    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)
    train_data = train_data[:, :, :, np.newaxis] / 255.0
    test_data = test_data[:, :, :, np.newaxis] / 255.0

    """
    image_gen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                             samplewise_center=False,
                                                             featurewise_std_normalization=False,
                                                             samplewise_std_normalization=False,
                                                             zca_whitening=False,
                                                             zca_epsilon=1e-06,
                                                             rotation_range=45,
                                                             width_shift_range=0.0,
                                                             height_shift_range=0.0,
                                                             brightness_range=None,
                                                             shear_range=0.0,
                                                             zoom_range=0.0,
                                                             channel_shift_range=0.0,
                                                             fill_mode='nearest',
                                                             cval=0.0,
                                                             horizontal_flip=True,
                                                             vertical_flip=False,
                                                             rescale=None,
                                                             preprocessing_function=None,
                                                             data_format='channels_last')

    train_generator = image_gen.flow_from_directory(
        train_folder,
        target_size=(128, 128),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=None,
        follow_links=False,
        subset=None,
        interpolation='nearest'
    )

    validation_generator = image_gen.flow_from_directory(
        test_folder,
        target_size=(128, 128),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=None,
        follow_links=False,
        subset=None,
        interpolation='nearest'
    )
    """
    return train_data, test_data, train_labels, test_labels, number_classes


def get_single_img(path, grey=True):
    if grey:
        return Image.open(path).convert('L')
    else:
        return Image.open(path)
