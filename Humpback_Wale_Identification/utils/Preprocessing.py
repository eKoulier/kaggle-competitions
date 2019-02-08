import numpy as np
import pandas as pd
import os

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Keras Imports
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


class ImagePreparation(object):
    """
    This class preprocesses the images and outputs them as a np.matrix
    Parameters
    ----------
        df : pandas df
        It should have a column named 'Image':
            Image
        0   image1.jpg
        1   image2.jpg
    """

    def __init__(self, image_path, df):
        self.df = df
        assert 'Image' in self.df.columns

    def convert_to_matrix(self, frame_size, print_remaining):

        """Loads all images from a file and converts them to np array.
        Parameters
        ----------
        frame_size : int
            The size of a single picture in pixels per row and per column
        print_remaining : bool
            It prints every 500 steps the number of images remaining to be processed.

        Returns
        -------
        X : np.array
            This is the preprocessed np matrix that has the following dimensions:
            (N_pictures, frame_size, frame_size, 3)
        y: np.array
            If the classes variable is true then this array corresponds to a vectorized
            class matrix.
        """
        assert (type(print_remaining) == bool)

        os.chdir(self.image_path)
        images = os.listdir()
        n_images = len(images)

        # count the number of pictures been processed already
        counter = 0

        # Initialize np array
        X = np.zeros((n_images, frame_size, frame_size, 3))

        for image_name in self.df['Images']:
            if image_name in images:
                img = image.load_img(image_name, target_size=(frame_size, frame_size, 3))
                x = image.img_to_array(img)
                x = preprocess_input(x)

                # Add the image matrix to X
                X[counter] = x

                if print_remaining:
                    if (n_images - counter) % 500 == 0 & counter != 0:
                        print(n_images - counter, ' images remain to be processed.')

            else:
                print('Warning! ', image_name,' is not in directory')

            counter += 1

        # Rescale pixel intensity from 0 to 1
        return X / 225

    def prepare_labels(self):
        """Returns a vectorized np.array that contains the class for each picture
        The self.df should have a column named 'Id'
        Returns
        -------
        one_hot_encoded_labels : np.array in a vectorized form (n_images, n_classes)
        label_encoder : to be used in case we need to find the original class name
        """

        assert 'Id' in self.df.columns

        values = np.array(self.df['Id'])
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        # print(integer_encoded)

        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        one_hot_encoded_labels = onehot_encoder.fit_transform(integer_encoded)

        # print(y.shape)
        return one_hot_encoded_labels, label_encoder
