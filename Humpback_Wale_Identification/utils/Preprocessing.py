import numpy as np
import pandas as pd
import os

class ImagePreparation(object):
    """
    This class preprocesses the images and outputs them as a np.matrix
    """

    def __init__(self, image_path):
        self.cwd = os.getcwd()
        self.image_path = image_path

    def convert_to_matrix(self, frame_size, print_remaining):

        """Create directory if it does not exist.
        Parameters
        ----------
        frame_size : int
            The size of a single pistuce in pixels per row and per column
        print_remaining : bool
            It prints every 500 steps how many images are to be preprocessed

        Returns
        -------
        X : np.array
            This is the preprocessed np matrix that has the following dimensions:
            (N_pictures, frame_size, frame_size, 3)
        """

        os.chdir(self.image_path)
        pictures = os.listdir()
        n_pictures = len(pictures)


        # count the number of pictures been processed already
        counter = 0

        # Initialize np array
        X = np.zeros((n_pictures, frame_size, frame_size, 3))


