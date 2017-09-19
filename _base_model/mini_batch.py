# coding=utf-8:
import numpy as np


class DataSet():
    def __init__(self, img_data, labels_data=None, batch_size=1, is_train=False, shuffle=False):
        self.img_data = img_data
        self.labels_data = labels_data
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = self.img_data.shape[0]
        self.num_batches = int(self.count * 1.0 / self.batch_size)
        self.current_index = 0
        self.indices = list(range(self.count))
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()
        start, end = self.current_index, self.current_index + self.batch_size
        current_idx = self.indices[start:end]
        img_file = self.img_data[current_idx]
        labels_file = self.labels_data[current_idx]
        if self.is_train:
            self.current_index += self.batch_size
            return img_file, labels_file
        else:
            self.current_index += self.batch_size
            return img_file, labels_file

    def has_next_batch(self):
        """ Determine whether there is any batch left. """
        return self.current_index + self.batch_size <= self.count


def train_data(images_file, labels_file, batch_size):
    """ Prepare relevant data for training the model. """
    print("Building the training dataset...")
    images_data = np.load(images_file)
    labels_data = np.load(labels_file)
    dataset = DataSet(img_data=images_data, labels_data=labels_data, batch_size=batch_size, is_train=True, shuffle=True)
    print("Dataset built.")
    return dataset


def val_data(images_file, labels_file):
    """ Prepare relevant data for testing the model. """
    images_data = np.load(images_file)
    labels_data = np.load(labels_file)

    print("Building the validation dataset...")
    dataset = DataSet(img_data=images_data, labels_data=labels_data, shuffle=False)
    print("Dataset built")
    return dataset

