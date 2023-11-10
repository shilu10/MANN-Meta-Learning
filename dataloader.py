from utils import get_images, image_file_to_array
import tensorflow as tf 
import os, sys, shutil 
import numpy as np 
import random 


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####

        all_image_batches = []
        all_label_batches = []

        for batch_indx in range(batch_size):
          inputs = np.empty((self.num_samples_per_class, self.num_classes, self.dim_input))
          outputs = np.array([np.eye(self.num_classes, self.num_classes) for _ in range(self.num_samples_per_class)])

          # sample a class|
          sampled_class_folders = [random.choice(folders) for _ in range(self.num_classes)]

          # sample image per class
          images_labels = get_images(paths=sampled_class_folders,
                                    labels=[i for i in range((self.num_classes))],
                                    nb_samples=self.num_samples_per_class,
                                    shuffle=False)
          n_shot = 0
          shot = 0
          for image_label in images_labels:
            if shot > (self.num_samples_per_class)-1:
              shot = 0
              n_shot += 1

            label, image_path = image_label
            inputs[shot][n_shot] = image_file_to_array(image_path, self.dim_input)
            shot += 1

          all_image_batches.append(inputs)
          all_label_batches.append(outputs)

        all_image_batches, all_label_batches = np.array(all_image_batches), np.array(all_label_batches)
        #############################

        return all_image_batches.astype(np.float32), all_label_batches.astype(np.float32)


if __name__ == '__main__':
    num_classes = 3 
    num_samples_per_class = 2 
    dataloader = DataGenerator(num_classes, num_samples_per_class) 
    images, labels = dataloader.sample_batch('train', 32)

    print(f'Meta Data Image shape: {images.shape}')
    print(f'Meta data Label shape: {labels.shape}')

