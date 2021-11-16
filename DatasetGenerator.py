import re
from glob import glob
from os import path
import cv2
import numpy as np

import tensorflow as tf


class SegmentationDataset:

    def __init__(self,
                images_path: str,
                segmentations_path: str,
                file_id_regex: str,
                batch_size: int = 8,
                image_size: tuple = (64, 64),
                interpolation='bicubic',
                fill_mode='reflect',
                validation_split: float = 0.1,
                augment_dataset=True,
                seed=0):
        """
        Creates a binary segmentation dataset from images and their matching segmentations

        :param images_path: string path of the image directory
        :param segmentations_path: string path of the segmentation directory
        :param file_id_regex: regex string used to extract the id in the file name
        :param batch_size: size of the batches to generate
        :param image_size: size to resize the images to
        :param interpolation: string of the interpolation method to use when applying transformation to images
        :param fill_mode: string of the padding method to use when padding images
        :param validation_split: percentage of the validation split
        :param augment_dataset: Whether to augment the images or not
        :param seed: Seed used to initialize the RNG
        """
        self._images_path = path.dirname(images_path)
        self._segmentations_path = path.dirname(segmentations_path)
        self._batch_size = batch_size
        self._image_size = image_size
        self._file_id_regex = re.compile(file_id_regex)
        self._resize_interpolation = interpolation
        self._validation_split = validation_split
        self._seed = seed
        self._rng = tf.random.Generator.from_seed(self._seed, alg='philox')
        self._augment_dataset = augment_dataset

        self._image_segmentation_filepath_pairs = self._group_images_to_segmentations()
        self._training_dataset, self._validation_dataset = self._create_datasets()

    def _group_images_to_segmentations(self) -> list[tuple[str, str]]:
        """
        Creates a list of tuple from the image and segmentation filenames using the id extracted from their filenames
        using the file_id_regex to create pairs
        :return: [(image_path, segmentation_path)]
        """
        images_filenames, segmentation_filenames = self._list_file_names()

        image_indexed_file_ids: list[tuple[str, int]] = self._extract_indexed_file_id(images_filenames)
        segmentation_indexed_file_ids = self._extract_indexed_file_id(segmentation_filenames)

        image_segmentation_pairs: list[tuple[str, str]] = []

        for image_file_id, image_index in image_indexed_file_ids:
            segmentation_indexed_file_id = segmentation_indexed_file_ids[image_index]

            assert segmentation_indexed_file_id[0] == image_file_id, \
                "Mismatch between image id and segmentation, \
                make sure that all images have a corresponding segmentation"

            segmentation_filepath = segmentation_filenames[segmentation_indexed_file_id[1]]
            image_filepath = images_filenames[image_index]
            image_segmentation_pairs.append((image_filepath, segmentation_filepath))

        return image_segmentation_pairs

    def _list_file_names(self) -> tuple[list[str], list[str]]:
        """
        List all filenames of the images and segmentations folders

        :return (image_filenames, segmentation_filenames)
        """
        image_filenames = glob(self._images_path + '/*')
        segmentation_filenames = glob(self._segmentations_path + '/*')
        return image_filenames, segmentation_filenames

    def _extract_indexed_file_id(self, filenames: list[str]) -> list[tuple[str, int]]:
        """
        Extract the file ids from the file paths and add the current index of the element to the tuple
        :param filenames: list of file paths
        :return: list of (file_id, index)
        """
        indexed_file_ids = []

        for index, filename in enumerate(filenames):
            filename = filename.replace("_", '')
            filename = filename.replace("(", '')
            filename = filename.replace(")", '')

            file_id = self._file_id_regex.findall(filename)

            if file_id:
                indexed_file_ids.append((file_id[0], index))

        return indexed_file_ids

    def _create_datasets(self):
        """
        Create the training and validation tf.data dataset with the list of files
        :return: (train_dataset, validation_dataset)
        """
        sample_count = len(self._image_segmentation_filepath_pairs)
        split = int(sample_count * (1.0 - self._validation_split))

        train_file_pairs_tensor = tf.convert_to_tensor(self._image_segmentation_filepath_pairs[:split])
        val_file_pairs_tensor = tf.convert_to_tensor(self._image_segmentation_filepath_pairs[split:])

        train_dataset = tf.data.Dataset.from_tensor_slices(train_file_pairs_tensor)
        validation_dataset = tf.data.Dataset.from_tensor_slices(val_file_pairs_tensor)

        train_dataset = train_dataset.map(self._load_images, num_parallel_calls=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.map(self._load_images, num_parallel_calls=tf.data.AUTOTUNE)

        if self._augment_dataset:
            counter = tf.data.experimental.Counter()
            train_dataset = tf.data.Dataset.zip((train_dataset, (counter, counter)))
            train_dataset = train_dataset.map(self._resize_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            train_dataset = train_dataset.map(self._resize, num_parallel_calls=tf.data.AUTOTUNE)

        validation_dataset = validation_dataset.map(self._resize, num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = train_dataset.batch(self._batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.batch(self._batch_size, num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, validation_dataset

    def _load_images(self, file_pairs: tuple[str, str]):
        """
        Load image and segmentation from a pair of filepath
        :param file_pairs: Pair of an image path and a segmentation filepath
        :return: (image_pixels, segmentation_pixels)
        """
        image_file = tf.io.read_file(file_pairs[0])
        segmentation_file = tf.io.read_file(file_pairs[1])

        image = tf.image.decode_jpeg(image_file, channels=3)
        segmentation = tf.image.decode_jpeg(segmentation_file, channels=3)

        image = tf.cast(image, tf.float32) * (1.0 / 255.0)
        segmentation = tf.cast(segmentation, tf.float32) * (1.0 / 255.0)

        return image, segmentation

    def _resize_and_augment(self, image_segmentation_pair, seed):
        """
        Augment the image and segmentation by applying random transformations
        :param image_segmentation_pair: Pair image and segmentation
        :param seed: The seed for the augmenting functions
        :return: (augmented_image, augmented_segmentation)
        """

        image, segmentation = image_segmentation_pair

        if tf.shape(image)[0] < self._image_size[0] or tf.shape(image)[1] < self._image_size[1]:
            image = tf.image.resize_with_pad(image,
                                            target_width=self._image_size[0],
                                            target_height=self._image_size[1],
                                            method=self._resize_interpolation)
            segmentation = tf.image.resize_with_pad(segmentation,
                                                    target_width=self._image_size[0],
                                                    target_height=self._image_size[1],
                                                    method=self._resize_interpolation)

        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

        image, segmentation = self._resize(image, segmentation)
        # image = tf.image.stateless_random_crop(image, size=[self._image_size[0], self._image_size[1], 3], seed=seed)
        # segmentation = tf.image.stateless_random_crop(segmentation,
        #                                               size=[self._image_size[0], self._image_size[1], 1],
        #                                               seed=seed)

        image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=new_seed)

        image = tf.clip_by_value(image, 0, 1)
        segmentation = tf.clip_by_value(segmentation, 0, 1)

        return image, segmentation

    def _resize(self, x, y):
        if tf.shape(x)[0] < self._image_size[0] or tf.shape(x)[1] < self._image_size[1]:
            x = tf.image.resize_with_pad(x,
                                        target_width=self._image_size[0],
                                        target_height=self._image_size[1],
                                        method=self._resize_interpolation)

        x = tf.image.stateless_random_crop(x, size=[self._image_size[0], self._image_size[1], 3], seed=[self._seed, self._seed])
        # image = tf.image.resize(x,
        #                         size=[self._image_size[0], self._image_size[1]],
        #                         method=self._resize_interpolation,
        #                         antialias=False)
        # segmentation = tf.image.resize(y,
        #                             size=[self._image_size[0], self._image_size[1]],
        #                             method=self._resize_interpolation,
        #                             antialias=False)

        return self.convert_to_yuv(x)

    def convert_to_yuv(self, image):
        img_yuv = tf.image.rgb_to_yuv(image)  #Convertit les images array RGB a des images arrays YUV

        #img_yuv = img_yuv[0:self._image_size[0], 0:self._image_size[0]]

        # img_yuv = img_yuv.astype(np.float32)

        y = img_yuv[:, :, 0]

        return y, img_yuv[:, :, 1:2]

    def get_training_dataset(self):
        return self._training_dataset

    def get_validation_dataset(self):
        return self._validation_dataset

    def get_train_val_datasets(self):
        return self._training_dataset, self._validation_dataset
