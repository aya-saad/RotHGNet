#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the DatasetExporter class
# Author: Aya Saad
# Date created: 30 September 2019
#
#################################################################################################################

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, utils, datasets
import imageio
import csv
import cv2
import random
from PIL import Image

from utils import prepare_dirs
from config import get_config
from dataset import random_rotation

class PlanktonDataSet(Dataset):
    """ Planktonic dataset data loader """
    classList = None
    data_list = None       #self.input_data = self.get_data()

    def __init__(self, data_dir='./dataset', header_file = 'header.tfl.txt',
                 csv_file='image_set.dat', transform = None):
        '''
        Dataset constructor
        :param data_dir:    name of the data directory
        :param header_file: name of the header file
        :param filename:    name of the dataset file
        '''
        print('data_dir ', data_dir, ' header file ', header_file, ' csv_file ', csv_file)
        self.header_file = header_file
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.load_data()

        self.transform = transform
        print(self.data_dir, self.header_file, self.csv_file)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #print('idx ', idx)
        img_name = self.data_list.iloc[idx, 0].split(' ')[0]

        #img_name = os.path.join(self.data_list.iloc[idx, 0])
        #print(self.data_list.iloc[idx, 0])
        #print('img_name ', img_name)
        image = io.imread(img_name)
        #print('img_shape ', image.shape)
        label = self.data_list.iloc[idx, 0].split(' ')[1]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        #print('image size after resize: ', sample['image'].shape)

        return sample

    def load_data(self):
        # get the class list from the directory or the header.tfl.txt file
        if os.path.isfile(os.path.join(self.data_dir, self.header_file)):
            self.get_classes_from_file()
        else:
            self.get_classes_from_directory()  # assign the class list based on the directory structure
            self.save_classes_to_file()  # save the class list into the header file
        # get the data list from the directory or the image_set.dat file
        if os.path.isfile(os.path.join(self.data_dir, self.csv_file)):
            print(self.csv_file, "File exist")
            self.get_data_from_file()
        else:
            self.save_to_data_list()  # load the data from the directories onto the input_data variable,
            self.save_data_to_file(self.data_list, os.path.join(self.data_dir, self.csv_file))
        # after this step the input data consists of the images
        # along with their corresponding classes
        # data is shuffled before saving it to the dataset
        print(os.path.join(self.data_dir, self.csv_file))
        self.data_list = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
    ## functions for the data directory structure
    def get_classes_from_file(self):
        '''
        Get the list of classes from the header file
        '''
        print('Get the list of classes from the header file ', self.header_file)
        cl_file = self.header_file
        with open(cl_file) as f:
            reader = csv.reader(f)
            cl = [r for r in reader]
        self.classList = cl[0]
        return self.classList
    def get_classes_from_directory(self):
        '''
        Get the list of classes from the directory
        '''
        print('Get classes from the database directory ', self.data_dir)
        self.classList = [o for o in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, o))]
        print('List of classes from the directory ', self.classList)
    def save_classes_to_file(self):
        '''
        save the list of classes into the header file
        :param classList:  the list of classes
        '''
        print('Save classes to file ', self.header_file)
        df_classes = pd.DataFrame(columns=self.classList)
        df_classes.to_csv(self.header_file, index=False)
    def save_to_data_list(self):
        '''
        # load the data onto the input_data variable
        # after this step the input data consists of the images along with their corresponding classes
        '''
        print('Import file list from the directory structure ')
        fileList = []
        for c_ind, c in enumerate(self.classList):
            print('  ', c)
            filepath = os.path.join(self.data_dir, c)   #DATABASE_PATH
            files = [o for o in os.listdir(filepath) if o.endswith('.tiff')]
            for f in files:
                fileList.append([os.path.join(filepath, f), str(c_ind)])
        fileList = np.array(fileList)
        print('Shuffle dataset....')
        np.random.shuffle(fileList)
        self.data_list = fileList
    ## functions related to the data save in the csv/text files
    def get_data_from_file(self):
        '''
        Read the data file and get the list of images along with their labels
        and assign the input_data to the data set
        '''
        print('Get data from file ', self.data_dir, self.csv_file)
        self.input_data = pd.read_csv(os.path.join(self.data_dir, self.csv_file), header=None, delimiter=' ')
        print(self.input_data.head())
    def save_data_to_file(self, dataset, filename):
        '''
        Save the labeled data to the data file
        :param dataset: the dataset to be saved in the file
        :param filename: the name of the file
        '''
        print('Save into the data file ....', filename)
        np.savetxt(filename, dataset, delimiter=' ', fmt='%s')

class Resize(object):
    """Resize the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'label': label}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}

class RandomRotate(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w, c = image.shape  #[:2]
        image = random_rotation(image, h, w, c)

        return {'image': image, 'label': label}

class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        #print('h, w ', h, w)
        #print('new_h, new_w ', new_h, new_w)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        #print('top , left ', top, left)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}

class RandomRotation(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w, c = image.shape
        # print('image[0][0][0] ',int(np.uint8(image)[0][0][0]), int(np.amax(np.uint8(image*255))))

        fill_color = int(np.amax(np.uint8(image * 255)))
        if int(np.uint8(image * 255)[0][0][0]) > 0:
            fill_color = int(np.uint8(image * 255)[0][0][0])

        randRot = transforms.RandomRotation(360, expand=True, resample=Image.BICUBIC, fill=fill_color)
        pil_image = transforms.ToPILImage()(np.uint8(image*255))
        rot_image = randRot(pil_image)
        rot_image = np.array(rot_image)

        return {'image': rot_image,
                'label': label}

class Normalization(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        #norm = transforms.Normalize(mean=[0.0, 0.0], std=[0.5, 0.5])
        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        h, w = image.shape[:2]
        resize = transform.resize(image, (h, w))
        #pil_image = transforms.ToPILImage()(image)

        image = norm(image) # torch.from_numpy(resize(image))
        return {'image': image,
                'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(np.array([int(label)]))}

# Helper function to show a batch
def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label_batch = \
            sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)

    print('images_batch.shape ', images_batch.shape)


    grid = utils.make_grid(torch.from_numpy(np.array(images_batch)))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    #plt.imshow(grid.numpy().transpose((1, 0, 2)))


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.random_seed)
    prepare_dirs(config)
    header_file = config.data_dir + '/header.tfl.txt'
    log_file = os.path.join(config.model_dir, 'out.txt')
    filename = 'image_set.dat'
    print(config.data_dir, config.model_dir, header_file, log_file, filename)
    resize = Resize(64)
    crop = RandomCrop(64)
    trotate = RandomRotation()

    composed = transforms.Compose([Resize(64), RandomRotation(), Resize(64),
                                   ToTensor(), Normalization()])
    dataset = PlanktonDataSet(data_dir=config.data_dir, header_file = header_file,
                 csv_file=filename, transform=composed)  # ,transform=Resize(64)

    print('dataset.get_classes_from_file(): ', len(dataset.get_classes_from_file()))

    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=batch_size)

    ####  split dataset into train, test and validate
    print(len(dataset))
    split_size = int(len(dataset) * 0.15)
    indices = list(range(len(dataset)))
    train, validate, test = random_split(dataset,
                                         [len(dataset) - 2 * split_size,
                                          split_size,
                                          split_size])
    print(len(train))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    print(len(validate))
    validate_loader = DataLoader(validate, batch_size=batch_size, shuffle=True)
    print(len(test))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    i = 0
    plt.figure()
    for i_batch, sample_batched in enumerate(test_loader):
        i = i + 1
        #print(i_batch)  #, sample_batched['image'].size(),sample_batched['label']
        print(i_batch, sample_batched['image'].size(),
              sample_batched['label'])

        ax = plt.subplot(1, batch_size, i)
        plt.tight_layout()
        show_batch(sample_batched)
        plt.axis('off')
        plt.ioff()

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.show()
            break

'''
    scale = Rescale(256)
    crop = RandomCrop(8)  # 128
    rotate = RandomRotate()
    trotate = RandomRotation()
    composed = transforms.Compose([ #Rescale(256),
                                   RandomRotate(),  # Normalization(),  RandomCrop(224),
                                   ToTensor()])  # 224

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = dataset[62]
    print('sample[label] ', sample['label'])
    for i, tsfrm in enumerate([trotate, rotate, composed]):  # scale, crop,
        transformed_sample = tsfrm(sample)
        print('transformed_sample[label] ', transformed_sample['label'])

        ax = plt.subplot(1, 5, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        plt.imshow(transformed_sample['image'])

    plt.show()

'''
'''
    fig = plt.figure()
    i = 0
    #for batch_index, (inputs, labels) in enumerate(dataset):
    for i in range(len(dataset)):
        inputs, labels = dataset[i]['image'], dataset[i]['label']
        #print(i, inputs.shape, labels)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(labels))
        ax.axis('off')
        plt.imshow(inputs)

        if i == 3:
            plt.show()
            break
    
'''
'''
####  split dataset into train, test and validate
    
    batch_size = 16
    print(len(dataset))
    split_size = int(len(dataset) * 0.15)
    indices = list(range(len(dataset)))
    train, validate, test = random_split(dataset,
                                              [len(dataset) - 2 * split_size,
                                               split_size,
                                               split_size])
    print(len(train))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    print(len(validate))
    validate_loader = DataLoader(validate, batch_size=batch_size, shuffle=True)
    print(len(test))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)


    for batch_index, (faces, labels) in enumerate(train_loader):
        print(labels)

'''
'''
    shuffle_dataset = True
    random_seed = 42
    print(len(dataset))
    split_size = int(len(dataset) * 0.15)
    indices = list(range(len(dataset)))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[2 * split_size:], \
                                               indices[split_size:2 * split_size], \
                                               indices[:split_size]
    print('_____________________________')
    print('train_indices ',train_indices)
    print('val_indices ', val_indices)
    print('test_indices ', test_indices)
    print('_____________________________')

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=test_sampler)

    '''