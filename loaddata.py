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

from utils import prepare_dirs
from config import get_config
import os
#import h5py
import imageio
import csv
import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
import cv2
from matplotlib import pyplot as plt
import random

from dataset import random_rotation

class DatasetLoader:
    classList = None
    input_data = None       #self.input_data = self.get_data()

    def __init__(self, data_dir, header_file, filename):
        '''
        Dataset constructor
        :param data_dir:    name of the data directory
        :param header_file: name of the header file
        :param filename:    name of the dataset file
        '''
        print('data_dir ', data_dir, ' header file ', header_file, ' filename ', filename)
        self.header_file = header_file
        self.data_dir = data_dir
        self.filename = filename
        print(self.data_dir, self.header_file, self.filename)
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
    def save_to_input_data(self):
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
        self.input_data = fileList
    ## functions related to the data save in the csv/text files
    def get_data_from_file(self):
        '''
        Read the data file and get the list of images along with their labels
        and assign the input_data to the data set
        '''
        print('Get data from file ', self.data_dir, self.filename)
        self.input_data = pd.read_csv(os.path.join(self.data_dir, self.filename), header=None, delimiter=' ')
        print(self.input_data.head())
    def save_data_to_file(self, dataset, filename):
        '''
        Save the labeled data to the data file
        :param dataset: the dataset to be saved in the file
        :param filename: the name of the file
        '''
        print('Save into the data file ....', filename)
        np.savetxt(filename, dataset, delimiter=' ', fmt='%s')
    def split_data(self, split_percent =0.15):
        '''
        Split the dataset into training and testing
        :param split_percent: the percentage of splitting ex: 15% testing, 15% validation & 75% training
        :return train, validate, test
        '''
        print('Formatting database....')
        print ('classlist', self.classList)
        #classes = list(map(int, self.classList)) # self.classList.astype(int)
        classes = self.classList
        print('number of items for the training set ', int(self.input_data.shape[0] * 0.70))
        print('number of items for the validation set ', int(self.input_data.shape[0] * 0.15))
        print('label list ', self.input_data[:,1], len(self.input_data[:,1]))  # , X = self.input_data[:,0] lbls = self.input_data[:,1]
        train = self.input_data.copy()
        print(len(train), len(self.input_data))
        val = []
        test = []
        for c in range(len(classes)):
            # ind = np.argwhere(self.input_data[:, 1] == c)
            print (c, self.input_data[:, 1][2])
            #ind = [i for i in range(len(self.input_data[:, 1])) if int(self.input_data[:, 1][i]) == c]
            ind = [i for i, value in enumerate(train[:, 1]) if int(value) == c]
            print(len(ind), 'images in class', c)
            step = int(len(ind) * split_percent)  # X.shape[0]  / len(classes)
            step = max(step, 1)
            print('  to be shortened by', len(ind) * split_percent)
            ind1 = ind[0:step]
            ind2 = ind[step:2 * step]
            test.extend(train[ind1])
            val.extend(train[ind2])

            train = np.delete(train, ind1, 0)
            train = np.delete(train, ind2, 0)

        print('train ', len(train))
        print('val ', len(val))
        print('val ', val[5])
        print('test ', len(test))
        print('input_data ', len(self.input_data))
        return train, val, test
    def read_img(self, path, width=28, height=28, channels=3):
        nmin = 0
        nmax = 255
        im = skimage.io.imread(path)
        im = cv2.normalize(im, None, alpha=nmin, beta=nmax, norm_type=cv2.NORM_MINMAX)
        # 2. blur
        im = im.astype('float32')
        im = cv2.medianBlur(im, 3)
        imrs = skimage.transform.resize(im, (width, height, channels), mode='reflect',
                                        preserve_range=True)
        return imrs
    def loadDataset(self, mode='', data_set_folder='dataset/',width=28, height=28, channels=3):
        '''
        Split the dataset into training and testing
        :param mode: train / test / val
        :return samples as numpy array from the file
        '''
        print('Loading dataset', mode, 'images')
        with open(os.path.join(data_set_folder, mode + 'image_set.dat')) as f:
            path_and_labels = f.readlines()
        print(f.name)

        samples = []
        for entry in path_and_labels:
            path = entry.split(' ')[0]
            label = int(entry.split(' ')[1])
            img = self.read_img(path,width, height, channels)
            samples.append([img, label])
        return samples
    def save_dataset_to_file(self, dataset, data_dir='./dataset', mode=''):
        imgset = np.zeros([dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2], len(dataset)])
        lbl = np.zeros([len(dataset)])
        #train_samples[0][0].shape, train_samples[0][1]
        for i in range(len(dataset)):
            imgset[:, :, :, i] = dataset[i][0]
            lbl[i] = dataset[i][1]
        np.save(os.path.join(data_dir, mode +'data'), dataset)
        np.save(os.path.join(data_dir, mode +'lbl'), lbl)
        plt.imshow(dataset[3][0][:, :, 2])
        #print(dataset[3][0][:, :, :])
        plt.show()
    def make_rotate(self, all_samples, rotation_times = 3, width = 28, height = 28, channels = 3):
        #data = np.zeros([dataset[0][0].shape[0], dataset[0][0].shape[1],dataset[0][0].shape[3], len(dataset)])
        #label = np.zeros([len(dataset)])
        print('len(all_samples) ',len(all_samples))

        # Empty arrays
        #data = np.zeros([width, height, channels, len(all_samples)*3])
        data = np.zeros([len(all_samples) * 3, width, height, channels])
        label = np.zeros([len(all_samples)*rotation_times])
        print(data.shape)

        for j in range(len(all_samples)):
            for i in range(rotation_times):
                #print(3*j+i,end=' ')
                sample = all_samples[j]
                print('sample[0].shape', sample[0].shape)
                #data[:, :, :, 3*j+i] = random_rotation(sample[0], width, height, channels)
                data[3 * j + i,:, :, :] = random_rotation(sample[0], width, height, channels)
                label[3*j+i] = sample[1]
        print('make_rotate data.shape ',data.shape, len(data))
        print('make_rotate label.shape ', label.shape, len(label))
        shuffler = np.random.permutation(len(label))
        data = data[shuffler,:,:,:]
        label = label[shuffler]

        return data, label

    def split_rot_data(self, data_dir='./dataset', mode='all_rot_', split_percent = 0.15):
        data = np.load(data_dir + '/' + mode + 'data.npy')
        lbls = np.load(data_dir + '/' + mode + 'lbl.npy')
        print('len(data) ', data.shape)
        print('len(lbls) ', lbls.shape)
        len_test = int(lbls.shape[0]*split_percent)


        data_split = np.split(data, [len_test,
                                    len_test*2,
                                    lbls.shape[0]*2*len_test])      # , axis=3
        label_split = np.split(lbls, [len_test,
                                    len_test*2,
                                    lbls.shape[0]*2*len_test])
        print(data_split[0].shape, data_split[1].shape, data_split[2].shape)
        print(label_split[0].shape, label_split[1].shape, label_split[2].shape)
        train = zip(data_split[2], label_split[2])
        np.save(os.path.join(data_dir, 'train_rot_' + 'data'), data_split[2])
        np.save(os.path.join(data_dir, 'train_rot_' + 'lbl'), label_split[2])
        test = zip(data_split[1], label_split[1])
        np.save(os.path.join(data_dir, 'test_rot_' + 'data'), data_split[1])
        np.save(os.path.join(data_dir, 'test_rot_' + 'lbl'), label_split[1])
        val = zip(data_split[0], label_split[0])
        np.save(os.path.join(data_dir, 'val_rot_' + 'data'), data_split[0])
        np.save(os.path.join(data_dir, 'val_rot_' + 'lbl'), label_split[0])
        return train, test, val



def loadRotData(data_dir):
    def load_and_make_list(mode):
        data = np.load(data_dir + '/' + mode + '_data.npy')
        lbls = np.load(data_dir + '/' + mode + '_lbl.npy')
        print(mode + '_data shape: ', data.shape, data.shape[0])
        print(mode + '_lbl shape: ', lbls.shape, lbls.shape[0])
        return zip(data,lbls)

    train = load_and_make_list('train_rot')
    val = load_and_make_list('val_rot')
    test = load_and_make_list('test_rot')

    return train,val,test

def load_data(data_dir, header_file, filename, make_rotate=True):
    dataset_loader = DatasetLoader(data_dir, header_file, data_dir + '/' + filename)
    dataset_loader.get_classes_from_directory() # assign the class list based on the directory structure
    dataset_loader.save_classes_to_file()   # save the class list into the header file
    dataset_loader.save_to_input_data()     # load the data onto the input_data variable,
                                            # after this step the input data consists of the images
                                            # along with their corresponding classes
                                            # data is shuffled before saving it to the dataset

    dataset_loader.save_data_to_file(dataset_loader.input_data, dataset_loader.filename)   # save what exists in the input_data into the file
    train, val, test = dataset_loader.split_data()
    print('filename ',filename)
    f = filename.split('/')
    print(len(f), f[len(f)-1])
    # saving data into text files
    dataset_loader.save_data_to_file(train, data_dir + '/' + 'train_' + filename)
    dataset_loader.save_data_to_file(test, data_dir + '/' + 'test_' + filename)
    dataset_loader.save_data_to_file(val, data_dir + '/' + 'val_' + filename)
    width = 28
    height = 28
    channels = 3
    # load the images
    train_samples = dataset_loader.loadDataset('train_',data_dir,width,height,channels)
    test_samples = dataset_loader.loadDataset('test_',data_dir,width,height,channels)
    val_samples = dataset_loader.loadDataset('val_',data_dir,width,height,channels)
    #print('train_samples ', len(train_samples), len(train_samples[0]), train_samples[0][0].shape, train_samples[0][1])
    # images are saved as (width, height, channels, len(dataset))
    dataset_loader.save_dataset_to_file(train_samples,data_dir,'train_')
    dataset_loader.save_dataset_to_file(test_samples, data_dir, 'test_')
    dataset_loader.save_dataset_to_file(val_samples, data_dir, 'val_')
    if make_rotate:
        samples = dataset_loader.loadDataset('', data_dir, width, height, channels)
        rot_data, rot_lbl = dataset_loader.make_rotate(samples, rotation_times=3, width = 28, height = 28, channels = 3)
        np.save(data_dir + '/all_rot_data', rot_data)
        np.save(data_dir + '/all_rot_lbl', rot_lbl)
        train, test, val = dataset_loader.split_rot_data(data_dir, mode='all_rot_', split_percent=0.15)

        #dataset_loader.save_dataset_to_file(data_rot, data_dir,'rot_')


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.random_seed)
    prepare_dirs(config)
    header_file = config.data_dir + '/header.tfl.txt'
    log_file = os.path.join(config.model_dir, 'out.txt')
    filename = 'image_set.dat'
    print(config.data_dir, config.model_dir, header_file, log_file, filename)
    load_data(config.data_dir, header_file, filename)
    train,val,test = loadRotData(config.data_dir)
    d, l = list(zip(*train))

    data = np.asarray(d, dtype=np.float32)
    lbls = np.asarray(l, dtype=np.float32)
    print('##### FROM THE MAIN #####')
    print('train_data shape: ', data.shape, lbls.shape)


