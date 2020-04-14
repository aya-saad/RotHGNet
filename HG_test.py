import os
import torch
import torch.nn as nn
from torch.utils import data
from torch.nn import functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from torch import optim
from net import Net, HGNet, SimpleNet, COAPNet, COAPModNet
from ResNet import resnet18, resnet34, resnet50, resnet101, resnet152
from dataloader import *
import numpy as np
from metrics import METRIX
from tqdm import tqdm
import time
# save np.load
np_load_old = np.load
import random
from dataset import random_rotation, linear_interpolation_2D
from utils import getGrid, rotate_grid_2D
from config import get_config
from loaddata import loadRotData
gpu_no = 1 # 0 # Set to False for cpu-version
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def rotate_im(im, theta, width = 28, height = 28, channels = 3):
    grid = getGrid([width, height])
    grid = rotate_grid_2D(grid, theta)
    grid += 13.5
    print('rotate_im: im.shape ', im.shape)
    data = linear_interpolation_2D(im.T, grid)
    print('rotate_im: data.shape ', data.shape)
    data = np.reshape(data, [width, height, channels])
    #print('data.shape ', data.shape, 'data[0].shape', data[0].shape,
    #      'data[:,:,0].shape', data[:, :, 0].shape)
    #data[:, :, 0] = data[:, :, 0] / float(np.max(data[:, :, 0]))
    #data[:, :, 1] = data[:, :, 1] / float(np.max(data[:, :, 1]))
    #data[:, :, 2] = data[:, :, 2] / float(np.max(data[:, :, 2]))
    return data.T.astype('float32')
def adjust_learning_rate(optimizer, epoch, start_lr):
        """Gradually decay learning rate"""
        lr = start_lr

        if epoch > 180:
            lr = lr / 1000000
        elif epoch > 150:
            lr = lr / 100000
        elif epoch > 120:
            lr = lr / 10000
        elif epoch > 90:
            lr = lr / 1000
        elif epoch > 60:
            lr = lr / 100
        elif epoch > 30:
            lr = lr / 10

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def train(model, device, train_loader, optimizer, criterion):
    train_loss = 0
    counter = 0
    model.train()
    for idx_batch, sample in tqdm(enumerate(train_loader),
                                  total=len(train_loader),
                                  leave=False):
        t0 = time.time()
        input, target = sample['image'], sample['label']
        # Move to device
        input, target = input.to(device), target.to(device)
        print('input.shape\t', input.shape, '\ttarget.shape\t', target.shape)
        # Clear optimizers - Clear all accumulated gradients
        optimizer.zero_grad()
        # Predict classes using images from the test set
        outputs = model(input.float())
        # Loss
        #print('target.squeeze(1).long() ', target.squeeze(1).long())
        loss = criterion(outputs, target.squeeze(1).long()) # torch.max(target, 1)[1])
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item() * input.size(0)
        print('TRAIN input.size(0)\t', input.size(0))
        counter += 1
        print(counter, "/", len(train_loader), ' {} seconds'.format(time.time() - t0))
    return train_loss

def val(model, device, val_loader, optimizer, criterion):
    model.eval()
    val_loss = 0
    accuracy = 0
    balanced_accuracy = 0
    with torch.no_grad():
        for idx_batch, sample in enumerate(val_loader):
            input, target = sample['image'], sample['label']
            input, target = input.to(device), target.to(device)
            outputs = model(input.float())
            #print('target.squeeze(1).long() ', target.squeeze(1).long())
            # Calculate Loss
            loss = criterion(outputs, target.squeeze(1).long())
            # Add loss to the validation set's running loss
            val_loss += loss.item() * input.size(0)
            #accuracy, balanced_accuracy, precision, recall, f1_score, rep =
            acc, bacc, _, _, _, _ = \
                METRIX(target.squeeze(1).long(), outputs)
            accuracy+=acc
            balanced_accuracy+=bacc

            #print('VALIDATE input.size(0)', input.size(0))
            # Get the top class of the output
            #top_p, top_class = outputs.topk(1, dim=1)
            #print('outputs ', outputs)
            #print('top_p ', top_p)
            #print('top_class ', top_class)
            # See how many of the classes were correct?
            #equals = top_class == target.view(*top_class.shape)
            #print('labels.view(*top_class.shape ', target.view(*top_class.shape))
            #print('equals ', equals, top_class == target.view(*top_class.shape))
            #correct += equals.type(torch.FloatTensor).sum()
            #correct += torch.sum(equals.type(torch.FloatTensor)).item()
            #print('VALIDATE correct ', correct)
    return val_loss, accuracy, balanced_accuracy

def test(model, device, test_loader):
    # Testing
    model.eval()
    test_loss = 0
    accuracy = 0
    balanced_accuracy = 0
    for sample in test_loader:
        data, target = sample['image'], sample['label']
        data, target = data.to(device), target.to(device)

        outputs = net(data.float())
        #loss = criterion(outputs, torch.max(target, 1)[1])
        loss = criterion(outputs, target.squeeze(1).long())
        test_loss += loss.item() * data.size(0)
        acc, bacc, precision, recall, f1_score, rep = \
            METRIX(target.squeeze(1).long(), outputs)
        accuracy += acc
        balanced_accuracy += bacc
        # ----------------------
        print('Acc:\t{:.3f}%\tBalanced Acc.:\t{:.3f}%\tPrecision:\t{:.3f}%\t'
              'Recall:\t{:.3f}%\tF1 Score:\t{:.3f}%\t'.format(acc * 100, bacc * 100, precision * 100,
                                                          recall * 100, f1_score * 100))
        print('Report: \n', rep)
        # ----------------------

    return test_loss, accuracy, balanced_accuracy

def loadData(data_dir, header_file, filename):
    print('##### DATASET ######')
    composed = transforms.Compose([Resize(64), RandomRotation(), Resize(64), ToTensor(), Normalization()])
    dataset = PlanktonDataSet(data_dir=data_dir, header_file=header_file,
                              csv_file=filename, transform=composed)  # ,transform=Resize(64)

    #dataloader = DataLoader(dataset, batch_size=batch_size,
    #                        shuffle=True, num_workers=batch_size)
    #print('dataset.get_classes_from_file(): ', len(dataset.get_classes_from_file()))
    class_list = dataset.get_classes_from_file()
    print('class_list, len(class_list) ', class_list, len(class_list))

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
    return train_loader, validate_loader, test_loader, class_list

def save_models(epoch, model, name):
    #torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
    torch.save(model.state_dict(), name +'_{:03d}_model.pt'.format(epoch))

    print("Chekcpoint saved")


if __name__ == '__main__':
    # Find the device available to use using torch library
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device("cpu") #

    config, unparsed = get_config()
    np.random.seed(config.random_seed)
    prepare_dirs(config)
    header_file = config.data_dir + '/header.tfl.txt'
    log_file = os.path.join(config.model_dir, 'out.txt')
    filename = 'image_set.dat'
    print(config.data_dir, config.model_dir, header_file, log_file, filename)
    start_lr = 0.001
    weight_decay = 0.0001
    batch_size = 128  # 128
    number_of_epochs = 200 # 200
    best_saved_model = []

    train_loader, validate_loader, test_loader, class_list = loadData(config.data_dir, header_file, filename)
    print('class_list, len(class_list): ', class_list, len(class_list))

    net = COAPNet(num_classes=len(class_list))
    name = 'COAPNet'
    #net = SimpleNet(num_classes=len(class_list))
    #name = 'SimpleNet'
    #net = COAPModNet(num_classes=len(class_list))
    #name = 'COAPModNet'
    #net = resnet18(3, len(class_list))
    #name = 'ResNet'

    #summary(net, (3, 64, 64),batch_size=batch_size)


    net = net.float()
    print(net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=start_lr, weight_decay=weight_decay)
    best_acc = 0
    acc_list = []
    balacc_list = []
    for epoch_no in range(1, number_of_epochs+1):
        print('epoch_no:\t', epoch_no)
        net.to(device)
        t0 = time.time()
        train_loss = train(net, device, train_loader, optimizer, criterion)
        print('Total training time {} seconds'.format(time.time() - t0))
        t0 = time.time()
        valid_loss, accuracy, balaccuracy = val(net, device, validate_loader, optimizer, criterion)
        print('Validation time {} seconds'.format(time.time() - t0))

        # Get the average loss for the entire epoch
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(validate_loader.dataset)
        # Print out the information
        #print('Accuracy: ', accuracy / len(validate_loader))
        print('Epoch:\t{}\tTraining Loss:\t{:.6f}\tValidation Loss:\t{:.6f}'.format(epoch_no, train_loss, valid_loss))
        print('Accu:\t{:.3f}%\tBalanced Accu:\t{:.3f}%\t'.format(accuracy*100 / len(validate_loader),
                                                               balaccuracy*100 / len(validate_loader)))
        acc_list.append(accuracy)
        balacc_list.append(balaccuracy)

        # Validation
        print('*************VALIDATION, val_set_list:*************')
        print('Validation accuracy: \t{:.3f}%\t '
              'Validation_Balanced_Acc: \t{:.3f}% \t'
              'len(validate_loader): \t{}\t'.format(accuracy*100 / len(validate_loader),
                                                        balaccuracy*100 / len(validate_loader),
                                                        len(validate_loader)))
        acc = balaccuracy*100 / len(validate_loader)
        print('Validation epoch:\t', epoch_no, '\tacc:{:.3f}%\t'.format(acc))

        # Save model if better than previous
        filename = os.path.join(config.model_dir, name, name)
        save_models(epoch_no, net, filename)
        if acc > best_acc:
            best_acc = acc
            best_saved_model.append([str('{:03d}').format(epoch_no),acc])
            print('Model saved')

        adjust_learning_rate(optimizer, epoch_no, start_lr)
    # Finally test on test-set with the best model
    #'best_coap_{}_model.pt'.format(epoch))
    # return the best saved_model to load for testing
    print('Accuracy List:\t', *acc_list, sep='\t') # *list, sep='\t'
    print('Balanced accuracy List:\t', *balacc_list, sep='\t')

    m = max(best_saved_model, key=lambda x: x[1])
    print('Loading best model saved: ' + filename + '_' + m[0] + '_model.pt')
    net.load_state_dict(torch.load(filename + '_' + m[0] + '_model.pt'))
    print('*************TEST *************')
    t0 = time.time()
    test_loss, test_acc, test_balacc = test(net, device, test_loader)
    time_test = time.time() - t0
    print('Testing time\t{} seconds'.format(time_test))
    print('Accu:\t {:.3f}%'.format(test_acc*100/ len(test_loader)))
    print('Test acc:\t{:.3f}%\t'.format(test_acc*100 / len(test_loader)),
          'balance acc:\t{:.3f}%\t'.format(test_balacc*100 / len(test_loader)),
          'loss:\t{:.6f}\t'.format(test_loss),
          'test batch_size\t',batch_size)
