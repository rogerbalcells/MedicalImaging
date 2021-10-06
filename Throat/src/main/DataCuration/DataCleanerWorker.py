import torchvision
import torch
import os
import torch.nn as nn
from numpy import ndarray
from torchvision import transforms
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import random
import scipy as sp

class DataCleanerWorker:
    def __init__(self, data):
        self.data = data

        dataCleanerScript()

    def renameFiles(self):
        for i in range(len(strep_names)):
            src = strep_folder + strep_names[i]
            if random.choices(mylist, weights=[5, 1])[0] == "test":
                trg = test_folder + 'strep_' + str(i) + '.jpg'
                os.rename(src, trg)
            else:
                trg = train_folder + 'strep_' + str(i) + '.jpg'
                os.rename(src, trg)

    def resizeImage(self):
        healthy_names = [name for name in os.listdir(healthy_folder)]
        batch_size = len(healthy_names)
        healthy = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
        img = torch.zeros(3, 256, 256, dtype=torch.uint8)
        downSample = nn.AvgPool2d((256, 256))

        p = transforms.Compose([transforms.Resize((256, 256))])

        for i in range(len(healthy_names)):
            img = torchvision.io.read_image(healthy_folder + healthy_names[i])
            img = p(img)
            healthy[i] = img

        strep_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/pharyngitis/'
        strep_names = [name for name in os.listdir(strep_folder)]
        batch_size = len(strep_names)
        strep = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

    def dataCleanerScript(self):
        healthy_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/no_pharyngitis/'
        healthy_names = [name for name in os.listdir(healthy_folder)]
        batch_size = len(healthy_names)
        healthy = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
        img = torch.zeros(3, 256, 256, dtype=torch.uint8)
        downSample = nn.AvgPool2d((256, 256))

        p = transforms.Compose([transforms.Resize((256, 256))])

        for i in range(len(healthy_names)):
            img = torchvision.io.read_image(healthy_folder + healthy_names[i])
            img = p(img)
            healthy[i] = img

        strep_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/pharyngitis/'
        strep_names = [name for name in os.listdir(strep_folder)]
        batch_size = len(strep_names)
        strep = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

        for i in range(len(strep_names)):
            img = torchvision.io.read_image(strep_folder + strep_names[i])
            img = p(img)
            strep[i] = img

        healthy = healthy.float()
        healthy /= 255.0
        strep = strep.float()
        strep /= 255.0

        test_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/test_data/'
        train_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/training_data/'
        mylist = ["train", "test"]
        strep_names = [name for name in os.listdir(strep_folder)]
        healthy_names = [name for name in os.listdir(healthy_folder)]

        print(random.choices(mylist, weights=[5, 1]))
        for i in range(len(strep_names)):
            src = strep_folder + strep_names[i]
            if random.choices(mylist, weights=[5, 1])[0] == "test":
                trg = test_folder + 'strep_' + str(i) + '.jpg'
                os.rename(src, trg)
            else:
                trg = train_folder + 'strep_' + str(i) + '.jpg'
                os.rename(src, trg)

        for i in range(len(healthy_names)):
            src = healthy_folder + healthy_names[i]
            if random.choices(mylist, weights=[5, 1])[0] == "test":
                trg = test_folder + 'healthy_' + str(i)
                os.rename(src, trg)
            else:
                trg = train_folder + 'healthy_' + str(i)
                os.rename(src, trg)

        test_names = [name for name in os.listdir(test_folder)]
        train_names = [name for name in os.listdir(train_folder)]
        '''
        for i in range(len(test_names)):
            src = test_folder + test_names[i]
            if "strep" in test_names[i]:
                trg = strep_folder + 'strep_' + str(i)
                os.rename(src,trg)
            else:
                trg = healthy_folder + 'healthy_' + str(i)
                os.rename(src,trg)
        '''
        for i in range(len(train_names)):
            src = train_folder + train_names[i]
            if "strep" in train_names[i]:
                trg = strep_folder + 'strep_' + str(i)
                os.rename(src, trg)
            else:
                trg = healthy_folder + 'healthy_' + str(i)
                os.rename(src, trg)

        healthy_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/data/healthy/'
        strep_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/data/strep/'
        strep_names = [name for name in os.listdir(strep_folder)]
        healthy_names = [name for name in os.listdir(healthy_folder)]

        healthy_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/data/healthy/'
        healthy_names = [name for name in os.listdir(healthy_folder)]
        batch_size = len(healthy_names)
        healthy = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
        img = torch.zeros(3, 256, 256, dtype=torch.uint8)
        downSample = nn.AvgPool2d((256, 256))

        p = transforms.Compose([transforms.Resize((256, 256))])

        for i in range(len(healthy_names)):
            img = torchvision.io.read_image(healthy_folder + healthy_names[i])
            img = p(img)
            healthy[i] = img

        strep_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/data/strep/'
        strep_names = [name for name in os.listdir(strep_folder)]
        batch_size = len(strep_names)
        strep = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

        for i in range(len(strep_names)):
            img = torchvision.io.read_image(strep_folder + strep_names[i])
            img = p(img)
            strep[i] = img

        healthy_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/data/healthy/'
        strep_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/data/strep/'
        strep_names = [name for name in os.listdir(strep_folder)]
        healthy_names = [name for name in os.listdir(healthy_folder)]

        for i in range(len(strep_names)):
            src = strep_folder + strep_names[i]
            image_to_transform = sk.io.imread(src)

            fourier_image = sp.ndimage.gaussian_filter(image_to_transform, 1, order=1)

            new_file_path = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/processed_data/strep/' + strep_names[
                i] + '.jpg'

            # write image to the disk
            io.imsave(new_file_path, fourier_image)

        for i in range(len(healthy_names)):
            src = healthy_folder + healthy_names[i]
            image_to_transform = sk.io.imread(src)

            fourier_image = sp.ndimage.gaussian_filter(image_to_transform, 1, order=1)

            new_file_path = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/processed_data/healthy/' + \
                            healthy_names[
                                i] + '.jpg'

            # write image to the disk
            io.imsave(new_file_path, fourier_image)

        test_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/test_data/'
        test_names = [name for name in os.listdir(test_folder)]

        for i in range(len(test_names)):
            src = test_folder + test_names[i]
            if "strep" in test_names[i]:
                image_to_transform = sk.io.imread(src)
                fourier_image = sp.ndimage.gaussian_filter(image_to_transform, 1, order=1)
                new_file_path = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/processed_test_data/strep/' + \
                                test_names[i] + '.jpg'
                io.imsave(new_file_path, fourier_image)
            else:
                image_to_transform = sk.io.imread(src)
                fourier_image = sp.ndimage.gaussian_filter(image_to_transform, 1, order=1)
                new_file_path = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/processed_test_data/healthy/' + \
                                test_names[i] + '.jpg'
                io.imsave(new_file_path, fourier_image)
