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


def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}

folder_path = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/pharyngitis/'
num_files_desired = 1000

# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    # random num of transformation to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1

    new_file_path = '%s/strep_augmented_image_%s.jpg' % (folder_path, num_generated_files)

    # write image to the disk
    io.imsave(new_file_path, transformed_image)
    num_generated_files += 1

healthy_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/no_pharyngitis/'
healthy_names = [name for name in os.listdir(healthy_folder)]
batch_size = len(healthy_names)
healthy = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
img = torch.zeros(3, 256, 256, dtype=torch.uint8)
downSample = nn.AvgPool2d((256, 256))


p = transforms.Compose([transforms.Resize((256,256))])

for i in range(len(healthy_names)):
    img = torchvision.io.read_image(healthy_folder + healthy_names[i])
    img = p(img)
    healthy[i] = img

strep_folder='/home/rxb5452/Desktop/Deep Learning/Medical Imaging/pharyngitis/'
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
        trg = test_folder + 'strep_' + str(i)
        os.rename(src, trg)
    else:
        trg = train_folder + 'strep_' + str(i)
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
        os.rename(src,trg)
    else:
        trg = healthy_folder + 'healthy_' + str(i)
        os.rename(src,trg)

healthy_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/data/healthy/'
strep_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/data/strep/'
strep_names = [name for name in os.listdir(strep_folder)]
healthy_names = [name for name in os.listdir(healthy_folder)]

for i in range(len(strep_names)):
    src = strep_folder + strep_names[i]
    trg = src + '.jpg'
    os.rename(src,trg)

for i in range(len(healthy_names)):
    src = healthy_folder + healthy_names[i]
    trg = src + '.jpg'
    os.rename(src,trg)

healthy_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/data/healthy/'
healthy_names = [name for name in os.listdir(healthy_folder)]
batch_size = len(healthy_names)
healthy = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
img = torch.zeros(3, 256, 256, dtype=torch.uint8)
downSample = nn.AvgPool2d((256, 256))


p = transforms.Compose([transforms.Resize((256,256))])

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

    new_file_path = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/processed_data/healthy/' + healthy_names[
        i] + '.jpg'

    # write image to the disk
    io.imsave(new_file_path, fourier_image)

test_folder = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/test_data/'
test_names = [name for name in os.listdir(test_folder)]

for i in range(len(test_names)):
    src = test_folder + test_names[i]
    if "strep" in test_names[i]:
        image_to_transform = sk.io.imread(src)
        fourier_image = sp.ndimage.gaussian_filter(image_to_transform, 1, order = 1)
        new_file_path = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/processed_test_data/strep/'+ test_names[i] +'.jpg'
        io.imsave(new_file_path, fourier_image)
    else:
        image_to_transform = sk.io.imread(src)
        fourier_image = sp.ndimage.gaussian_filter(image_to_transform, 1, order = 1)
        new_file_path = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/processed_test_data/healthy/'+ test_names[i] +'.jpg'
        io.imsave(new_file_path, fourier_image)
