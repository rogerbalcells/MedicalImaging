import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix

import pdb

torch.set_printoptions(linewidth=120)

from collections import OrderedDict
from collections import namedtuple
from itertools import product

import time

from torch.utils.tensorboard import SummaryWriter

from IPython.display import clear_output

import json

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import itertools
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import cv2

data_dir = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/processed_data/'

train_set = datasets.ImageFolder(data_dir, transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Resize((120,120))
    ]))

image, label = train_set[0]

print(image.shape)

loader = torch.utils.data.DataLoader(train_set, batch_size=62, num_workers=1)
data = next(iter(loader))
mean = data[0].mean()
std = data[0].std()

train_set_normal = datasets.ImageFolder(
    data_dir
    ,transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1)
        ,transforms.ToTensor()
        ,transforms.Normalize(mean, std)
        ,transforms.Resize((120,120))
    ])
)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=5)

        self.fc1 = nn.Linear(in_features=6 * 11 * 11 + 100, out_features=112)
        self.fc2 = nn.Linear(in_features=112, out_features=56)
        self.fc3 = nn.Linear(in_features=56, out_features=12)
        self.out = nn.Linear(in_features=12, out_features=2)

    def forward(self, t):
        # (1) input layer
        t = t
        batch_size = t.size()[0]
        colorhist = [None] * batch_size
        for i in range(batch_size):
            colorhist[i] = torch.histc(t[i] * 255, bins=100, min=0, max=255).tolist()

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        # print(t.shape)
        t = t.reshape(-1, 6 * 11 * 11)
        tt = torch.empty(batch_size, t.size()[1] + 100).to('cuda')
        for i in range(batch_size):
            tt[i] = torch.cat((t[i], torch.Tensor(colorhist[i]).to('cuda')), 0)
        tt = self.fc1(tt)
        tt = F.relu(tt)

        # (5) hidden linear layer
        tt = self.fc2(tt)
        tt = F.relu(tt)

        # (5) hidden linear layer
        tt = self.fc3(tt)
        tt = F.relu(tt)

        # (6) output layer
        tt = self.out(tt)
        # t = F.softmax(t, dim=1)

        return tt

torch.set_grad_enabled(True)

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


class Epoch():
    # This Epoch class is useless at the moment
    def __init__(self):
        self.count = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = None


class RunManager():
    def __init__(self):

        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images.to(getattr(run, 'device', 'cpu')))

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        # Next two lines are only for Jupyter Notebook
        clear_output(wait=True)
        display(df)

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):

        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

'''torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([1000, 6, 12, 12], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(6, 12, kernel_size=[5, 5], padding=[0, 0], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().float()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()'''
print(len(train_set[0][0][0]))

torch.backends.cudnn.enabled = True
trainsets = {
    'not_normal': train_set
    , 'normal': train_set_normal
}

params = OrderedDict(
    lr=[0.001]
    , batch_size=[40]
    , shuffle=[True]
    , num_workers=[1]
    , device=['cuda']
    , trainset=['normal']
)

m = RunManager()

for run in RunBuilder.get_runs(params):

    device = torch.device(run.device)

    network = Network().to(device)

    # Training process given the set of parameters
    # num_workers preloads batches of data
    loader = torch.utils.data.DataLoader(
        trainsets[run.trainset]
        , batch_size=run.batch_size
        , shuffle=run.shuffle
        , num_workers=run.num_workers
    )

    optimizer = optim.Adam(
        network.parameters(), lr=run.lr
    )

    m.begin_run(run, network, loader)

    for epoch in range(50):
        m.begin_epoch()
        for batch in loader:
            images = batch[0].to(device)  # Get Batch
            labels = batch[1].to(device)
            preds = network(images)  # Pass Batch
            loss = F.cross_entropy(preds, labels)  # Calculate Loss
            optimizer.zero_grad()  # Zero Gradients
            loss.backward()  # Calculate Gradients
            optimizer.step()  # Update Weights

            m.track_loss(loss)
            m.track_num_correct(preds, labels)

        m.end_epoch()
    m.end_run()
m.save('results')

test_data_dir = '/home/rxb5452/Desktop/Deep Learning/Medical Imaging/processed_test_data/'

test_set = datasets.ImageFolder(test_data_dir, transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Resize((120,120))
    ]))

image, label = train_set[0]

print(image.shape)

loader = torch.utils.data.DataLoader(test_set, batch_size=62, num_workers=1)
data = next(iter(loader))
mean = data[0].mean()
std = data[0].std()

test_set_normal = datasets.ImageFolder(
    test_data_dir
    ,transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1)
        ,transforms.ToTensor()
        ,transforms.Normalize(mean, std)
        ,transforms.Resize((120,120))
    ])
)
loader = torch.utils.data.DataLoader(test_set_normal, batch_size=62, num_workers=1)

@torch.no_grad()
def get_all_preds(model, loader):
    model = model.to('cuda')
    all_preds = torch.tensor([]).to('cuda')
    labels_total = []
    count = 0
    count_num_correct = 0
    for batch in loader:
        images = batch[0].to('cuda') # Get Batch
        labels = batch[1].to('cuda')

        preds = model(images)
        all_preds = torch.cat(
            (all_preds.to('cuda'), preds.to('cuda'))
            ,dim=0
        )
        labels_total.append(labels)
        count += len(loader)
        count_num_correct += preds.argmax(dim=1).eq(labels).sum().item()
    test_accuracy = count_num_correct / len(loader.dataset)
    return [all_preds, labels_total, test_accuracy]

test1, test_labels, test_accuracy = get_all_preds(network, loader)
print(test_accuracy)

print(test1.argmax(dim=1))