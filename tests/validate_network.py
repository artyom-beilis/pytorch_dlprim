import torch
import copy
import torchvision
import json
import os
import PIL
import argparse
import time
import numpy as np
import sys
import torch.nn.functional as F
import torch.nn as nn


def make_batch():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(base_path,'tests/samples')
    samples = {'cat':281,'dog':207,'parrot':87,'goldfish':1}
    data = np.zeros((4,3,224,224)).astype(np.float32)
    labels = np.zeros(4).astype(np.int64)
    mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).astype(np.float32)
    N=0
    for name in samples:
        lbl = samples[name]
        path = base_path +'/' + name + '.ppm'
        img = PIL.Image.open(path)
        npimg = np.array(img).astype(np.float32) * (1.0 / 255)
        fact = 1.0 / np.array(std)
        off  = -np.array(mean) * fact
        for k in range(3):
            data[N,k,:,:] = npimg[:,:,k] * fact[k] + off[k]
        labels[N]=lbl
        N+=1

    return torch.from_numpy(data),torch.from_numpy(labels)


def get_grads(model,and_state):
    with torch.no_grad():
        calc = {}
        for name,param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                continue
            calc[name]=param.grad.detach().to('cpu').numpy()
        if and_state:
            sd = model.state_dict()
            for name in sd:
                calc['state_' + name] = sd[name].detach().to('cpu').numpy()
    return calc

def _det(t):
    return t.detach().to('cpu').numpy()

def step(model,data,lables,opt_steps=0,iter_size=1):
    optimizer = torch.optim.Adam(model.parameters())
    save_res={}
    for o in range(max(1,opt_steps)):
        optimizer.zero_grad()
        for k in range(iter_size):
            sm = torch.nn.LogSoftmax(dim=1)
            nll = torch.nn.NLLLoss()
            res = model(data)
            if not isinstance(res,torch.Tensor):
                for i,n in enumerate(res):
                    name = 'output' if i == 0 else 'output_' + n
                    save_res[name] = _det(res[n])
            else:
                save_res['output'] = _det(res)
                res = sm(res)
                loss=nll(res,lables)
                loss.backward()
        if opt_steps > 0:
            optimizer.step()
    r = get_grads(model,and_state=(opt_steps > 0))

    r.update(save_res)
    return r

def train_on_images(model,batch,device,test,iter_size = 1,opt_steps = 0):
    data,labels = batch
    data_dev = data.to(device)
    labels_dev = labels.to(device)

    if test:
        model.eval()
    else:
        model.train()
    
    state  = copy.deepcopy(model.state_dict())
    model.to(device)
    calc = step(model,data_dev,labels_dev,opt_steps,iter_size)
    model.to('cpu')
    model.load_state_dict(state)
    ref = step(model,data,labels,opt_steps,iter_size)
    max_diff = 0
    max_name = None
    for name in ref:
        d_cal = calc[name]
        d_ref = ref[name]
        diff = np.abs(d_cal - d_ref)
        md = np.max(diff)
        if name == 'output':
            output_diff=md
        if md > max_diff:
            max_diff = md
            max_name = name
    print("Output distance",output_diff)
    print("Max dispance",max_diff,"on",max_name)


class MnistNetConv(torch.nn.Module):
    def __init__(self):
        super(MnistNetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=0,bias=True)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=0)
        self.fc1 = nn.Linear(5*5*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        return output

class MnistNetBN(torch.nn.Module):
    def __init__(self):
        super(MnistNetBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x1 = self.conv3(x)
        x1 = self.bn3(x1)
        x  = x1 + x
        x  = F.relu(x)
        x = torch.mean(x,axis=(2,3),keepdim=True)
        x = torch.flatten(x,1)
        output = self.fc1(x)
        return output

class MnistNetMLP(torch.nn.Module):
    def __init__(self):
        super(MnistNetMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(torch.flatten(x,1))
        x = F.relu_(x)
        output = self.fc2(x)
        return output


def make_mnist_batch():
    return torch.randn(4,1,28,28),torch.randint(10,size=(4,))


def main(args):

    models = dict(
        mnist_mlp = MnistNetMLP,
        mnist_cnn = MnistNetConv,
        mnist_bn = MnistNetBN
    )

    if args.model in models:
        m=models[args.model]()
        batch = make_mnist_batch()
    else:
        if args.model.find('segmentation.')==0:
            m=getattr(torchvision.models.segmentation,args.model[len('segmentation.'):])(pretrained = args.pretrained,aux_loss=False)
        else:
            m = getattr(torchvision.models,args.model)(pretrained = args.pretrained)
        batch  = make_batch()

    if args.eval:
        m.eval()
    else:
        m.train()

    train_on_images(m,batch,args.device,args.eval,iter_size = args.iter_size,opt_steps = args.opt)

if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--opt',default=0,type=int,help='Optimizer steps')
    p.add_argument('--iter-size',default=1,type=int,help='Number of mini batches in iteration')
    p.add_argument('--model',default='resnet18')
    p.add_argument('--device',default='cuda')
    p.add_argument('--eval',default=False)
    p.add_argument('--pretrained',type=bool,default=True)
    r = p.parse_args()
    if r.device.find('opencl')==0:
        torch.ops.load_library("build/libpt_ocl.so")
    main(r)
