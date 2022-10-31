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


def make_batch(size):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirname = {"standard" : "samples", "small" : "small_samples", "large":"big_samples" }[size]
    dim = {"standard":224,"small":150,"large":256}[size]
    base_path = os.path.join(base_path,'tests/%s' % dirname)
    samples = {'cat':281,'dog':207,'parrot':87,'goldfish':1}
    data = np.zeros((4,3,dim,dim)).astype(np.float32)
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

def step(model,data,lables,opt_steps=0,iter_size=1,fwd=False,test=True):
    if test:
        model.eval()
    else:
        model.train()
    if fwd:
        return dict(output=_det(model(data)))
    optimizer = torch.optim.Adam(model.parameters())
    save_res={}
    for o in range(max(1,opt_steps)):
        optimizer.zero_grad()
        for k in range(iter_size):
            sm = torch.nn.LogSoftmax(dim=1)
            nll = torch.nn.NLLLoss()
            res = model(data)
            #with torch.no_grad():
            #    print(torch.argmax(res,dim=1).to('cpu'))
            #    print(res[:,0:8].to('cpu'))
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

def train_on_images(model,batch,device,test,iter_size = 1,opt_steps = 0,fwd=False):
    data,labels = batch
    data_dev = data.to(device)
    labels_dev = labels.to(device)

    with torch.no_grad():
        state  = copy.deepcopy(model.state_dict())
    model.to(device)
    calc = step(model,data_dev,labels_dev,opt_steps,iter_size,fwd=fwd,test=test)
    model.to('cpu')
    with torch.no_grad():
        model.load_state_dict(state)
    ref = step(model,data,labels,opt_steps,iter_size,fwd=fwd,test=test)
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
    if max_diff > 1e-2:
        print("    FAIL     od=%0.5f md=%0.5f"% (output_diff,max_diff))
    else:
        print("    Ok       od=%0.5f md=%0.5f"% (output_diff,max_diff))


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
    print("Testing ",args.model)
    def _sn():
        from shufflenetv2 import shufflenet_v2_x1_0 
        return shufflenet_v2_x1_0(pretrained=True)
    models = dict(
        mnist_mlp = MnistNetMLP,
        mnist_cnn = MnistNetConv,
        mnist_bn = MnistNetBN,
    )

    if args.model in models:
        m=models[args.model]()
        batch = make_mnist_batch()
    else:
        weights = 'IMAGENET1K_V1' if args.pretrained else None
        if args.model == 'sn':
            m=_sn() 
        elif args.model.find('segmentation.')==0:
            m=getattr(torchvision.models.segmentation,args.model[len('segmentation.'):])(aux_loss=False,weights=weights)
        else:
            m=getattr(torchvision.models,args.model)(weights=weights)
        batch  = make_batch(args.size)

    if args.eval:
        m.eval()
    else:
        m.train()

    train_on_images(m,batch,args.device,args.eval,iter_size = args.iter_size,opt_steps = args.opt,fwd=args.fwd)

if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--all',default=False,action='store_true')
    p.add_argument('--opt',default=0,type=int,help='Optimizer steps')
    p.add_argument('--iter-size',default=1,type=int,help='Number of mini batches in iteration')
    p.add_argument('--model',default='resnet18')
    p.add_argument('--size',default='standard',choices=['standard','small','large'])
    p.add_argument('--fwd',default=False,action='store_true')
    p.add_argument('--device',default='cuda')
    p.add_argument('--eval',default=False,action='store_true')
    p.add_argument('--pretrained',type=bool,default=True)
    r = p.parse_args()
    if r.device.find('ocl')==0:
        if os.name == 'nt':
            torch.ops.load_library(r"build\pt_ocl.dll")
        else:
            torch.ops.load_library("build/libpt_ocl.so")
        try:
            torch.utils.rename_privateuse1_backend('ocl')
        except:
            r.device = r.device.replace('ocl','privateuseone')
    if r.all:
        ocl_blacklist = []
        for net in [ 
            dict(model='mnist_mlp'),
            dict(model='mnist_cnn'),
            dict(model='mnist_bn',iter_size = 2, opt = 5),
            dict(model='alexnet',eval=True),
            dict(model='resnet18'),
            dict(model='resnet50'),
            dict(model='vgg16',eval=True),
            dict(model='squeezenet1_0',eval=True),
            dict(model='densenet161'),
            dict(model='inception_v3',fwd=True,eval=True),
            dict(model='googlenet',eval=True),
            dict(model='shufflenet_v2_x1_0',eval=True),
            dict(model='mobilenet_v2',eval=True),
            dict(model='mobilenet_v3_large',eval=True),
            dict(model='mobilenet_v3_small',eval=True,fwd=True), # fails for cuda bwd as well
            dict(model='resnext50_32x4d'),
            dict(model='wide_resnet50_2'),
            dict(model='mnasnet1_0',eval=True),
            dict(model='efficientnet_b0',eval=True),
            dict(model='efficientnet_b4',eval=True),
            dict(model='regnet_y_400mf')
            ]:
            if net['model'] in ocl_blacklist and (r.device.find('ocl')==0 or r.device.find('private')==0):
                print(net['model'],"is blacklisted")
                continue
            new_r=copy.deepcopy(r)
            for n in net:
                setattr(new_r,n,net[n])
            try:
                main(new_r)
            except Exception as e:
                print("Fail",str(e))
    else:
        main(r)
