import torch
import torchvision
import json
import os
import PIL
import argparse
import time
import numpy as np
import sys


def make_batch():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(base_path)
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


def get_grads(model):
    with torch.no_grad():
        calc = {}
        for name,param in model.named_parameters():
            if not param.requires_grad:
                continue
            calc[name]=param.grad.detach().to('cpu').numpy()
    return calc

def step(model,data,lables):
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    sm = torch.nn.LogSoftmax(dim=1)
    nll = torch.nn.NLLLoss()
    res = model(data)
    save_res = res.detach().to('cpu').numpy()
    res = sm(res)
    loss=nll(res,lables)
    loss.backward()
    r = get_grads(model)
    r['output'] = save_res
    return r

def train_on_images(model,device):
    data,labels = make_batch()
    data_dev = data.to(device)
    labels_dev = labels.to(device)

    model.train()

    calc = step(model,data_dev,labels_dev)
    model.to('cpu')
    ref = step(model,data,labels)
    max_diff = 0
    max_name = None
    for name in calc:
        d_cal = calc[name]
        d_ref = ref[name]
        diff = np.abs(d_cal - d_ref)
        md = np.max(diff)
        if md > max_diff:
            max_diff = md
            max_name = name
        print(name,np.max(diff))
    print("Max dispance",max_diff,"on",max_name)



def main(args):
    m = getattr(torchvision.models,args.model)(pretrained = args.pretrained)
    m.to(args.device)
    train_on_images(m,args.device)

if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--model',default='resnet18')
    p.add_argument('--device',default='cuda')
    p.add_argument('--pretrained',type=bool,default=True)
    r = p.parse_args()
    if r.device.find('opencl')==0:
        torch.ops.load_library("build/libpt_ocl.so")
    main(r)
