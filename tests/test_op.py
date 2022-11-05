import torch
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


def get_diff(cpu,dev):
    c_dev = dev.to('cpu')
    r = torch.max(torch.abs(cpu - c_dev)).item()
    if r > 1e-4:
        print(cpu)
        print(c_dev)
    return r

def test_fwd(inputs,call,device):
    xs_cpu = []
    xs_dev = []
    with torch.no_grad():
        for s,limit in inputs:
            if limit <= 0:
                x_cpu = torch.randn(s)
                x_dev = x_cpu.to(device)
            else:
                x_cpu = torch.randint(limit,s)
                x_dev = x_cpu.to(device)
            xs_cpu.append(x_cpu)
            xs_dev.append(x_dev)

    y_cpu = call(*xs_cpu)
    y_dev = call(*xs_dev)

    if get_diff(y_cpu,y_dev) > 1e-6:
        raise Exception("Diff too big")
    print("Ok")

def test_fwd_bwd_op(inputs,call,device,randgen=torch.randn):
    xs_cpu = []
    xs_dev = []
    p_names = set()
    with torch.no_grad():
        p_names = set(call.state_dict())

    with torch.no_grad():
        for s,limit in inputs:
            if limit <= 0:
                x_cpu = randgen(s)
                x_dev = x_cpu.to(device)
                x_cpu.requires_grad = True
                x_dev.requires_grad = True
            else:
                x_cpu = torch.randint(limit,s)
                x_dev = x_cpu.to(device)
            xs_cpu.append(x_cpu)
            xs_dev.append(x_dev)


    y_cpu = call(*xs_cpu)

    with torch.no_grad():
        dy_cpu = torch.randn(y_cpu.shape)
        dy_dev = dy_cpu.to(device)
    y_cpu.backward(dy_cpu,retain_graph=True)

    dW_cpu = {}
    for n in p_names:
        dW_cpu[n] = call.get_parameter(n).grad.detach().clone()

    call.zero_grad()
    call = call.to(device)
    y_dev = call(*xs_dev)
    y_dev.backward(dy_dev,retain_graph=True)
    dW_dev = {}
    for n in p_names:
        dW_dev[n] = call.get_parameter(n).grad.detach().clone()


    with torch.no_grad():
        diffs = []
        diffs.append(('y',get_diff(y_cpu,y_dev)))
        for i in range(len(inputs)):
            if inputs[i][1] <= 0:
                diffs.append(('x%d' % i ,get_diff(xs_cpu[i].grad,xs_dev[i].grad)))
        for name in p_names:
            diffs.append(('p_' + name,get_diff(dW_cpu[name],dW_dev[name])))

        diffs.sort(key=lambda x:x[1],reverse=True)

    for name,diff in diffs:
        print("%10s %f" % (name,diff))
    max_diff = diffs[0][1]
    if max_diff > 1e-3:
        raise Exception("Diff too big")


def test_fwd_bwd(inputs,call,device,randgen=torch.randn):
    xs_cpu = []
    xs_dev = []
    with torch.no_grad():
        for s,limit in inputs:
            if limit <= 0:
                x_cpu = randgen(s)
                x_dev = x_cpu.to(device)
                x_cpu.requires_grad = True
                x_dev.requires_grad = True
            else:
                x_cpu = torch.randint(limit,s)
                x_dev = x_cpu.to(device)
            xs_cpu.append(x_cpu)
            xs_dev.append(x_dev)

    y_cpu = call(*xs_cpu)
    y_dev = call(*xs_dev)

    print(y_cpu.shape)
    print(y_dev.shape)

    if y_cpu.shape:
        with torch.no_grad():
            dy_cpu = torch.randn(y_cpu.shape)
            dy_dev = dy_cpu.to(device)
        y_cpu.backward(dy_cpu,retain_graph=True)
        y_dev.backward(dy_dev,retain_graph=True)
    else:
        y_cpu.backward(retain_graph=True)
        y_dev.backward(retain_graph=True)


    with torch.no_grad():
        diffs = []
        diffs.append(('y',get_diff(y_cpu,y_dev)))
        for i in range(len(inputs)):
            if inputs[i][1] <= 0:
                diffs.append(('x%d' % i ,get_diff(xs_cpu[i].grad,xs_dev[i].grad)))

        diffs.sort(key=lambda x:x[1],reverse=True)

    for name,diff in diffs:
        print("%10s %f" % (name,diff))
    max_diff = diffs[0][1]
    if max_diff > 1e-3:
        raise Exception("Diff too big")
    

def test_all(device):
    print("Mean 1d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=0,keepdim=True),device)
    print("Mean 2d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=(1,2),keepdim=True),device)
    print("Mean 1d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=0,keepdim=False),device)
    print("Mean 2d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x,dim=(0,2),keepdim=False),device)
    print("Mean all squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.mean(x),device)

    print("Sum 1d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=0,keepdim=True),device)
    print("Sum 2d")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=(1,2),keepdim=True),device)
    print("Sum 1d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=0,keepdim=False),device)
    print("Sum 2d squeeze")
    test_fwd_bwd([([2,3,4],-1)],lambda x:torch.sum(x,dim=(0,2),keepdim=False),device)

    print("LogSoftmax")
    test_fwd_bwd([([4,3],-1)],torch.nn.LogSoftmax(dim=1),device)
    print("NLLLoss");
    test_fwd_bwd([([4,3],-1),([4],3)],torch.nn.NLLLoss(),device)
    print("AAPool2d")
    test_fwd_bwd([([4,8,2,2],-1)],torch.nn.AdaptiveAvgPool2d((1,1)),device)
    print("Abs")
    test_fwd_bwd([([4,3],-1)],torch.abs,device)
    print("Abs_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.abs_(x*1.0),device)
    print("Hardtanh")
    test_fwd_bwd([([4,3],-1)],torch.nn.Hardtanh(),device)
    print("Hardtanh_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.Hardtanh(inplace=True)(x*1.0),device)
    print("Sigmoid")
    test_fwd_bwd([([4,3],-1)],torch.nn.Sigmoid(),device)
    print("Sigmoid_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.sigmoid_(x*1.0),device)
    print("Hardsigmoid")
    test_fwd_bwd([([4,3],-1)],torch.nn.Hardsigmoid(),device)
    print("Hardsigmoid_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.Hardsigmoid(inplace=True)(x*1.0),device)
    print("ReLU")
    test_fwd_bwd([([4,3],-1)],torch.nn.ReLU(),device)
    print("ReLU_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.ReLU(inplace=True)(x*1.0),device)
    print("LReLu")
    test_fwd_bwd([([4,3],-1)],torch.nn.LeakyReLU(),device)
    print("LReLU_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.LeakyReLU(inplace=True)(x*1.0),device)
    print("Tanh")
    test_fwd_bwd([([4,3],-1)],torch.nn.Tanh(),device)
    print("Tanh_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.tanh_(x*1.0),device)
    print("SiLU")
    test_fwd_bwd([([4,3],-1)],torch.nn.SiLU(),device)
    print("SiLU_")
    test_fwd_bwd([([4,3],-1)],lambda x:torch.nn.SiLU(inplace=True)(x*1.0),device)
    #print("ChannelShuffle")
    #test_fwd_bwd([([3, 4, 2, 2],-1)],torch.nn.ChannelShuffle(2),device)
    print("BCE Loss")
    test_fwd_bwd([([4,3,5],-1),([4,3,5],-1)],torch.nn.BCELoss(),device,torch.rand)
    print("BCE Loss no reduction")
    test_fwd_bwd([([4,3,5],-1),([4,3,5],-1)],torch.nn.BCELoss(reduction='none'),device,torch.rand)
    print("MSE Loss")
    test_fwd_bwd([([4,3,5],-1),([4,3,5],-1)],torch.nn.MSELoss(),device,torch.rand)
    print("MSE Loss no reduction")
    test_fwd_bwd([([4,3,5],-1),([4,3,5],-1)],torch.nn.MSELoss(reduction='none'),device,torch.rand)
    print("Min")
    test_fwd([([4,3,5],-1)],torch.min,device)
    print("Max")
    test_fwd([([4,3,5],-1)],torch.max,device)
    print("Dot")
    test_fwd([([16],-1),([16],-1)],torch.dot,device)
    print("Clamp 1")
    test_fwd([([4,3,5],-1)],lambda x:torch.clamp(x,min=-0.2,max=0.3),device)
    print("Clamp 2")
    test_fwd([([4,3,5],-1)],lambda x:torch.clamp(x,min=-0.2),device)
    print("Clamp 3")
    test_fwd([([4,3,5],-1)],lambda x:torch.clamp(x,max=0.3),device)

    print("Linear 2d")
    test_fwd_bwd_op([([8,10],-1)],torch.nn.Linear(10,5),device)
    print("Linear 3d")
    test_fwd_bwd_op([([2,6,10],-1)],torch.nn.Linear(10,5),device)

    print("Conv")
    test_fwd_bwd_op([([2,6,10,20],-1)],torch.nn.Conv2d(6,8,[3,5],stride=[1,2],padding=[1,2],dilation=1,groups=2),device)
    print("ConvTr")
    test_fwd_bwd_op([([2,6,10,20],-1)],torch.nn.ConvTranspose2d(6,8,[3,5],stride=[1,2],padding=[1,2],dilation=1,groups=2),device)
    print("ConvTr pad")
    test_fwd_bwd_op([([2,6,10,20],-1)],torch.nn.ConvTranspose2d(6,8,[3,5],stride=[1,2],padding=[1,2],output_padding=[0,1],dilation=1,groups=2),device)
    print("ConvTr")
    test_fwd_bwd_op([([2,6,10,20],-1)],torch.nn.ConvTranspose2d(6,8,[3,5],stride=[1,2],padding=[1,2],dilation=1,groups=1,bias=False),device)



if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--device',default='ocl:0')
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
    test_all(r.device)
