# Benchmarks

Below benchmarks done for comparison of rx6600xt and gtx960 - GPUs 
of cuda and rocm backends vs `pytorch_ocl`

Depending on the network training performance is around 60 to 90 percent
inference performance is somewhat better.

Notes: time in ms per batch - smaller is better, input is standard imagenet
input Batchx3x224x224


## Training


    rx6600xt/8gb        batch size   rocm/hip     opencl    Raito %    
    alexnet                     64     57.848     82.381       70.2
    resnet18                    64    146.917    238.889       61.5
    resnet50                    32    266.441    357.985       74.4
    convnext_small              16    337.252    583.794       57.8
    vgg16                       16    206.312    348.692       59.2
    densenet161                 16    296.807    485.035       61.2
    mobilenet_v2                32    157.476    197.886       79.6
    mobilenet_v3_small          64     92.506    120.406       76.8
    mobilenet_v3_large          64    286.795    319.938       89.6
    resnext50_32x4d             32    336.464    491.112       68.5
    wide_resnet50_2             32    466.841    642.973       72.6
    mnasnet1_0                  32     159.97    167.306       95.6
    efficientnet_b0             32     205.69    305.157       67.4
    regnet_y_400mf              64    171.691    244.587       70.2

    Average                                                    71.8
                                                       
    gtx960/4gb          batch size c     cuda     opencl    Raito %    
    alexnet                     64    128.142    270.006       47.5
    resnet18                    64    415.589    746.578       55.7
    resnet50                    16    373.932    599.182       62.4
    convnext_small               8   1128.995   1175.585       96.0
    vgg16                        8    364.176    561.695       64.8
    densenet161                  8    463.427    728.693       63.6
    mobilenet_v2                16     173.13    352.728       49.1
    mobilenet_v3_small          32    101.621    206.353       49.2
    mobilenet_v3_large          32    263.055    523.575       50.2
    resnext50_32x4d             16    539.007     846.71       63.7
    wide_resnet50_2             16     677.57   1040.154       65.1
    mnasnet1_0                  16    167.542    322.004       52.0
    efficientnet_b0             16    241.023     540.09       44.6
    regnet_y_400mf              32    353.889    391.025       90.5

    Average                                                    61.0
                                                               
## Inference

Note, since my AMD and Nvidia gpus have different memory size differnet
batch sizes were used


    rx6600xt/8gb          rocm/hip     opencl    Ratio %    Batch=64   
    convnext_small         476.549    600.921       79.3
    alexnet                 24.587     26.311       93.4
    resnet18                41.375     59.375       69.7
    resnet50               165.261    194.512       85.0
    vgg16                  205.124    309.937       66.2
    densenet161             409.38    414.496       98.8
    inception_v3            90.635    131.685       68.8
    mobilenet_v2            77.691     93.701       82.9
    mobilenet_v3_small      22.203     26.151       84.9
    mobilenet_v3_large      63.229     70.458       89.7
    resnext50_32x4d        244.676    274.791       89.0
    wide_resnet50_2        320.313    402.687       79.5
    mnasnet1_0              74.141     75.162       98.6
    efficientnet_b0        104.396    114.898       90.9
    efficientnet_b4        303.468    276.226      109.9
    regnet_y_400mf          43.298     57.491       75.3

    Average                                         85.1
                                                       
    gtx960/4gb                cuda    opencl     Ratio %  Batch=32   
    convnext_small         751.713   1206.871       62.3
    alexnet                 29.446      44.27       66.5
    resnet18                66.053     93.352       70.8
    resnet50               214.787    316.754       67.8
    vgg16                  350.278    486.743       72.0
    densenet161            511.183    587.856       87.0
    inception_v3           167.233    217.664       76.8
    mobilenet_v2            86.572    161.797       53.5
    mobilenet_v3_small      27.748     49.359       56.2
    mobilenet_v3_large       68.79    121.644       56.6
    resnext50_32x4d        284.697    440.466       64.6
    wide_resnet50_2        376.114    587.801       64.0
    mnasnet1_0              82.576    132.463       62.3
    efficientnet_b0        111.154    202.593       54.9
    efficientnet_b4        299.779    499.841       60.0
    regnet_y_400mf          99.336     95.446      104.1

    Average                                         67.5                                                                                                                             
                                                                                                                                 
                                                                                                                             
                                                                                                                             
                                                                                                                             
                                                                                                                             
                                                                                                                             
                                                                                                                             
                                                                                                                                 
                                                                                                                                 
                                                                                                                                 
                                                                                                                                     
                                                                                                                                     
                                                                                                                                     
                                                                                                                                     
                                                                                                                                     
                                                                                                                                     
                                                                                                                                     
                                                                                                                                     
                                                                                                                                     
                                                                                                                                     
                                                                                                                                     
                                                                                                                                     
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
                                                                                                                                         
           ppppppppppppppppppp                                                                                                           
