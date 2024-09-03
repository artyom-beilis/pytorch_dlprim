device='ocl:0'
base="$1"
to=60


if [ "$2" != "" ]
then
        device="$2"
fi

if true
then
    nst='alexnet,64 
        resnet18,64 
        resnet50,32 
        convnext_small,16 
        vgg16,16 
        densenet161,16 
        mobilenet_v2,32 
        mobilenet_v3_small,64 
        mobilenet_v3_large,64 
        resnext50_32x4d,32 
        wide_resnet50_2,32 
        mnasnet1_0,32 
        efficientnet_b0,32 
        regnet_y_400mf,64'
else
    nst='alexnet,64 
        resnet18,64 
        resnet50,16 
        convnext_small,8 
        vgg16,8 
        densenet161,8 
        mobilenet_v2,16 
        mobilenet_v3_small,32 
        mobilenet_v3_large,32 
        resnext50_32x4d,16 
        wide_resnet50_2,16 
        mnasnet1_0,16 
        efficientnet_b0,16 
        regnet_y_400mf,32'
fi
for netb in $nst
do
    net=$(echo $netb | cut -d, -f1)
    batch=$(echo $netb | cut -d, -f2)
    printf "%20s %2d " $net $batch
    timeout $to python $base/tools/validate_network.py --train --model=$net --benchmark --batch=$batch --device=$device | tail -n 1 | awk '{print $4}'
done | tee log-train-$(echo $device | tr ':' '_').txt

