if [ "$1" == "" ] || [ "$2" == "" ] || [ "$3" == "" ] 
then
    echo "/benchmark_train.sh device name (small|large)"
    exit 1
fi
device="$1"
name="$2"
size="$3"
base=./dlprimitives
to=60


if [ "$size" == large ]
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
elif [ "$size" == small ]
then
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
else
    echo "Invalid size"
fi

for netb in $nst
do
    net=$(echo $netb | cut -d, -f1)
    batch=$(echo $netb | cut -d, -f2)
    printf "%20s %2d " $net $batch
    timeout $to python $base/tools/validate_network.py --train --model=$net --benchmark --batch=$batch --device=$device | tail -n 1 | awk '{print $4}'
done | tee log-train-$name-$size-$(echo $device | tr ':' '_').txt

