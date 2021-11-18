batch=64
device='opencl:0'
base="$1"
for net in  alexnet \
            resnet18 \
            vgg16 \
            densenet161 \
            inception_v3 \
            mobilenet_v2 \
            mobilenet_v3_small \
            mobilenet_v3_large \
            resnext50_32x4d \
            wide_resnet50_2 \
            mnasnet1_0 \
            efficientnet_b0 \
            efficientnet_b4 \
            regnet_y_400mf 
do
    echo $net $(timeout 60 python $base/tools/validate_network.py --model=$net --benchmark --batch=$batch --device=$device | tail -n 1 | awk '{print $4}')
done
