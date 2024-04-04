batch=64
device='privateuseone:0'
if [ "$1" == "" ]
then 
    base=./dlprimitives/
else
    base="$1"
fi

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
    echo -n "$net "
    OUTPUT=$(timeout 60 python $base/tools/validate_network.py --model=$net --benchmark --batch=$batch --device=$device)
    retVal=$?
    if [ $retVal -eq 124 ]
    then
      echo 'Error: Time exceeded!'
    else
      if [ $retVal -ne 0 ]
      then
        echo -n "Error: "
        echo "$OUTPUT" | tail -n 1
      else
        echo "$OUTPUT" | tail -n 1 | awk '{print $4}'
      fi
    fi
done
