if [ "$1" == "" ] || [ "$2" == "" ] || [ "$3" == "" ] 
then
    echo "/benchmark_train.sh device name batch_size"
    exit 1
fi
device="$1"
name="$2"
batch="$3"
base=./dlprimitives


logname=./$(basename $VIRTUAL_ENV)-test-$name-batch$batch-${device/:/_}.txt

for net in  alexnet \
            resnet18 \
            resnet50 \
            convnext_small \
            vgg16 \
            densenet161 \
            mobilenet_v2 \
            mobilenet_v3_small \
            mobilenet_v3_large \
            resnext50_32x4d \
            wide_resnet50_2 \
            mnasnet1_0 \
            efficientnet_b0 \
            regnet_y_400mf 
do
    echo -n "$net "
    OUTPUT=$(timeout 120 python $base/tools/validate_network.py --model=$net --benchmark --batch=$batch --device=$device)
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
done | tee $logname
