# Setup

Tested 3 setups, pytorch 2.4

1. AMD rx6600XT, OpenCL drivers vs official ROCM pytorch (6.1)
2. NVidia rx960, OpenCL drivers vs official CUDA 12.2
3. Inter Arc A380, OpenCL NEO driver vs XPU - intel extension for pytorch (2.1 since it what was released)

Input is standard Image net batchx3x224x224, time in milliseconds, lower is better.

# Training



|AMD||||||Nvidia||||||Intel|||||
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|rx6600xt|batch|OpenCL|ROCM|% Perf||gtx960|batch|OpenCL|CUDA|% Perf||A380|batch|OpenCL|XPU|% Perf|
|alexnet|64|75.239|57.957|77||alexnet|64|257.09|130.561|51||alexnet|64|482.139|133.512|28|
|resnet18|64|238.927|147.099|62||resnet18|64|695.096|419.69|60||resnet18|64|1044.985|397.738|38|
|resnet50|32|358.872|266.155|74||resnet50|16|591.143|375.644|64||resnet50|16|640.916|329.849|51|
|convnext_small|16|608.297|337.736|56||convnext_small|8|1001.294|1120.676|112||convnext_small|8|841.302|259.292|31|
|vgg16|16|343.962|206.243|60||vgg16|8|520.75|363.288|70||vgg16|8|780.692|479.314|61|
|densenet161|16|494.175|297.001|60||densenet161|8|698.842|464.051|66||densenet161|8|834.207|423.883|51|
|mobilenet_v2|32|206.255|157.743|76||mobilenet_v2|16|335.279|173.748|52||mobilenet_v2|16|405.541|153.694|38|
|mobilenet_v3_small|64|130.571|92.83|71||mobilenet_v3_small|32|196.173|102.561|52||mobilenet_v3_small|32|275.302|92.086|33|
|mobilenet_v3_large|64|330.269|287.3|87||mobilenet_v3_large|32|497.168|264.072|53||mobilenet_v3_large|32|642.568|226.292|35|
|resnext50_32x4d|32|490.971|336.183|68||resnext50_32x4d|16|807.178|539.026|67||resnext50_32x4d|16|1068.918|396.39|37|
|wide_resnet50_2|32|643.083|468.04|73||wide_resnet50_2|16|1023.105|677.723|66||wide_resnet50_2|16|1373.346|634.213|46|
|mnasnet1_0|32|167.934|160.254|95||mnasnet1_0|16|302.854|167.911|55||mnasnet1_0|16|383.069|126.56|33|
|efficientnet_b0|32|313.972|205.674|66||efficientnet_b0|16|515.058|241.311|47||efficientnet_b0|16|531.724|203.157|38|
|regnet_y_400mf|64|246.069|171.841|70||regnet_y_400mf|32|361.507|353.584|98||regnet_y_400mf|32|635.279|224.228|35|
|Average||||71||Average||||65||Average||||40|

# Inference


|AMD||||||Nvidia||||||Intel|||||
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|rx6600xt|batch|OpenCL|ROCM|% Perf||gtx960|batch|OpenCL|CUDA|% Perf||A380|batch|OpenCL|XPU|% Perf|
|alexnet|64|24.543|24.642|100||alexnet|32|45.007|30.271|67||alexnet|32|55.5|25.835|47|
|resnet18|64|59.428|41.569|70||resnet18|32|94.044|66.61|71||resnet18|32|113.002|55.647|49|
|resnet50|64|196.75|165.706|84||resnet50|32|316.899|215.245|68||resnet50|32|271.778|145.842|54|
|convnext_small|64|632.215|478.088|76||convnext_small|32|881.586|751.286|85||convnext_small|32|670.291|294.405|44|
|vgg16|64|310.767|205.745|66||vgg16|32|490.68|351.488|72||vgg16|32|801.684|333.954|42|
|densenet161|64|415.707|410.906|99||densenet161|32|589.712|510.883|87||densenet161|32|685.154|315.407|46|
|mobilenet_v2|64|93.699|77.774|83||mobilenet_v2|32|162.4|87.376|54||mobilenet_v2|32|100.363|51.589|51|
|mobilenet_v3_small|64|25.653|22.253|87||mobilenet_v3_small|32|50.097|28.739|57||mobilenet_v3_small|32|36.92|26.508|72|
|mobilenet_v3_large|64|70.409|63.28|90||mobilenet_v3_large|32|122.416|69.432|57||mobilenet_v3_large|32|84.413|52.328|62|
|resnext50_32x4d|64|274.967|245.411|89||resnext50_32x4d|32|440.411|284.571|65||resnext50_32x4d|32|359.037|169.194|47|
|wide_resnet50_2|64|404.214|321.398|80||wide_resnet50_2|32|589.164|376.938|64||wide_resnet50_2|32|682.184|321.014|47|
|mnasnet1_0|64|75.027|74.211|99||mnasnet1_0|32|133.324|83.407|63||mnasnet1_0|32|91.441|51.785|57|
|efficientnet_b0|64|114.735|104.417|91||efficientnet_b0|32|203.531|111.822|55||efficientnet_b0|32|129.755|88.131|68|
|regnet_y_400mf|64|57.408|43.313|75||regnet_y_400mf|32|96.079|99.022|103||regnet_y_400mf|32|87.756|56.503|64|
|Average||||85||Average||||69||Average||||54|
