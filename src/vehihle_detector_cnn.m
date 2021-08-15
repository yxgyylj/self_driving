
% 搭建输入层。其维数和图片的大小相同，
% 是 32（横轴） * 32（纵轴） * 3（RGB 三种颜色）
inputLayer = imageInputLayer([32 32 3], 'Name','input');
​
% 搭建中间层（隐藏层）。一般分为三个步骤：
% 卷积: 可以理解为对图片加滤镜，滤镜种类由所谓的"核函数确定"；
% 激活：对特征张量进行一定程度的归一化，这也是神经网络中非线性的主要来源；
% 池化：把特征张量进行简单的降维处理，减轻摄像头的学业负担；
filterSize = [3 3];
numFilters = 32;
middleLayers = [
convolution2dLayer(filterSize, numFilters, 'Padding', 1, 'Name','conv1')
reluLayer('Name', 'relu1')
convolution2dLayer(filterSize, numFilters, 'Padding', 1, 'Name','conv2')
reluLayer('Name', 'relu2')
maxPooling2dLayer(3, 'Stride',2, 'Name','mp')
];
​
% 搭建输出层
finalLayers = [
fullyConnectedLayer(64, 'Name','fc1')
reluLayer('Name', 'relu3')
fullyConnectedLayer(width(vehicleDataset), 'Name','fc2')
softmaxLayer('Name', 'softmax')
classificationLayer('Name', 'classification')
];

layers = [
inputLayer
middleLayers
finalLayers
];

options = trainingOptions('sgdm',...
    'MaxEpochs',10,...
    'MiniBatchSize',1,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationData,...
    'ExecutionEnvironment', 'auto');

[detector, info] = trainFastRCNNObjectDetector(trainingData,layers,options);

