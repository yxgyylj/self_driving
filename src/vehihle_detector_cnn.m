%clear all; close all; clc;

% Create image input layer.
inputLayer = imageInputLayer([32 32 3], 'Name','input');

% Define the convolutional layer parameters.
filterSize = [3 3];
numFilters = 32;

% Create the middle layers.
middleLayers = [
convolution2dLayer(filterSize, numFilters, 'Padding', 1, 'Name','conv1')
reluLayer('Name', 'relu1')
convolution2dLayer(filterSize, numFilters, 'Padding', 1, 'Name','conv2')
reluLayer('Name', 'relu2')
maxPooling2dLayer(3, 'Stride',2, 'Name','mp')
];

finalLayers = [
% Add a fully connected layer with 64 output neurons. The output size
% of this layer will be an array with a length of 64.
fullyConnectedLayer(64, 'Name','fc1')
% Add a ReLU non-linearity.
reluLayer('Name', 'relu3')
% Add the last fully connected layer. At this point, the network must
% produce outputs that can be used to measure whether the input image
% belongs to one of the object classes or background. This measurement
% is made using the subsequent loss layers.
fullyConnectedLayer(width(vehicleDataset), 'Name','fc2')
% Add the softmax loss layer and classification layer.
softmaxLayer('Name', 'softmax')
classificationLayer('Name', 'classification')
];

layers = [
inputLayer
middleLayers
finalLayers
];
inputSize = [224 224 3];
numClasses = width(vehicleDataset)-1;
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(trainingData,numAnchors);
% featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
% lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,layers);


options = trainingOptions('sgdm',...
    'MaxEpochs',10,...
    'MiniBatchSize',1,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationData);

[detector, info] = trainFastRCNNObjectDetector(trainingData,layers,options);

% [detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
%     'NegativeOverlapRange',[0 0.3], ...
%     'PositiveOverlapRange',[0.6 1]);


% detector = trainFasterRCNNObjectDetector(trainingData, layers, options, ...
%     'NegativeOverlapRange', [0 0.3], ...
%     'PositiveOverlapRange', [0.6 1], ...
%     'BoxPyramidScale', 1.2);
