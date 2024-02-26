% deep learning Network

%deepnet=googlenet;

%Get training images and process them using datastores.
pathToImages = "/CourseData/Flowers5/";
flower_ds = imageDatastore(pathToImages,"IncludeSubfolders",true,"LabelSource","foldernames");
[trainImgs,testImgs] = splitEachLabel(flower_ds,0.6);
resizeTrainImgs = augmentedImageDatastore([224 224],trainImgs);
resizeTestImgs = augmentedImageDatastore([224 224],testImgs);
numClasses = numel(categories(flower_ds.Labels));

%Create a network by modifying GoogLeNet. First, launch the Deep Network Designer.
deepNetworkDesigner

%Then replace the fullyConnectedLayer (Convolution and Fully Connected) and the classificationLayer (Output).
load untrainedNetwork

%Set training algorithm options.
%The default number of epochs is 30, which means the entire training data set goes through the network 30 times. To reduce training time on the CPU, the number of epochs is limited to 1. The verbose frequency is also decreased so that you see updates more frequently.
opts = trainingOptions("sgdm","InitialLearnRate",0.001,"MaxEpochs",1,"VerboseFrequency",2);

%Perform training by passing all three inputs to the trainNetwork function. The network training should take about two minutes on a CPU.
[flowernet,info] = trainNetwork(resizeTrainImgs,lgraph,opts)
