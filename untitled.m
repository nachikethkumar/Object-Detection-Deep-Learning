% % Create a Yolo2 object detection network
% inputSize = [450 450 3];
% 
% numClasses = 7;
% 
% network = resnet50();
% featureLayer = "activation_40_relu";
% 
% 
% preprocessedTrainingData = transform(dsTrain,@(data)resizeImageAndLabel(data, inputSize));

%ground_data=gTruth_at


%% %% Data Pre-processing
% Importing ground truth data from image labler app

if ~isfolder(fullfile("TrainingData"))
    mkdir TrainingData
end

trainingData = objectDetectorTrainingData(gTruth,'SamplingFactor',1,...
    'WriteLocation','TrainingData');
% Display first few rows of the data set.
%trainingData(1:4,:)

TDimg_DS=imageDatastore(trainingData.imageFilename);
BBOX_DS=boxLabelDatastore(trainingData(:,2:end))



%TDimg_DS = augmentedImageDatastore([450 450],TDimg_DS,'ColorPreprocessing','gray2rgb');
%Tdata1=combine(TDimg_DS,BBOX_DS)

%inputSize = [450 450];
%preprocessedTestData = transform(Tdata, @(Tdata)resizeImageAndLabel(Tdata, inputSize));
Tdata=combine(TDimg_DS,BBOX_DS)
scaledData_Tdata= transform(Tdata,@scaleGT)
%scaledData_Tdata=preprocessedTestData
%%% view the scaled dataset
% newGT = preview(scaledData_Tdata)
% im = insertObjectAnnotation(newGT{1},"rectangle",newGT{2},newGT{3});
% imshow(im)


%% Data Augmentation

augmentedTrainingData = transform(scaledData_Tdata, @augmentData);

%augmentedTrainingData = transform(Tdata1, @augmentData);
%augmentedTrainingData=scaledData_Tdata;


%% % Create YOLO architecture
%data = readall(augmentedTrainingData);
%anchorBoxes = estimateAnchorBoxes(data, 6);
anchorBoxes = estimateAnchorBoxes(scaledData_Tdata,6)

inputSize = [450 450 3];
numClasses = 6;
network = resnet50();
%network = resnet50('Weights','none');
%network=lgraph_1
featureLayer = "activation_40_relu";
lgraph = yolov2Layers(inputSize, numClasses, anchorBoxes, network, featureLayer); 

%for resnet18
% lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,net,...
%     "res5b_relu","ReorgLayerSource","res3a_relu")

%% Training YOLO v2 Object detector 

% opts = trainingOptions("rmsprop",...
%         InitialLearnRate=0.001,...
%         MiniBatchSize=4,...
%         MaxEpochs=10,...
%         LearnRateSchedule="piecewise",...
%         LearnRateDropPeriod=3,...
%         VerboseFrequency=30, ...
%         L2Regularization=0.001,...
%         ValidationData=dsVal,...
%         ValidationFrequency=50);

opts = trainingOptions("adam",InitialLearnRate=0.0001,MiniBatchSize=2,MaxEpochs=50,VerboseFrequency=2,Plots="training-progress",ExecutionEnvironment="gpu");

[detectorAugmented, infoAugmented] = trainYOLOv2ObjectDetector(augmentedTrainingData,lgraph, opts);


%% % Evaluvation of model

pathtoimages="D:\YOLO\Testing_data"
Testing_ds = imageDatastore(pathtoimages)
resizeTestImgs = augmentedImageDatastore([450 450],Testing_ds);

%preprocessedTestData = transform(dsTest, @(data)resizeImageAndLabel(data, inputSize));
%view data store images in figure window
%montage(Testing_ds)

test_images_extract = readimage(Testing_ds,1);
test_images_extract=imresize(test_images_extract,[450 450]);
imshow(test_images_extract)

detector=detectorAugmented
[dbox,dscore,dlabel] = detect(detector,test_images_extract)

detectedObjects = insertObjectAnnotation(test_images_extract, "rectangle", dbox, dlabel);
imshow(detectedObjects)


%% Evaluvation
%results = detect(detector, Testing_ds)


%% webcam import image
detector=detectorAugmented
cam=webcam(1);
figure;
while ishandle(1) % Check if the figure is still open
    % Acquire a frame from the webcam
    I = imresize((snapshot(cam)),[450,450]);


[bbox, score, label]  = detect(detector, I);
%Visualize the predictions by overlaying the detected bounding boxes on the
%image using the insertObjectAnnotation function.

imshow(I);
showShape("rectangle", bbox, Label=label);
drawnow;
end



