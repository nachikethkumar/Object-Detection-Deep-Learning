% Create YOLO v4 object detector.
detector = yolov4ObjectDetector("csp-darknet53-coco");

% Detect objects in an unknown image by using the detector.
img = imread("Train_64.png");
[bboxes,scores,labels] = detect(detector,img);

% Display the detection results.
detectedImg = insertObjectAnnotation(img,"Rectangle",bboxes,labels);
figure
imshow(detectedImg)