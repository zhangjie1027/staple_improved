time = 0;
name = "tiny-yolov4-coco";
% name = "csp-darknet53-coco";
% name = 'darknet53-coco';
detector = yolov4ObjectDetector(name);
img = imread("E:\DataSets\CFTracker\People\Human4\img\0493.jpg");
% img = imread([img_path params.img_files{5}]);
tic()
[bboxes,scores,labels] = detect(detector,img);
toc()
detectedImg = insertObjectAnnotation(img,"Rectangle",bboxes,labels);
figure
imshow(detectedImg)
% fprintf("time:%10.f",double(time))