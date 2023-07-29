gpuDevice(1)
clear();
%% preprocess dataset

labelInfo                   =  dir("/media/airl/T7/matlab_project/data/train/label/*json");
imgFileName         = strings(length(labelInfo),1);
pill_bbox               = cell(length(labelInfo),1);

for ii = 1 : length(labelInfo)
    imgFileName(ii) = "/media/airl/T7/matlab_project/data/train/data/"+labelInfo(ii).name(1:length(labelInfo(ii).name)-5) + ".png";

    labelFile       = fullfile(labelInfo(ii).folder,labelInfo(ii).name);
    txt             = fileread(labelFile);
    labelsJSON      = jsondecode(txt);
    fields          = [fieldnames(labelsJSON)];

    if length(fields) == 4
        a = cat(1,reshape(labelsJSON.(fields{1}),[1,4]),reshape(labelsJSON.(fields{2}),[1,4]), ...
            reshape(labelsJSON.(fields{3}),[1,4]),reshape(labelsJSON.(fields{4}),[1,4]));
    
    elseif length(fields) == 3
        a = cat(1,reshape(labelsJSON.(fields{1}),[1,4]),reshape(labelsJSON.(fields{2}),[1,4]), ...
            reshape(labelsJSON.(fields{3}),[1,4]));
    
    elseif length(fields) == 2
        a = cat(1,reshape(labelsJSON.(fields{1}),[1,4]),reshape(labelsJSON.(fields{2}),[1,4]));
    
    else
        a = [reshape(labelsJSON.(fields{1}),[1,4])];
    end 
    pill_bbox{ii} = a+1;
end

pilldata    = table(imgFileName,pill_bbox);


rng(42);
shuffledIndices = randperm(height(pilldata));
idx             = floor(0.7 * length(shuffledIndices) );

trainingIdx     = 1:idx;
trainingDataTbl = pilldata(shuffledIndices(trainingIdx),:);

validationIdx       = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl   = pilldata(shuffledIndices(validationIdx),:);

testIdx         = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl     = pilldata(shuffledIndices(testIdx),:);


imdsTrain       = imageDatastore(trainingDataTbl{:,'imgFileName'});
bldsTrain       = boxLabelDatastore(trainingDataTbl(:,"pill_bbox"));

imdsValidation  = imageDatastore(validationDataTbl{:,"imgFileName"});
bldsValidation  = boxLabelDatastore(validationDataTbl(:,"pill_bbox"));

imdsTest        = imageDatastore(testDataTbl{:,"imgFileName"});
bldsTest        = boxLabelDatastore(testDataTbl(:,"pill_bbox"));

trainingData    = combine(imdsTrain,bldsTrain);
validationData  = combine(imdsValidation,bldsValidation);
testData        = combine(imdsTest,bldsTest);

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

inputSize = [416 416 3];
className = "pill_bbox";



trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));

numAnchors = 9;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);

area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    anchors(7:9,:)
    };

detector = yolov4ObjectDetector("csp-darknet53-coco",className,anchorBoxes,InputSize=inputSize);

augmentedTrainingData = transform(trainingDataForEstimation,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},"rectangle",data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,BorderSize=10)

tempdir = "checkpoints";
options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.0001,...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor = 0.2000, ...
    LearnRateDropPeriod = 4, ...
    MiniBatchSize=16,...
    L2Regularization=0.0005,...
    MaxEpochs=10,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=10,...
    CheckpointPath=tempdir,...
    ValidationData=validationData, ...
    Plots="training-progress", ...
    ValidationFrequency=200);


[detector, info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);

pretrained = load("checkpoints/net_checkpoint__5230__2023_06_16__23_42_21.mat");
detector = pretrained.net;

detectionResults = detect(detector,testData);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults,testData);

figure
plot(recall,precision)
xlabel("Recall")
ylabel("Precision")
grid on
title(sprintf("Average Precision = %.2f",ap))

pretrain = load("checkpoints/net_checkpoint__5230__2023_06_16__23_42_21.mat")
detector = pretrain.net

testDataTbl{1,1}
testDataTbl{2,1}


for i = 1:50
    image = imread(testDataTbl{i,1});
    
    bbox = detect(detector,image)
    annotatedImage = insertShape(image,"Rectangle",bbox);
    % figure
    % imshow(annotatedImage)
    path = "result_image/test_data_"+i+".png"
    imwrite(annotatedImage,path);
end

detect(detector,testData)

