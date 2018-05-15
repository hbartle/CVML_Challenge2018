%
%   HOG Feature Extraction
%
%
%
close all
clear
clc

%% Get the original Images and extract HOG Features
numTrainImages = 5830;
numValImages = 2298;
numTestImages = 3460;

rescaling = 1;
resolution = ceil(rescaling*256);

reload = 1;

disp('Load original training images...')
% Load Training Images
if exist('data/train/TrainImages/trainimages.mat','file') == 2 && reload==0
    load('data/train/TrainImages/trainimages.mat');
else
    %trainImagesOriginal = cell(1,numTrainImages);
    for i=1:numTrainImages
        img =  rgb2gray(imread(['data/train/TrainImages/Image' num2str(i) '.jpg']));
        [f, ~] = extractHOGFeatures(img);
        trainVectorsHOG(i,:) = f;
        i
    end
end
disp('Done!')

disp('Load original validation images...')
% Load Validation Images
if exist('data/validation/ValidationImages/valimages.mat','file') == 2 && reload==0
    load('data/validation/ValidationImages/valimages.mat');
else
    %valImagesOriginal = cell(1,numValImages);
    for i=1:numValImages
        img = rgb2gray(imread(['data/validation/ValidationImages/Image' num2str(i) '.jpg']));
        [f, ~] = extractHOGFeatures(img);
        valVectorsHOG(i,:) = f;
        i
    end
end
disp('Done!')

disp('Load original testing images...')
% Load Testing Images
if exist('data/test/TestImages/testimages.mat','file') == 2 && reload==0
    load('data/test/TestImages/testimages.mat');
else
    %testImagesOriginal = cell(1,numTestImages);
    for i=1:numTestImages
        img = rgb2gray(imread(['data/test/TestImages/Image' num2str(i) '.jpg']));
        [f,~] = extractHOGFeatures(img);
        testVectorsHOG(i,:)= f;
        i
    end
end
disp('Done!')

%% Safe the Features
save('data/train/trainVectorsHOG.mat','trainVectorsHOG')
save('data/validation/valVectorsHOG.mat','valVectorsHOG')
save('data/test/testVectorsHOG.mat','testVectorsHOG')
