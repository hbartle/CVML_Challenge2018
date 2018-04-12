%
%   SIFT Feature Extraction
%  
%
%
close all
clc



VLFEATROOT = 'lib/vlfeat-0.9.21';
addpath([VLFEATROOT,'/toolbox'])
vl_setup()

%% Get the original Images
numTrainImages = 5830;
numValImages = 2298;
numTestImages = 3460;

rescaling = 0.5;
resolution = ceil(rescaling*256);


% Load Training Images
trainImagesOriginal = cell(1,numTrainImages);
for i=1:numTrainImages
    img = rgb2gray(imread(['data/train/TrainImages/Image' num2str(i) '.jpg']));
    img = imresize(img,rescaling);
    trainImagesOriginal{i} = img;
end

% Load Validation Images
valImagesOriginal = cell(1,numValImages);
for i=1:numValImages
    img = rgb2gray(imread(['data/validation/ValidationImages/Image' num2str(i) '.jpg']));
    img = imresize(img,rescaling);
    valImagesOriginal{i} = img;
end

% Load Testing Images
testImagesOriginal = cell(1,numTestImages);
for i=1:numTestImages
    img = rgb2gray(imread(['data/test/TestImages/Image' num2str(i) '.jpg']));
    img = imresize(img,rescaling);
    testImagesOriginal{i} = img;
end

%%
[f1, d1] = vl_sift(single(trainImagesOriginal{1})) ;
% Display detected features
imshow(trainImagesOriginal{1})
title('Features on 1st image')
hold on
plot(f1(1,:),f1(2,:),'r.','MarkerSize',3);