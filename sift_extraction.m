%
%   SIFT Feature Extraction
%
%
%
close all
clear 
clc



VLFEATROOT = 'lib/vlfeat-0.9.21';
addpath([VLFEATROOT,'/toolbox'])
vl_setup()

%% Get the original Images
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
        img = rgb2gray(imread(['data/train/TrainImages/Image' num2str(i) '.jpg']));
        img = imresize(img,rescaling);
        trainImagesOriginal(:,:,i) = img;
    end
    save('data/train/TrainImages/trainimages.mat','trainImagesOriginal');
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
        img = imresize(img,rescaling);
        valImagesOriginal(:,:,i) = img;
    end
    save('data/validation/ValidationImages/valimages.mat','valImagesOriginal');
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
        img = imresize(img,rescaling);
        testImagesOriginal(:,:,i) = img;
    end
    save('data/test/TestImages/testimages.mat','testImagesOriginal');
end
disp('Done!')

%% SIFT Feature extraction
% for i=1:numTrainImages
%     [f, d] = vl_sift(single(trainImagesOriginal{i}));
%     trainVectorsSIFT{i} = d;
% end
