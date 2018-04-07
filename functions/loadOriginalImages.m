function [trainVectorsOriginal, valVectorsOriginal, testVectorsOriginal] = loadOriginalImages()
% Load Original Images, convert them to grayscale and vectorize them
numTrainImages = 5830;
numValImages = 2298;
numTestImages = 3460;

rescaling = 0.3;
resolution = ceil(rescaling*256);


% Load Training Images
trainVectorsOriginal = nan(resolution^2,numTrainImages);
for i=1:numTrainImages
    img = rgb2gray(imread(['data/train/TrainImages/Image' num2str(i) '.jpg']));
    img = imresize(img,rescaling);
    trainVectorsOriginal(:,i) = img(:);
end

% Load Validation Images
valVectorsOriginal = nan(resolution^2,numValImages);
for i=1:numValImages
    img = rgb2gray(imread(['data/validation/ValidationImages/Image' num2str(i) '.jpg']));
    img = imresize(img,rescaling);
    valVectorsOriginal(:,i) = img(:);
end

% Load Testing Images
testVectorsOriginal = nan(resolution^2,numTestImages);
for i=1:numTestImages
    img = rgb2gray(imread(['data/test/TestImages/Image' num2str(i) '.jpg']));
    img = imresize(img,rescaling);
    testVectorsOriginal(:,i) = img(:);
end

end