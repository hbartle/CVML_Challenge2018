%
% Kaggle Challenge Submission
% Computer Vision and Machine Learning
% Aarhus University, 2018
% Hannes Bartle
%
% clear 
% close all
% clc
% 
% addpath('functions/')
% 
% %% Load data
% 
% load('data/train/trainLbls.mat')
% load('data/train/trainVectors.mat')
% load('data/validation/valLbls.mat')
% load('data/validation/valVectors.mat')
% load('data/test/testVectors.mat')
% 
% % [trainVectorsOriginal, valVectorsOriginal, testVectorsOriginal] = loadOriginalImages()
% 
% [~,N] = size(trainVectors);
% [~,M] = size(valVectors);
% [D,L] = size(testVectors);
% classes = unique(trainLbls);
% K = length(classes);
% 
% data = [trainVectors,valVectors,testVectors];
% dataStd = (data - repmat(mean(data,2),1,N+M+L));%./repmat(var(data),D,1);
% trainVectorsStd = dataStd(:,1:N);
% valVectorsStd = dataStd(:,N+1:N+M);
% testVectorsStd = dataStd(:,N+M+1:N+M+L);

%% Nearest Centroid 

% score_nc = ncClassifier(trainVectors, trainLbls,valVectors, valLbls, testVectors);
% score_nc_std = ncClassifier(trainVectorsStd, trainLbls,valVectorsStd, valLbls, testVectorsStd);

%% Nearest Subclass Centroid 
% score_nsc = nscClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors);
% score_nsc_std = nscClassifier(trainVectorsStd, trainLbls,valVectorsStd, valLbls, testVectorsStd);

%% Nearest Neighbor 

% score_nn = nnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors);
% score_nn_std = nnClassifier(trainVectorsStd, trainLbls,valVectorsStd, valLbls, testVectorsStd);

%% Linear Regression 

% score_linreg = linRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,K,N);
% score_linreg_std = linRegression(trainVectorsStd,trainLbls,valVectorsStd,valLbls,testVectorsStd,K,N);

%% Kernel-based Regression

% score_rbfreg = rbfRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,K,N);
% score_rbfreg_std = rbfRegression(trainVectorsStd,trainLbls,valVectorsStd,valLbls,testVectorsStd,K,N);

%% SVM

% score_svm = svm(trainVectors,trainLbls,valVectors,valLbls,testVectors);


%% MLP
%Label Matrix
% T = zeros(K,N);
% for n=1:K
%     T(n,trainLbls==n) = 1;
% end
% 
% net =feedforwardnet(100,'traingdx');
% 
% [net,tr] = train(net,trainVectors,T);
% 
% predLbls = vec2ind(net(valVectors));
% score_mlp = length(find(predLbls-valLbls'==0)) / length(valLbls);


%% CNN
imageSize = [64 64 1];

layers = [
    imageInputLayer(imageSize,'Name','input')
    

    fullyConnectedLayer(29)
    softmaxLayer
    classificationLayer];


options = trainingOptions('sgdm', ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'Verbose',0, ...
    'Plots','training-progress',...
    'ValidationData',imdsVal,...
    'ValidationPatience',Inf);

imdsTrain = imageDatastore('data/train/cnnfeatures/');
imdsTrain.Labels = categorical(trainLbls);

imdsVal = imageDatastore('data/validation/cnnfeatures/');
imdsVal.Labels = categorical(valLbls);

imdsTest = imageDatastore('data/test/cnnfeatures/');

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-30,30], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5],...
    'RandXScale',[0.5 3],...
    'RandYScale',[0.5 3],...
    'RandXReflection',true, ...
    'RandYReflection',true);

augimdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);


net = trainNetwork(imdsTrain,layers,options);

% pred_val_lbls = classify(net,imdsVal);
% score_cnn = sum(pred_val_lbls == imdsVal.Labels)/numel(imdsVal.Labels);















