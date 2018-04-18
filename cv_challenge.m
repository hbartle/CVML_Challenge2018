%
% Kaggle Challenge Submission
% Computer Vision and Machine Learning
% Aarhus University, 2018
% Hannes Bartle
%
clear 
close all
clc

addpath('functions/')

%% Load data

data_type = "sift_features";

if data_type == "cnn_features"

    load('data/train/trainLbls.mat')
    load('data/train/trainVectors.mat')
    load('data/validation/valLbls.mat')
    load('data/validation/valVectors.mat')
    load('data/test/testVectors.mat')
    [~,N] = size(trainVectors);
    [~,M] = size(valVectors);
    [D,L] = size(testVectors);
    classes = unique(trainLbls);
    K = length(classes);
    
    % Standardize Data
    % data = [trainVectors,valVectors,testVectors];
    % dataStd = (data - repmat(mean(data,2),1,N+M+L));%./repmat(var(data),D,1);
    % trainVectorsStd = dataStd(:,1:N);
    % valVectorsStd = dataStd(:,N+1:N+M);
    % testVectorsStd = dataStd(:,N+M+1:N+M+L);

elseif data_type == "original"
    [trainVectorsOriginal, valVectorsOriginal, testVectorsOriginal] = loadOriginalImages();


elseif data_type == "sift_features"
    
    % Load SIFT Features
    load('data/train/trainVectorsSIFT.mat')
    load('data/validation/valVectorsSIFT.mat')
    load('data/test/testVectorsSIFT.mat')
end

%% Preprocess data

if data_type == "sift_features"
    % Pre-Process SIFT Data
    sift_features = double(cell2mat(trainVectorsSIFT));
    %feature_means = kmeans(sift_features',200);
end



%% Nearest Centroid 

% score_nc = ncClassifier(trainVectors, trainLbls,valVectors, valLbls, testVectors);
% score_nc_std = ncClassifier(trainVectorsStd, trainLbls,valVectorsStd, valLbls, testVectorsStd);

%% Nearest Subclass Centroid 
% score_nsc = nscClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors);
% score_nsc_std = nscClassifier(trainVectorsStd, trainLbls,valVectorsStd, valLbls, testVectorsStd);

%% Nearest Neighbor 

score_nn = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,4);
% score_nn_std = knnClassifier(trainVectorsStd, trainLbls,valVectorsStd, valLbls, testVectorsStd);

%% Linear Regression 

% score_linreg = linRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,K,N);
% score_linreg_std = linRegression(trainVectorsStd,trainLbls,valVectorsStd,valLbls,testVectorsStd,K,N);

%% Kernel-based Regression

% score_rbfreg = rbfRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,K,N);
% score_rbfreg_std = rbfRegression(trainVectorsStd,trainLbls,valVectorsStd,valLbls,testVectorsStd,K,N);

%% SVM

% score_svm = svm(trainVectors,trainLbls,valVectors,valLbls,testVectors);


%% MLP
% Label Matrix
% T = zeros(K,N);
% for n=1:K
%     T(n,trainLbls==n) = 1;
% end
% 
% net =feedforwardnet(100,'traingdx');
% 
% [net,tr] = train(net,trainVectors,trainLbls');
% 
% predLbls = vec2ind(net(valVectors));
% score_mlp = length(find(predLbls-valLbls'==0)) / length(valLbls);


%% CNN
% imageSize = [64 64 1];
% 
% layers = [
%     imageInputLayer(imageSize,'Name','input')
%     
% 
%     fullyConnectedLayer(29)
%     softmaxLayer
%     classificationLayer];
% 
% 
% options = trainingOptions('sgdm', ...
%     'MaxEpochs',20,...
%     'InitialLearnRate',1e-3, ...
%     'Shuffle','every-epoch', ...
%     'Verbose',0, ...
%     'Plots','training-progress',...
%     'ValidationData',imdsVal,...
%     'ValidationPatience',Inf);
% 
% imdsTrain = imageDatastore('data/train/cnnfeatures/');
% imdsTrain.Labels = categorical(trainLbls);
% 
% imdsVal = imageDatastore('data/validation/cnnfeatures/');
% imdsVal.Labels = categorical(valLbls);
% 
% imdsTest = imageDatastore('data/test/cnnfeatures/');
% 
% imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-30,30], ...
%     'RandXTranslation',[-5 5], ...
%     'RandYTranslation',[-5 5],...
%     'RandXScale',[0.5 3],...
%     'RandYScale',[0.5 3],...
%     'RandXReflection',true, ...
%     'RandYReflection',true);
% 
% augimdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);
% 
% 
% net = trainNetwork(imdsTrain,layers,options);
% 
% % pred_val_lbls = classify(net,imdsVal);
% % score_cnn = sum(pred_val_lbls == imdsVal.Labels)/numel(imdsVal.Labels);
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
