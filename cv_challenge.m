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

load('data/train/trainLbls.mat')
load('data/train/trainVectors.mat')
load('data/validation/valLbls.mat')
load('data/validation/valVectors.mat')
load('data/test/testVectors.mat')

% [trainVectorsOriginal, valVectorsOriginal, testVectorsOriginal] = loadOriginalImages()

[~,N] = size(trainVectors);
[~,M] = size(valVectors);
[D,L] = size(testVectors);
classes = unique(trainLbls);
K = length(classes);

trainVectorsStd = (trainVectors - repmat(mean(trainVectors,2),1,N))./repmat(var(trainVectors),D,1);
valVectorsStd = (valVectors - repmat(mean(valVectors,2),1,M))./repmat(var(valVectors),D,1);
testVectorsStd = (testVectors - repmat(mean(testVectors,2),1,L))./repmat(var(testVectors),D,1);

%% Nearest Centroid 

score_nc = ncClassifier(trainVectors, trainLbls,valVectors, valLbls, testVectors);
score_nc_std = ncClassifier(trainVectorsStd, trainLbls,valVectorsStd, valLbls, testVectorsStd);

%% Nearest Subclass Centroid 
score_nsc = nscClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors);
score_nsc_std = nscClassifier(trainVectorsStd, trainLbls,valVectorsStd, valLbls, testVectorsStd);

%% Nearest Neighbor 

score_nn = nnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors);
score_nn_std = nnClassifier(trainVectorsStd, trainLbls,valVectorsStd, valLbls, testVectorsStd);

%% Linear Regression 

score_linreg = linRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,K,N);
score_linreg_std = linRegression(trainVectorsStd,trainLbls,valVectorsStd,valLbls,testVectorsStd,K,N);

%% Kernel-based Regression

score_rbfreg = rbfRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,K,N);
score_rbfreg_std = rbfRegression(trainVectorsStd,trainLbls,valVectorsStd,valLbls,testVectorsStd,K,N);

%% SVM

score_svm = svm(trainVectors,trainLbls,valVectors,valLbls,testVectors);


%% MLP
% Label Matrix
T = zeros(K,N);
for n=1:K
    T(n,trainLbls==n) = 1;
end

net =feedforwardnet(100,'traingdx');

[net,tr] = train(net,trainVectors,T);

predLbls = vec2ind(net(valVectors));
score_mlp = length(find(predLbls-valLbls'==0)) / length(valLbls);





