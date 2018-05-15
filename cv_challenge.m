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
disp('[*] Load Data...')
load('data/train/trainLbls.mat')
load('data/train/trainVectors.mat')
load('data/validation/valLbls.mat')
load('data/validation/valVectors.mat')
load('data/test/testVectors.mat')
N = length(trainLbls);
M = length(valLbls);
[D,L] = size(testVectors);
classes = unique(trainLbls);
number_of_classes = length(classes);


% Load SIFT Features
disp('[*] Load SIFT Features...')
load('data/train/trainVectorsSIFT.mat')
load('data/validation/valVectorsSIFT.mat')
load('data/test/testVectorsSIFT.mat')
load('data/sift_feature_idx.mat')
load('data/sift_feature_means.mat')

% Load HOG Features
disp('[*] Load HOG Features...')
load('data/train/trainVectorsHOG.mat')
load('data/validation/valVectorsHOG.mat')
load('data/test/testVectorsHOG.mat')

number_of_sift_classes = 200;


%% Preprocess data
% number_of_sift_classes = 200;
% if data_type == "sift_features"
%     % Pre-Process SIFT Data
%     sift_features = double(cell2mat(trainVectorsSIFT));
%     [feature_idx, feature_means, feature_sumd, feature_distance] = kmeans(sift_features',number_of_sift_classes);
% end
% save('sift_feature_idx.mat','feature_idx')
% save('sift_feature_means.mat','feature_means')
% save('sift_feature_sumd.mat','feature_sumd')
% save('sift_feature_distance.mat','feature_distance')



%% Construct bag of words
disp('[*] Construct bag of words...')
n = 1;
j = 0;
for i = 1:length(trainVectorsSIFT)
    s =  size(trainVectorsSIFT{i},2);
    j = j + s;
    for k = 1:number_of_sift_classes
        K(k,i) = length(find(feature_idx(n:j)==k));
    end
    n = n + s;
end

% Get Histograms of Validation set

for i=1:size(valVectorsSIFT,2)
    current = double(valVectorsSIFT{i});
    for k =1:number_of_sift_classes
        m = feature_means(k,:);
        M = repmat(m',1,size(current,2));
        d(k,:) = vecnorm(current - M,2).^2;
    end
    [~,pred_val_lbls] = min(d);
    
    for k = 1:number_of_sift_classes
        K_val(k,i) = length(find(pred_val_lbls==k));
    end
    clear pred_val_lbls d
end

% Get Histograms for Testing set
for i=1:size(testVectorsSIFT,2)
    current = double(testVectorsSIFT{i});
    for k =1:number_of_sift_classes
        m = feature_means(k,:);
        M = repmat(m',1,size(current,2));
        d(k,:) = vecnorm(current - M,2).^2;
    end
    [~,pred_test_lbls] = min(d);
    
    for k = 1:number_of_sift_classes
        K_test(k,i) = length(find(pred_test_lbls==k));
    end
    clear pred_val_lbls d
end

%%
K_norm = (K - repmat(min(K),number_of_sift_classes,1))./repmat(max(K)-min(K),number_of_sift_classes,1);
K_val_norm = (K_val - repmat(min(K_val),number_of_sift_classes,1))./repmat(max(K_val)-min(K_val),number_of_sift_classes,1);
K_test_norm = (K_test - repmat(min(K_test),number_of_sift_classes,1))./repmat(max(K_test)-min(K_test),number_of_sift_classes,1);


K_std = (K -  repmat(mean(K),number_of_sift_classes,1))./repmat(var(K),number_of_sift_classes,1);
K_val_std = (K_val -  repmat(mean(K_val),number_of_sift_classes,1))./repmat(var(K_val),number_of_sift_classes,1);
K_test_std = (K_test -  repmat(mean(K_test),number_of_sift_classes,1))./repmat(var(K_test),number_of_sift_classes,1);

%% Nearest Centroid

score_nc = ncClassifier(trainVectors, trainLbls,valVectors, valLbls, testVectors);
score_nc_sift = ncClassifier(K_std, trainLbls,K_val_std, valLbls, K_test_std);
score_nc_hog = ncClassifier(trainVectorsHOG', trainLbls,valVectorsHOG', valLbls, testVectorsHOG');

%% Nearest Subclass Centroid
score_nsc = nscClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors);
score_nsc_sift = nscClassifier(K_std,trainLbls,K_val_std,valLbls, K_test_std);
score_nsc_hog = nscClassifier(trainVectorsHOG',trainLbls,valVectorsHOG',valLbls,testVectorsHOG');

%% Nearest Neighbor

score_knn3 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,3);
score_knn_sift3 = knnClassifier(K, trainLbls,K_val, valLbls, K_test,3);
score_knn_hog3 = knnClassifier(trainVectorsHOG',trainLbls,valVectorsHOG',valLbls,testVectorsHOG',3);

score_knn4 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,4);
score_knn_sift4 = knnClassifier(K, trainLbls,K_val, valLbls, K_test,4);
score_knn_hog4 = knnClassifier(trainVectorsHOG',trainLbls,valVectorsHOG',valLbls,testVectorsHOG',4);

score_knn5 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,5);
score_knn_sift5= knnClassifier(K, trainLbls,K_val, valLbls, K_test,5);
score_knn_hog5 = knnClassifier(trainVectorsHOG',trainLbls,valVectorsHOG',valLbls,testVectorsHOG',5);

score_knn6 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,6);
score_knn_sift6= knnClassifier(K, trainLbls,K_val, valLbls, K_test,6);
score_knn_hog6 = knnClassifier(trainVectorsHOG',trainLbls,valVectorsHOG',valLbls,testVectorsHOG',6);

score_knn7 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,7);
score_knn_sift7= knnClassifier(K, trainLbls,K_val, valLbls, K_test,7);
score_knn_hog7 = knnClassifier(trainVectorsHOG',trainLbls,valVectorsHOG',valLbls,testVectorsHOG',7);

score_knn8 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,8);
score_knn_sift8 = knnClassifier(K, trainLbls,K_val, valLbls, K_test,8);
score_knn_hog8 = knnClassifier(trainVectorsHOG',trainLbls,valVectorsHOG',valLbls,testVectorsHOG',8);

score_knn_sift12 = knnClassifier(K, trainLbls,K_val, valLbls, K_test,12);
score_knn_hog10 = knnClassifier(trainVectorsHOG',trainLbls,valVectorsHOG',valLbls,testVectorsHOG',10);

%% Linear Regression

score_linreg = linRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,number_of_classes,N);
score_linreg_sift = linRegression(K_norm, trainLbls,K_val_norm, valLbls, K_test_norm,number_of_classes,N);
score_linreg_hog = linRegression(trainVectorsHOG',trainLbls,valVectorsHOG',valLbls,testVectorsHOG',number_of_classes,N);

%% Kernel-based Regression
Cvec = 0.18:0.01:0.2;
Svec = 0.5:0.1:0.6;

score_rbfreg = rbfRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,D,N,Cvec,Svec);
score_rbfreg_sift = rbfRegression(K, trainLbls,K_val, valLbls, K_test,number_of_classes,N,Cvec,Svec);

Cvec = 0.001:0.001:0.005;
Svec = 0.3:0.02:0.42;
score_rbfreg_hog = rbfRegression(trainVectorsHOG',trainLbls,valVectorsHOG',valLbls,testVectorsHOG',number_of_classes,N,Cvec,Svec);

%% SVM

score_svm = svm(trainVectors,trainLbls,valVectors,valLbls,testVectors);
score_svm_sift = svm(K_norm, trainLbls,K_val_norm, valLbls, K_test_norm);
%score_svm_hog = svm(trainVectorsHOG',trainLbls,valVectorsHOG',valLbls,testVectorsHOG');

%% MLP
% Label Matrix
disp('***********************************')
disp('Multilayer Perceptron')
T = zeros(number_of_classes,N);
for n=1:number_of_classes
    T(n,trainLbls==n) = 1;
end

%%
disp('[*] CNN features')
net =feedforwardnet([100 100] ,'trainscg'); % 50%
net =feedforwardnet([200 100] ,'trainscg'); % 46%
net =feedforwardnet([200 100 100] ,'trainscg'); %49
net =feedforwardnet([200 100 50] ,'trainscg'); % 53
net =feedforwardnet([200 200 50] ,'trainscg'); % 58


net.trainParam.max_fail= 100;

[net,tr] = train(net,trainVectors,T,'useParallel','yes');

predLbls = vec2ind(net(valVectors));
score_mlp = length(find(predLbls-valLbls'==0)) / length(valLbls);

disp(['[*] Max Score: ', num2str(score_mlp*100), '%'])

%%
disp('[*] SIFT features')
net =feedforwardnet([100 100 80] ,'trainscg'); 
net.trainParam.max_fail= 100;
[net,tr] = train(net,K_std,T,'useParallel','yes');
predLbls = vec2ind(net(K_val_std));
score_mlp_sift = length(find(predLbls-valLbls'==0)) / length(valLbls);
disp(['[*] Max Score: ', num2str(score_mlp_sift*100), '%'])

%%
disp('[*] HOG features')
net =feedforwardnet([500 300 80] ,'trainscg'); 
net.trainParam.max_fail= 100;
[net,tr] = train(net,trainVectorsHOG',T,'useParallel','yes');
predLbls = vec2ind(net(valVectorsHOG'));
score_mlp_sift = length(find(predLbls-valLbls'==0)) / length(valLbls);
disp(['[*] Max Score: ', num2str(score_mlp_sift*100), '%'])
%% CNN
% cnn_evaluation

%% Plots
generate_plots



