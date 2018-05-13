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
[~,N] = size(trainVectors);
[~,M] = size(valVectors);
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
% end

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


%% Nearest Centroid

score_nc = ncClassifier(trainVectors, trainLbls,valVectors, valLbls, testVectors);
score_nc_sift = ncClassifier(K, trainLbls,K_val, valLbls, K_test);

%% Nearest Subclass Centroid
score_nsc = nscClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors);
score_nsc_sift = nscClassifier(K,trainLbls,K_val,valLbls, K_test);

%% Nearest Neighbor

score_knn3 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,3);
score_knn_sift3 = knnClassifier(K, trainLbls,K_val, valLbls, K_test,3);
score_knn4 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,4);
score_knn_sift4 = knnClassifier(K, trainLbls,K_val, valLbls, K_test,4);
score_knn5 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,5);
score_knn_sift5= knnClassifier(K, trainLbls,K_val, valLbls, K_test,5);
score_knn6 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,6);
score_knn_sift6= knnClassifier(K, trainLbls,K_val, valLbls, K_test,6);
score_knn7 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,7);
score_knn_sift7= knnClassifier(K, trainLbls,K_val, valLbls, K_test,7);
score_knn8 = knnClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors,8);
score_knn_sift8 = knnClassifier(K, trainLbls,K_val, valLbls, K_test,8);

score_knn_sift12 = knnClassifier(K, trainLbls,K_val, valLbls, K_test,12);

%% Linear Regression

score_linreg = linRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,number_of_classes,N);
score_linreg_sift = linRegression(K, trainLbls,K_val, valLbls, K_test,number_of_classes,N);

%% Kernel-based Regression

score_rbfreg = rbfRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,D,N);
score_rbfreg_sift = rbfRegression(K, trainLbls,K_val, valLbls, K_test,number_of_classes,N);

%% SVM

score_svm = svm(trainVectors,trainLbls,valVectors,valLbls,testVectors);
score_svm_sift = svm(K, trainLbls,K_val, valLbls, K_test);

%% MLP
% Label Matrix
disp('***********************************')
disp('Multilayer Perceptron')
T = zeros(number_of_classes,N);
for n=1:number_of_classes
    T(n,trainLbls==n) = 1;
end

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

disp('[*] SIFT features')

%%
net =feedforwardnet([100 100 80] ,'trainscg'); 
net.trainParam.max_fail= 100;
[net,tr] = train(net,K,T,'useParallel','yes');
predLbls = vec2ind(net(K_val));
score_mlp_sift = length(find(predLbls-valLbls'==0)) / length(valLbls);
disp(['[*] Max Score: ', num2str(score_mlp_sift*100), '%'])

%% CNN
% cnn_evaluation

%% Plots
generate_plots



