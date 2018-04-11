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

[D,N] = size(trainVectors);
[D,M] = size(valVectors);
[D,L] = size(testVectors);
classes = unique(trainLbls);
K = length(classes);

trainVectorsStd = (trainVectors - repmat(mean(trainVectors,2),1,N))./repmat(var(trainVectors),D,1);
valVectorsStd = (valVectors - repmat(mean(valVectors,2),1,M))./repmat(var(valVectors),D,1);
testVectorsStd = (testVectors - repmat(mean(testVectors,2),1,L))./repmat(var(testVectors),D,1);

% %% Nearest Centroid 
% disp('***********************************')
% disp('Nearest Centroid Classification')
% [predLbls, mean_vectors] = ncClassifier(trainVectors,valVectors,trainLbls);
% nc_score = length(find(predLbls-valLbls'==0))/length(valLbls);
% disp(['[*] Max Score: ',num2str(nc_score*100),'%'])
% 
% %% Nearest Subclass Centroid 
% disp('***********************************')
% disp('Nearest Subclass Centroid Classification')
% disp('[*] Optimizing over hyperparameter space...')
% K_sc = 5;
% for k=1:K_sc
%     predLbls = nscClassifier(trainVectors,valVectors,trainLbls,k);
%     nsc_sc(k) = length(find(predLbls-valLbls'==0))/length(valLbls);
%     disp(['[*]' num2str(k) ' of ' num2str(K_sc)])
% end
% [nsc_score, Kmax] = max(nsc_sc);
% disp(['[*] K: ',num2str(Kmax)])
% disp(['[*] Max Score: ',num2str(nsc_score*100),'%'])
% 
% %% Nearest Neighbor 
% disp('***********************************')
% disp('Nearest Centroid Classification')
% predLbls = nnClassifier(trainVectors,valVectors,trainLbls);
% nn_score = length(find(predLbls-valLbls'==0))/length(valLbls);
% disp(['[*] Max Score: ',num2str(nn_score*100),'%'])
% 
% 
% %% Linear Regression 
% 
% % Label Matrix
% T = zeros(K,N);
% for n=1:K
%     T(n,trainLbls==n) = 1;
% end
% 
% 
% % Regularization Parameter Set
% Cvec = 10.^(-4:3);
% 
% 
% % Initialize Matrices
% pred_test_lbls = cell(length(Cvec));
% LMS_CR = zeros(length(Cvec));
% 
% % Precompute Training Vector Covariance
% X=trainVectors*trainVectors';
% disp('***********************************')
% disp('LMS Regression-based Classification')
% disp('[*] Optimizing over hyperparameter space...')
% for cc=1:length(Cvec)
%     
%     C = Cvec(cc);
% 
%     % Get the Weights
%     W = (X+ C*eye(D))\trainVectors * T';
%     
%     
%     % Classify Validation Samples
%     OutputVal = W'*valVectors;
%     [maxOt,pred_val_Lbls] = max(OutputVal);
%     
%     % Classify Testing Set
%     OutputTest = W'*testVectors;
%     [~,pred_test_lbls{cc}] = max(OutputTest);
%     
%     
%     % Measure Performance
%     LMS_CR(cc) = length(find(pred_val_Lbls-valLbls'==0)) / length(valLbls);
%     disp(['[*] ' num2str(cc) ' of ' num2str(length(Cvec))])
% end
% [score,ind] = max(LMS_CR(:));
% disp(['C-Value: ',num2str(Cvec(ind))])
% disp(['Max Score: ',num2str(score*100),'%'])
% 
% index = 1:size(testVectors,2);
% file = fopen('submissions/testLbls_LMS.txt','w');
% fprintf(file,'%s,%s\n','ID','Label');
% fprintf(file,'%d,%d\n',[index; pred_test_lbls{ind}]);
% fclose(file);

%% Kernel-based Regression

% rbfRegression([trainVectors valVectors(:,1:1500)],[trainLbls; valLbls(1:1500)],valVectors,valLbls,testVectors,K,N+15qqßq00)

%% MLP
% % Label Matrix
% T = zeros(K,N);
% for n=1:K
%     T(n,trainLbls==n) = 1;
% end
% 
% net =feedforwardnet([100 100],'trainscg');
% %net.layers{3}.transferFcn = 'softmax';
% 
% [net,tr] = train(net,trainVectors,T);
% 
% predLbls = vec2ind(net(valVectors));
% score = length(find(predLbls-valLbls'==0)) / length(valLbls);

%% Transferred-Learning with AlexNet
% 
% % Get Image Datastores
% imdsTrain = imageDatastore('data/train/TrainImages');
% imdsTrain.Labels = categorical(trainLbls);
% 
% imdsVal = imageDatastore('data/validation/ValidationImages');
% imdsVal.Labels = categorical(valLbls);
% 
% imdsTest = imageDatastore('data/test/TestImages');
% 
% 
% 
% net = alexnet;
% layersTransfer = net.Layers(1:end-8);
% layers = [  layersTransfer
%             fullyConnectedLayer(4096,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%             reluLayer
%             dropoutLayer
%             fullyConnectedLayer(4096,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%             reluLayer
%             dropoutLayer
%             fullyConnectedLayer(K,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%             softmaxLayer
%             classificationLayer];
%         
% inputSize = net.Layers(1).InputSize;
% 
% 
% pixelRange = [-21 21];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
%     'DataAugmentation',imageAugmenter);
% 
% 
% augimdsVal = augmentedImageDatastore(inputSize(1:2),imdsVal);
% 
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',50, ...
%     'MaxEpochs',6, ...
%     'InitialLearnRate',1e-2, ...
%     'ValidationData',augimdsVal, ...
%     'ValidationFrequency',3, ...
%     'ValidationPatience',Inf, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% 
% netTransfer = trainNetwork(augimdsTrain,layers,options);
% 





