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
classes = unique(trainLbls);
K = length(classes);

%% Nearest Centroid 
disp('***********************************')
disp('Nearest Centroid Classification')
[predLbls, mean_vectors] = ncClassifier(trainVectors,valVectors,trainLbls);
nc_score = length(find(predLbls-valLbls'==0))/length(valLbls);
disp(['[*] Max Score: ',num2str(nc_score*100),'%'])

%% Nearest Subclass Centroid 
disp('***********************************')
disp('Nearest Subclass Centroid Classification')
disp('[*] Optimizing over hyperparameter space...')
K_sc = 5;
for k=1:K_sc
    predLbls = nscClassifier(trainVectors,valVectors,trainLbls,k);
    nsc_sc(k) = length(find(predLbls-valLbls'==0))/length(valLbls);
    disp(['[*]' num2str(k) ' of ' num2str(K_sc)])
end
[nsc_score, Kmax] = max(nsc_sc);
disp(['[*] K: ',num2str(Kmax)])
disp(['[*] Max Score: ',num2str(nsc_score*100),'%'])

%% Nearest Neighbor 
disp('***********************************')
disp('Nearest Centroid Classification')
predLbls = nnClassifier(trainVectors,valVectors,trainLbls);
nn_score = length(find(predLbls-valLbls'==0))/length(valLbls);
disp(['[*] Max Score: ',num2str(nn_score*100),'%'])


%% Linear Regression 

% Label Matrix
T = zeros(K,N);
for n=1:K
    T(n,trainLbls==n) = 1;
end


% Regularization Parameter Set
Cvec = 10.^(-4:3);


% Initialize Matrices
pred_test_lbls = cell(length(Cvec));
LMS_CR = zeros(length(Cvec));

% Precompute Training Vector Covariance
X=trainVectors*trainVectors';
disp('***********************************')
disp('LMS Regression-based Classification')
disp('[*] Optimizing over hyperparameter space...')
for cc=1:length(Cvec)
    
    C = Cvec(cc);

    % Get the Weights
    W = (X+ C*eye(D))\trainVectors * T';
    
    
    % Classify Validation Samples
    OutputVal = W'*valVectors;
    [maxOt,pred_val_Lbls] = max(OutputVal);
    
    % Classify Testing Set
    OutputTest = W'*testVectors;
    [~,pred_test_lbls{cc}] = max(OutputTest);
    
    
    % Measure Performance
    LMS_CR(cc) = length(find(pred_val_Lbls-valLbls'==0)) / length(valLbls);
    disp(['[*] ' num2str(cc) ' of ' num2str(length(Cvec))])
end
[score,ind] = max(LMS_CR(:));
disp(['C-Value: ',num2str(Cvec(ind))])
disp(['Max Score: ',num2str(score*100),'%'])

index = 1:size(testVectors,2);
file = fopen('submissions/testLbls_LMS.txt','w');
fprintf(file,'%s,%s\n','ID','Label');
fprintf(file,'%d,%d\n',[index; pred_test_lbls{ind}]);
fclose(file);

%% Kernel-based Regression

% Label Matrix
T = zeros(K,N);
for n=1:K
    T(n,trainLbls==n) = 1;
end

% Hyperparameter  Set
Cvec = 0.18:0.01:0.2;
Svec = 0.5:0.1:0.6;



disp('***********************************')
disp('RBF-Kernel-Regression-based Classification')

% Precompute Training Set Covariance
G_train = distance_matrix(trainVectors,trainVectors);
sigma_train = mean(mean(G_train));
% Precompute Validation Set Covariance
Gval = distance_matrix(valVectors,trainVectors);
sigma_val = mean(mean(Gval));
% Precompute Testing Set Covariance
Gtest = distance_matrix(testVectors,trainVectors);
sigma_test = mean(mean(Gtest));



pred_test_lbls = cell(length(Svec),length(Cvec));
KR_CR = zeros(length(Svec),length(Cvec));
disp('[*] Optimizing over hyperparameter space...')
for ss=1:length(Svec)
    for cc=1:length(Cvec)
        
        S = Svec(ss);
        C = Cvec(cc);
        
        % Kernel Matrix of Training Set
        Ktrain = exp(-G_train/(S*sigma_train));
        % Get Weight Matrix
        A = (Ktrain + C*eye(N))\ T';
        
        % Kernel Matrix of Validation Set
        Kval = exp(-Gval/(S*sigma_val));
        % Classify Validation Set
        OutputVal = A'*Kval;
        [~,pred_val_lbls] = max(OutputVal);
        
        
        % Kernel Matrix of Testing Set
        Ktest = exp(-Gtest/(S*sigma_test));
        % Classify Validation Set
        OutputTest = A'*Ktest;
        [~,pred_test_lbls{ss,cc}] = max(OutputTest);
        
        
        % Measure Performance
        KR_CR(ss,cc) = length(find(pred_val_lbls-valLbls'==0)) / length(valLbls);
        disp(['S: ' num2str(ss) ' of ' num2str(length(Svec));
              'C: ' num2str(cc) ' of ' num2str(length(Cvec))])
    end
end
[score,ind] = max(KR_CR(:));
[Sidx,Cidx] = ind2sub(size(KR_CR),ind);
disp(['C-Value: ',num2str(Cvec(Cidx)),', S-Value: ',num2str(Svec(Sidx))])
disp(['Max Score: ',num2str(score*100),'%'])

index = 1:size(testVectors,2);
file = fopen('submissions/testLbls_KR.txt','w');
fprintf(file,'%s,%s\n','ID','Label');
fprintf(file,'%d,%d\n',[index; pred_test_lbls{Sidx,Cidx}]);
fclose(file);


%% 


