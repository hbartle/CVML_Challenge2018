function [score] = rbfRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,K,N)
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
        % Classify Testing Set
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
file = fopen('submissions/testLbls_rbfreg.txt','w');
fprintf(file,'%s,%s\n','ID','Label');
fprintf(file,'%d,%d\n',[index; pred_test_lbls{Sidx,Cidx}]);
fclose(file);
disp('[*] Done!')

end

