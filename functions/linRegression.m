function [score] = linRegression(trainVectors,trainLbls,valVectors,valLbls,testVectors,K,N)

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
    [~,pred_val_Lbls] = max(OutputVal);
    
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
file = fopen('submissions/testLbls_linreg.txt','w');
fprintf(file,'%s,%s\n','ID','Label');
fprintf(file,'%d,%d\n',[index; pred_test_lbls{ind}]);
fclose(file);
disp('[*] Done!')



end

