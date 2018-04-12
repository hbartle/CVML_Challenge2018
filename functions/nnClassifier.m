function [ score_nn ] = nnClassifier(trainVectors,trainLbls,valVectors, valLbls,testVectors)
%NN_CLASSIFY Nearest Neighborhood Classifier

disp('***********************************')
disp('Nearest Centroid Classification')
disp('[*] Validation Set...')
% Classify the test images
[~,N] = size(valVectors);
[~,M] = size(trainVectors);
[~,L] = size(testVectors);
for i=1:N
    for k =1:M
        d(k) = norm(valVectors(:,i) - trainVectors(:,k))^2;
    end
    [~,pred_val_lbls(i)] = min(d);
    pred_val_lbls(i) = trainLbls(pred_val_lbls(i));
end
score_nn = length(find(pred_val_lbls-valLbls'==0))/length(valLbls);
disp(['[*] Max Score: ',num2str(score_nn*100),'%'])
disp('[*] Testing Set...')

for i=1:L
    for k =1:M
        d(k) = norm(testVectors(:,i) - trainVectors(:,k))^2;
    end
    [~,idx] = min(d);
    pred_test_lbls(i) = trainLbls(idx);
end

index = 1:size(testVectors,2);
file = fopen('submissions/testLbls_nn.txt','w');
fprintf(file,'%s,%s\n','ID','Label');
fprintf(file,'%d,%d\n',[index; pred_test_lbls]);
fclose(file);
disp('[*] Done!')

end


