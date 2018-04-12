function [ score_nc ] = ncClassifier(trainVectors, trainLbls,valVectors, valLbls, testVectors)
disp('***********************************')
disp('Nearest Centroid Classification')

number_of_classes= 29;

% Calculate Class Mean
for i=1:number_of_classes
    idx = find(trainLbls == i);
    m(:,i)= mean(trainVectors(:,idx),2);
end

% Classify the test images
[~,number_of_samples] = size(valVectors);
for i=1:number_of_samples
    for k =1:number_of_classes
        d(k) = norm(valVectors(:,i) -m(:,k))^2;
    end
    [~,pred_val_lbls(i)] = min(d) ; 
end

score_nc = length(find(pred_val_lbls-valLbls'==0))/length(valLbls);
disp(['[*] Max Score: ',num2str(score_nc*100),'%'])

disp('[*] Classify test images...')
% Classify the test images
[~,number_of_samples] = size(testVectors);
for i=1:number_of_samples
    for k =1:number_of_classes
        d(k) = norm(testVectors(:,i) -m(:,k))^2;
    end
    [~,pred_test_lbls(i)] = min(d) ; 
end

index = 1:size(testVectors,2);
file = fopen('submissions/testLbls_nc.txt','w');
fprintf(file,'%s,%s\n','ID','Label');
fprintf(file,'%d,%d\n',[index; pred_test_lbls]);
fclose(file);
disp('[*] Done!')

end

