function [ score_nsc ] = nscClassifier(trainVectors,trainLbls,valVectors,valLbls,testVectors)
%NSC_CLASSIFY Nearest Subclass Centroid Classifier
disp('***********************************')
disp('Nearest Subclass Centroid Classification')
disp('[*] Optimizing over hyperparameter space...')

number_of_classes = 29;
K = 4;

for c=1:K
    subclass_mean = cell(1,number_of_classes);
    % Calculate Subclasses
    for i=1:number_of_classes
        idx = find(trainLbls == i);
        [~,subclass_mean{i}] = kmeans(trainVectors(:,idx)',c);
    end
    
    
    % Classify the test images
    [~,number_of_samples] = size(valVectors);
    for i=1:number_of_samples
        for k =1:number_of_classes
            for j = 1:c
                d_sub(j) = norm(valVectors(:,i) - subclass_mean{k}(j,:)')^2;
            end
            d(k) = min(d_sub);
        end
        [~,pred_val_lbls(i)] = min(d);
    end
    nsc_sc(c) = length(find(pred_val_lbls-valLbls'==0))/length(valLbls);
    disp(['[*]' num2str(c) ' of ' num2str(K)])
end
[score_nsc, Kmax] = max(nsc_sc);
disp(['[*] K: ',num2str(Kmax)])
disp(['[*] Max Score: ',num2str(score_nsc*100),'%'])

%% Test images

disp('[*] Classify training images...')
subclass_mean = cell(1,number_of_classes);

% Calculate Subclasses
for i=1:number_of_classes
    idx = find(trainLbls == i);
    [~,subclass_mean{i}] = kmeans(trainVectors(:,idx)',Kmax);
end


% Classify the test images
[~,number_of_samples] = size(testVectors);
for i=1:number_of_samples
    for k =1:number_of_classes
        for j = 1:Kmax
            d_sub(j) = norm(testVectors(:,i) - subclass_mean{k}(j,:)')^2;
        end
        d(k) = min(d_sub);
    end
    [~,pred_test_lbls(i)] = min(d);
end

% Safe to file
index = 1:size(testVectors,2);
file = fopen('submissions/testLbls_nsc.txt','w');
fprintf(file,'%s,%s\n','ID','Label');
fprintf(file,'%d,%d\n',[index; pred_test_lbls]);
fclose(file);
disp('[*] Done!')


end

