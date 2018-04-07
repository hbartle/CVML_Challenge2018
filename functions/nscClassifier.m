function [ labels ] = nscClassifier(data_train,data_test,labels_train,K)
%NSC_CLASSIFY Nearest Subclass Centroid Classifier
number_of_classes = 29;

subclass_mean = cell(1,number_of_classes);
% Calculate Subclasses 
for i=1:number_of_classes
    idx = find(labels_train == i);
    [~,subclass_mean{i}] = kmeans(data_train(:,idx)',K);
end


% Classify the test images
[~,number_of_samples] = size(data_test);
for i=1:number_of_samples
    for k =1:number_of_classes
        for j = 1:K
            d_sub(j) = norm(data_test(:,i) - subclass_mean{k}(j,:)')^2;
        end
        d(k) = min(d_sub);
    end
    [~,labels(i)] = min(d);
end


end

