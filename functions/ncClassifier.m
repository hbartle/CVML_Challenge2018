function [ labels,m ] = ncClassifier(data_train, data_test, labels_train, data_type)

number_of_classes= 29;

% Calculate Class Mean
for i=1:number_of_classes
    idx = find(labels_train == i);
    m(:,i)= mean(data_train(:,idx),2);
end


% Classify the test images
[~,number_of_samples] = size(data_test);
for i=1:number_of_samples
    for k =1:number_of_classes
        d(k) = norm(data_test(:,i) -m(:,k))^2;
    end
    [~,labels(i)] = min(d) ; 
end
end

