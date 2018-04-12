function [score_svm] = svm(trainVectors,trainLbls,valVectors,valLbls,testVectors)
%SVM Support Vector Machine

disp('***********************************')
disp('Support Vector Machine')
disp('[*] Training SVM...')
% clear model
model = fitcecoc(trainVectors',trainLbls);

disp('[*] Classify Validation Images...')
pred_val_lbls = model.predict(valVectors');
score_svm=length(find(pred_val_lbls-valLbls==0))/length(valLbls);
disp(['[*] Max Score: ',num2str(score_svm*100),'%'])

disp('[*] Training SVM...')
model = fitcecoc([trainVectors valVectors]',[trainLbls; valLbls]);
disp('[*] Classify Test Images...')
pred_test_lbls = model.predict(testVectors');

index = 1:size(testVectors,2);
file = fopen('submissions/testLbls_svm.txt','w');
fprintf(file,'%s,%s\n','ID','Label');
fprintf(file,'%d,%d\n',[index; pred_test_lbls']);
fclose(file);
disp('[*] Done!')


end

