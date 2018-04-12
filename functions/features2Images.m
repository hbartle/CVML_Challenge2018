function features2Images(trainVectors,valVectors,testVectors)
%FEATURES2IMAGES Convert feature vectors to images

L = size(trainVectors,2);
traindir = 'data/train/cnnfeatures/';
mkdir(traindir)
for i=1:L
    imwrite(reshape(trainVectors(:,i),[64,64]),[traindir,'image',num2str(i),'.png']);
end

M = size(valVectors,2);
valdir = 'data/validation/cnnfeatures/';
mkdir(valdir)
for i=1:M
    imwrite(reshape(valVectors(:,i),[64,64]),[valdir,'image',num2str(i),'.png']);
end

N = size(testVectors,2);
testdir = 'data/test/cnnfeatures/';
mkdir(testdir)
for i=1:M
    imwrite(reshape(testVectors(:,i),[64,64]),[testdir,'image',num2str(i),'.png']);
end



end

