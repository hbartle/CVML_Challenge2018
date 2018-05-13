
imdsTrain = imageDatastore('data/train/TrainImagesSorted/');
imdsTrain.Labels = categorical(trainLbls);

imdsVal = imageDatastore('data/validation/ValidationImagesSorted/');
imdsVal.Labels = categorical(valLbls);

imdsTest = imageDatastore('data/test/TestImagesSorted/');


%%

% net = alexnet;
% inputSize = net.Layers(1).InputSize;
% layersTransfer = net.Layers(1:end-3);
% numClasses = numel(categories(imdsTrain.Labels));
% layers = [
%     layersTransfer
%     fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%     softmaxLayer
%     classificationLayer];
% 
% 
% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
%     'DataAugmentation',imageAugmenter);
% augimdsVal = augmentedImageDatastore(inputSize(1:2),imdsVal);
% 
% options = trainingOptions('adam', ...
%     'ExecutionEnvironment','parallel',...
%     'Shuffle','every-epoch',...
%     'MiniBatchSize',64, ...
%     'MaxEpochs',6, ...
%     'InitialLearnRate',1e-4, ...
%     'ValidationData',augimdsVal,...
%     'ValidationFrequency',3, ...
%     'ValidationPatience',Inf, ...
%     'Verbose',false ,...
%     'Plots','training-progress');
% 
% netTransfer = trainNetwork(augimdsTrain,layers,options);
% 
% 
% 
% %%
% net = googlenet;
% 
% inputSize = net.Layers(1).InputSize;
% 
% lgraph = layerGraph(net);
% % figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
% % plot(lgraph)
% 
% %lgraph = removeLayers(lgraph, {'predictions','predictions_softmax','ClassificationLayer_predictions'});
% lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
% 
% numClasses = numel(categories(imdsTrain.Labels));
% newLayers = [
%     fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','classoutput')];
% lgraph = addLayers(lgraph,newLayers);
% 
% %lgraph = connectLayers(lgraph,'avg_pool','fc');
% lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');
% 
% % figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% % plot(lgraph)
% % ylim([0,10])
% 
% layers = lgraph.Layers;
% connections = lgraph.Connections;
% 
% %layers(1:822) = freezeWeights(layers(1:822));
% layers(1:110) = freezeWeights(layers(1:110));
% 
% lgraph = createLgraphUsingConnections(layers,connections);
% 
% 
% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
%     'DataAugmentation',imageAugmenter);
% 
% augimdsVal = augmentedImageDatastore(inputSize(1:2),imdsVal);
% 
% options = trainingOptions('sgdm', ...
%     'ExecutionEnvironment','parallel',...
%     'Shuffle','every-epoch',...
%     'MiniBatchSize',100, ...
%     'MaxEpochs',6, ...
%     'InitialLearnRate',1e-4, ...
%     'ValidationData',augimdsVal,...
%     'ValidationFrequency',3, ...
%     'ValidationPatience',Inf, ...
%     'Verbose',false ,...
%     'Plots','training-progress');
% 
% 
% net = trainNetwork(augimdsTrain,lgraph,options);

%%
% net = inceptionresnetv2;
% 
% inputSize = net.Layers(1).InputSize;
% 
% lgraph = layerGraph(net);
% % figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
% % plot(lgraph)
% 
% lgraph = removeLayers(lgraph, {'predictions','predictions_softmax','ClassificationLayer_predictions'});
% 
% numClasses = numel(categories(imdsTrain.Labels));
% newLayers = [
%     fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','classoutput')];
% lgraph = addLayers(lgraph,newLayers);
% 
% lgraph = connectLayers(lgraph,'avg_pool','fc');
% 
% % figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% % plot(lgraph)
% % ylim([0,10])
% 
% layers = lgraph.Layers;
% connections = lgraph.Connections;
% 
% layers(1:822) = freezeWeights(layers(1:822));
% 
% lgraph = createLgraphUsingConnections(layers,connections);
% 
% 
% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
%     'DataAugmentation',imageAugmenter);
% 
% augimdsVal = augmentedImageDatastore(inputSize(1:2),imdsVal);
% 
% options = trainingOptions('sgdm', ...
%     'ExecutionEnvironment','parallel',...
%     'Shuffle','every-epoch',...
%     'MiniBatchSize',100, ...
%     'MaxEpochs',6, ...
%     'InitialLearnRate',1e-4, ...
%     'ValidationData',augimdsVal,...
%     'ValidationFrequency',3, ...
%     'ValidationPatience',Inf, ...
%     'Verbose',false ,...
%     'Plots','training-progress');
% 
% 
% net = trainNetwork(augimdsTrain,lgraph,options);
% 


%%
% 
net = vgg16;
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];


pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true);
   % 'RandXTranslation',pixelRange, ...
    %'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsVal);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',50, ... 
    'ExecutionEnvironment','parallel',...
    'Shuffle','every-epoch',...
    'MaxEpochs',8, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation,...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false ,...
    'Plots','training-progress');


net = trainNetwork(augimdsTrain,layers,options);
