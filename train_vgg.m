rootFolder = fullfile('/home/prashant/101_ObjectCategories');
categories = {'ant', 'ferry', 'laptop'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds);

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category


% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

countEachLabel(imds)

imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

[trainingImages,validationImages] = splitEachLabel(imds,0.7,'randomized');

layers = importCaffeLayers('/home/prashant/vgg.deploy.prototxt');

options = trainingOptions('sgdm','MaxEpochs',100,... 
      'MiniBatchSize',300,...
      'CheckpointPath','/home/prashant/Data/VGG16');

net_new = trainNetwork(trainingImages,layers,options);

save net_new alexnet