numImages=[100];
maxEpoch = [500];
batchSize = [320];
for ll = 1 : length(numImages)
    %filePath = strcat('C:/HMM/Dataset/',num2str(numImages(ll)));
    rootFolder = fullfile('C:/HMM/Dataset/500');
    categories = {'Cat','Dog','Clutter', 'Motorbikes','Airplanes'};

    imds = imageDatastore(fullfile(rootFolder,categories), 'LabelSource', 'foldernames');

    % Use splitEachLabel method to trim the set.
    imds = splitEachLabel(imds, numImages(ll) , 'randomize');
    imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

    N     = length(imds.Files);
    % Number of partitions
    k     = 5;
    % Scatter row positions
    pos   = randperm(N);
    % Bin the positions into k partitions

    edges = round(linspace(1,N+1,k+1));
    clear 'prtA' 'lblA'; 
    clear 'lSet' 'testingLbl';
    clear 'testingSet' 'trainingLbl';
    clear 'trainingSet' 'tSet';
%     prtA  = zeros((length(imds.Files)/k),5);
%     lblA  = zeros((length(imds.Files)/k),5);
%     tSet  = zeros((length(imds.Files)/k),4);
%     lSet  = zeros((length(imds.Files)/k),4);
%     testingSet = zeros((length(imds.Files)*0.2),5);
%     testingLbl = zeros((length(imds.Files)*0.2),5);
%     trainingSet = zeros((length(imds.Files)*0.8),5);
%     trainingLbl = zeros((length(imds.Files)*0.8),5);
    for ii = 1:k
        idx      = edges(ii):edges(ii+1)-1;
        prtA(:,ii) = imds.Files(pos(idx)); % or apply code to the selection of A
        lblA(:,ii) = imds.Labels(pos(idx));
    end
    for i = 1 : 5
        k = i;
        l = 1;
        testingSet(:,i) = prtA(:,i);
        testingLbl(:,i) = lblA(:,i);

        for j = 1 : 4
           if((k+j)>5)
              %disp(i);
              tSet(:,j) = prtA(:,l);
              lSet(:,j) = lblA(:,l);
              l = l+1;
           else
           tSet(:,j) = prtA(:,k+j);
           lSet(:,j) = lblA(:,k+j);    
           end
       
       %k = k + 1; 
        end
        trainingSet(:,i) = tSet(:);
        trainingLbl(:,i) = lSet(:);
    end
    for i = 1 : 5
       AlexnetLayers = importCaffeLayers('C:/HMM/deploy.prototxt');
       VGGLayers = importCaffeLayers('C:/HMM/vgg.deploy.prototxt');
       imdsTraining = imageDatastore(trainingSet(:,i),'Labels',trainingLbl(:,i));
       imdsVal = imageDatastore(testingSet(:,i),'Labels',testingLbl(:,i));
       imdsTraining.ReadFcn = @(filename)readAndPreprocessImage(filename);
       imdsVal.ReadFcn = @(filename)readAndPreprocessImage(filename);
       
       options = trainingOptions('sgdm','MaxEpochs',maxEpoch(ll),... 
          'InitialLearnRate',0.001,...
          'MiniBatchSize',batchSize(ll),...
          'ValidationData',{imdsTraining,imdsTraining.Labels},...
          'ValidationFrequency',600,...
          'Plots','training-progress');

        alexNett= trainNetwork(imdsTraining,AlexnetLayers,options);
        fname = strcat('Alexnett_',num2str(numImages(ll)),'_Images_',num2str(maxEpoch(ll)),'_iterations_cat_', num2str(i));
        save(fname, 'alexNett');

    %plotconfusion(valLabels,predictedLabels);
        predictedLabels = classify(alexNett,imdsVal);
        valLabels = imdsVal.Labels;
        
        alexAccurcy{ll,1}=strcat('Alexnett_Accuracy_',num2str(numImages(ll)),'_Images_',num2str(maxEpoch(ll)),'_iterations');
        alexAccuracy{ll,i+1} = sum(predictedLabels == valLabels)/numel(valLabels);
        
        vggNet= trainNetwork(imdsTraining,AlexnetLayers,options);
        fname2 = strcat('VGGNett',num2str(numImages(ll)),'_Images_',num2str(maxEpoch(ll)),'_iterations_cat_', num2str(i));
        save(fname2, 'vggNett');

    %plotconfusion(valLabels,predictedLabels);
        predictedLabels = classify(vggNet,imdsVal);
        valLabels = imdsVal.Labels;
        
        vggAccurcy{ll,1}=strcat('vggNett_Accuracy_',num2str(numImages(ll)),'_Images_',num2str(maxEpoch(ll)),'_iterations');
        vggAccuracy{ll,i+1} = sum(predictedLabels == valLabels)/numel(valLabels);
        if(ll==10)
            fname1 = strcat('Alexnett_Accuracy');
            save(fname1,'alexAccuracy');
            fname4 = strcat('vggNett_Accuracy');
            save(fname4,'vggAccuracy');
        end
    end
    clear 'prtA' 'lblA' 'lset' 'testingLbl' 'testingSet' 'trainingLbl' 'trainingSet' 'tSet' 'N' 'pos' 'imds' 
end