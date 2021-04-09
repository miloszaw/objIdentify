% Testing Image segmentation
clc;
main();

function main()
%readImgFilesFromFolder();
cnn();
end

function [allImg, numOfFiles] = readImgFilesFromFolder(path, fileType)
    %path = '.\day01\';
    destFolder = '.\resizedImages';
    switch fileType
        case 1
            imgFile = dir([path '*.jpg']);
            if ~exist(destFolder, 'dir')
                mkdir(destFolder);
            end
        case 2
            imgFile = dir([path '*.jpg']); 
    end
    numOfFiles = length(imgFile);  
    for i = 1:numOfFiles
        currFilename = imgFile(i).name;
        currImg = imread([path currFilename]);
        currImg = currImg(:,:,min(1:3, end));
        %allImg{i} = imresize(currImg, [224 224]);
        currImg = imresize(currImg, [224 224]);
        switch fileType
            case 1
                if ~exist(destFolder, 'dir')
                    imgString = append(destFolder, '\', currFilename, 'Resized.jpg');
                    imwrite(currImg, imgString);                    
                end
                allImg = 0;
            case 2
                allImg{i} = currImg;
        end
    end
end

function data = customReadDatastoreImage(filename)  % code from matlab forum
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[224 224]);
end

function cnn()
    path = '.\day01\';
    pathIm = '.\test3imgtrain\';
    pathImgDS = '.\resizedImages';
    pathLab = '.\PixelLabelData_1\';
    dataSetDir = fullfile(pathImgDS);
    imgDir = fullfile(dataSetDir, '*.jpg');
    labelDir = fullfile(pathLab, '*.png');
    imds = imageDatastore(imgDir);
    %imds = imresize(imdsTemp, [32, 32]);
    
    [allImgTemp, ~] = readImgFilesFromFolder(pathIm, 1);
    %imds = imageDatastore('.\resizedImages', '*.jpg' );
    
    classNames = ["apple", "banana", "orange"];
    labelIDs   = [1 2 3];
    pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);   
    
    numFilters = 64;
    filterSize = 4;
    numClasses = 3;
    layers = [
        imageInputLayer([224 224 3])
        convolution2dLayer(filterSize,numFilters,'Padding',1) % 2
        reluLayer()
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(filterSize,numFilters,'Padding',2)
        reluLayer()
        transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
        convolution2dLayer(1,numClasses);
        softmaxLayer()
        pixelClassificationLayer()
    ]
    
    opts = trainingOptions('sgdm', ...
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',100, ...
        'MiniBatchSize',64);

    trainingData = pixelLabelImageDatastore(imds,pxds);
    net = trainNetwork(trainingData,layers,opts);
    [testImage, ~] = readImgFilesFromFolder(path, 2);
    
    C = semanticseg(testImage{1},net);
    B = labeloverlay(testImage{1},C);
    imshow(B)
    
end

% function segmentImageLazysnap()
% 	[I, N] = readImgFilesFromFolder();
%     disp(N);
%     %imageSegmenter(I{1});
%     
%     L = superpixels(I{1},500);
%     f = drawrectangle(gca,'Position',[100 128 350 150],'Color','g');
%     foreground = createMask(f,I{1});
%     b1 = drawrectangle(gca,'Position',[130 30 40 30],'Color','r');
%     b2 = drawrectangle(gca,'Position',[6 368 500 10],'Color','r');
%     background = createMask(b1,I{1}) + createMask(b2,I{1});
%     BW = lazysnapping(I{1},L,foreground,background);
%     imshow(labeloverlay(I{1},BW,'Colormap',[0 1 0]))
% end
% 
% function temp()
% 	inputSize = [32 32 3];
%     imgLayer = imageInputLayer(inputSize);
%     filterSize = 3;
%     numFilters = 32;
%     conv = convolution2dLayer(filterSize,numFilters,'Padding',1);
%     relu = reluLayer();
%     poolSize = 2;
%     maxPoolDownsample2x = maxPooling2dLayer(poolSize,'Stride',2);
%     downsamplingLayers = [
%         conv
%         relu
%         maxPoolDownsample2x
%         conv
%         relu
%         maxPoolDownsample2x
%     ]
%     filterSize = 4;
%     transposedConvUpsample2x = transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
%     upsamplingLayers = [
%         transposedConvUpsample2x
%         relu
%         transposedConvUpsample2x
%         relu
%     ]
%     numClasses = 3;
%     conv1x1 = convolution2dLayer(1,numClasses);    
%     finalLayers = [
%         conv1x1
%         softmaxLayer()
%         pixelClassificationLayer()
%     ]
% 
%     net = [
%         imgLayer    
%         downsamplingLayers
%         upsamplingLayers
%         finalLayers
%         %opts
%         ]
%     
% end