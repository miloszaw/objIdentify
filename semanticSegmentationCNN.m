% Program that labels bananas using semantic segmentation in a convoluted
% neural network
clc;
clear all;
close all;
main();

% Runs program and handles all of the functions
function main()
    path = '.\bananasImgs\';                                % Path to full resolution original images taken of bananas
    %[~,~] = readWriteImgFilesFromToFolder(path, 1);        % Uncomment this when creating dataset for use in CNN
    cnn();
end

% Reads images from file, writes resized images to folder, specify function
% variable "operation" for what you need the function to do
% 1 = read, resize and write to disk
% 2 = read, resize, put all img in variable which is returned as allImg
function [allImg, numOfFiles] = readWriteImgFilesFromToFolder(path, operation)
    dayCount = 0;
    destFolder = '.\resizedDatasetBananas';
    switch operation
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
        currImg = imresize(currImg, [224 224]);     % Image resized to 224x224x3
        switch operation
            case 1                                  % Creates new folder if it doesn't already exist
                                                    % Numerates images with banana Number and Day                                                    
                if exist(destFolder, 'dir')
                    if(mod(i-1, 10) == 0)
                        dayCount = dayCount + 1;
                    end
                    countStr = num2str(i);          % This is retarded
                    dayCountStr = num2str(dayCount);
                    imgString = append(destFolder, '\', 'banana', countStr, '_day', dayCountStr, '_Resized.jpg');
                    imwrite(currImg, imgString);                    
                end
                allImg = 0;                         % allImg needs a value because it's returned
            case 2
                allImg{i} = currImg;
        end
    end
end

% Convoluted neural network that performs semantic segmentation
function cnn()

    pathTestImgDataset = '.\testImgDataset\';
    pathImgDS = '.\resizedDatasetBananas';
    pathLab = '.\PixelLabelData_2\';

    classNames = ["normalbanana", "badbanana", "background"];
    labelIDs   = [1 2 3];
    
    if exist('trainedNet.mat', 'file')
        net = load('trainedNet.mat').net;
    else
        trainDatasetDir = fullfile(pathImgDS);
        trainImgDir = fullfile(trainDatasetDir, '*.jpg');
        labelDir = fullfile(pathLab, '*.png');
        trainImgds = imageDatastore(trainImgDir);    
        pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);   

        trainingData = pixelLabelImageDatastore(trainImgds,pxds);
        
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
            'MaxEpochs',1, ...
            'MiniBatchSize',64, ...
            'ExecutionEnvironment','cpu');      % Change this to 'cpu' if CUDA gpu is not available

        net = trainNetwork(trainingData,layers,opts);
        save('trainedNet.mat');	
    end
    
    [testImage, ~] = readWriteImgFilesFromToFolder(pathTestImgDataset, 2); 
    for i = 1:size(testImage,2)
            [C, score, allScores] = semanticseg(testImage{i},net);
        B = labeloverlay(testImage{i},C);
        c = size(C);
        bb = 0;
        gb = 0;
        for y=1:c(1)
            for x=1:c(2)
                switch C(y, x)
                    case classNames(1)
                        gb = gb + 1;
                    case classNames(2)
                        bb = bb + 1;
                end
            end
        end
        totalBBandGB = gb + bb;
        percentOfbb = (bb / totalBBandGB);
        disp(percentOfbb)
        strPrediction = "";
        if(percentOfbb > 0.5)
            strPrediction = "The banana has gone bad";
        else
            strPrediction = "The banana is still good";
        end
        dim = [.5 .6 .3 .3];
        figure,
        montage({B, testImage{i}}),
        annotation('textbox',dim,'BackgroundColor','w','String',(1-percentOfbb),'FitBoxToText','on');
        disp(strPrediction);
    end

    disp("Program finished")    
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