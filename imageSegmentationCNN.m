% Testing Image segmentation
clc;
main();

function main()
    path = '.\bananasImgs\';
    %[~,~] = readWriteImgFilesFromToFolder(path, 1);
    cnn();
end

% Reads images from file, writes resized images to folder, specify function
% variable "operation" for what you need the function to do
% 1 = read, resisze and write to disk
% 2 = read, resize, put all img in variable which is returned as allImg
function [allImg, numOfFiles] = readWriteImgFilesFromToFolder(path, operation)
    dayCount = 0;
    destFolder = '.\resizedDatasetBananas2';
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
            case 1
                if exist(destFolder, 'dir')
                    if(mod(i-1, 10) == 0)
                        dayCount = dayCount + 1;
                    end
                    countStr = num2str(i);                  % This is retarded
                    dayCountStr = num2str(dayCount);
                    imgString = append(destFolder, '\', 'banana', countStr, '_day', dayCountStr, '_Resized.jpg');
                    imwrite(currImg, imgString);                    
                end
                allImg = 0;
            case 2
                allImg{i} = currImg;
        end
    end
end

function cnn()
    path = '.\day01\';
    pathIm = '.\test3imgtrain\';
    pathImgDS = '.\resizedDatasetBananas';%'.\resizedImages';
    pathLab = '.\PixelLabelData_2\';%'.\PixelLabelData_1\';
    dataSetDir = fullfile(pathImgDS);
    imgDir = fullfile(dataSetDir, '*.jpg');
    labelDir = fullfile(pathLab, '*.png');
    imds = imageDatastore(imgDir);
    %imds = imresize(imdsTemp, [32, 32]);
    
    %[allImgTemp, ~] = readWriteImgFilesFromToFolder(pathIm, 1);
    %imds = imageDatastore('.\resizedImages', '*.jpg' );
    
    classNames = ["normalbanana", "badbanana", "background"];
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
    [testImage, ~] = readWriteImgFilesFromToFolder(pathImgDS, 2);
    
    C = semanticseg(testImage{98},net);
    B = labeloverlay(testImage{98},C);%, 'IncludedLabels', "banana", 'Colormap','autumn','Transparency',0.25);
    imshow(B)
    %tempLabelImg = imcrop(testImage{1}, B);
    %imshow(tempLabelImg);
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