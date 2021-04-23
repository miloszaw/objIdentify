function main()

    path = '.\testFolder\';

    dayCount = 0;
    destFolder = '.\testFolder2';
    imgFile = dir([path '*.jpg']);
    if ~exist(destFolder, 'dir')
        mkdir(destFolder);
    end

    numOfFiles = length(imgFile);  
    for i = 1:numOfFiles
        currFilename = imgFile(i).name;
        currImg = imread([path currFilename]);
        cis = size(currImg);
        currImg = currImg(100:cis(2)-100, :, min(1:3, end));
        currImg = imresize(currImg, [224 224]);     % Image resized to 224x224x3                                                
        if exist(destFolder, 'dir')
            if(mod(i-1, 10) == 0)
                dayCount = dayCount + 1;
            end
            imgString = append(destFolder, '\', 'banana',num2str(i), '_day', num2str(dayCount), '_Resized.jpg');
            imwrite(currImg, imgString);                    
        end
        allImg = 0;                         % allImg needs a value because it's returned
    end
end