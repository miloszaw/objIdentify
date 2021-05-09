# Banana Age Identification

This program is designed to identify freshness of bananas by utilizing a Convoluted Neural Network. It is written entirely in MATLAB.

Utilizes a self-made dataset for training
# Creating the dataset
<b>This part is included in the readme for project documentation purposes.</b> <br>
10 bananas each.</br>Take a picture every day of every banana for a week.</br>Every person assigned different conditions to diversify dataset.</br>Proposed Conditions: </br>
- Banana in fridge/cold environment (small batch of dataset, 3-5 bananas total?)

- "Normal" conditions (makes most sense for this to be the largest batch of our dataset)

- Rich sunlight

- Professional-grade light</br>

Proposed timeslot:</br>Easter Week - Mon 29.03 - Sun 04.04</br>This should create a rich and controlled dataset of 240-280 pictures

<b>The dataset is created, and used for training, testing and confirming the program functionality. It is included in the project repository.</b>

# Program Features
<b> semanticSegmentationCNN.m </b>
  
Using sematic segmentation with convoluted neural network the model tries to diffirentiate between a normal banana, a bad banana, background 
and then label the input image with the correct label corresponding to the pixel value.  
  
The script semanticSegmentationCNN.m provides to core functions readWriteImgFilesFromToFolder(path, operation) and cnn().  
readWriteImgFilesFromToFolder(path, operation) takes images from a folder specified in path, it resizes them to 224x224x3. 
Based on the operator variable it either writes these new resized images to disk in the folder "resizedDatasetBananas" 
with a uniqe number for each image and the day the picture was taken, or it shoves them into variable.  
  
cnn() is a convoluted neural network that performes semantic segmentation. Each image from resizedDatasetBananas has a ground trouth label, 
the gTruth labels are located in the folder "PixelLabelData_2". Training of the model is done on a CUDA GPU to speed up processing time, 
if CUDA GPU is unavailable on your system then change the 'ExecutionEnvironment' to cpu. The semanticseg method takes an image, it then 
runs the image trough the CNN and returns a semantic segmentation of the input image, the returned image is then overlayed with the gTruth labels.  

# Issues
* The program struggles to identify late stage (dark brown/black) bananas.
* There are also issues with identifying bananas where the background of the image correlates to classification-classes.

# Further Development
This program is part of the project work for the Computer Vision course. It is programmed in an easily expandable way. Functionality currently is as described above. The program needs more labelled data to learn from, especially with late-stage (dark brown/black) bananas. By further diversifying the dataset, a much better accuracy can be achieved.<br> <br>
More classification-classes can also be added. For this functionality to be implemented, the entire dataset needs to be re-labeled to fit the new classification-classes. <br> <br> <br>

Authored and developed by Group 5 for IDATG2206 - Computer Vision - Project - Spring 2020 </br>
Group members:</br>
Andreas Blakli (<b>andrbl</b>) </br>
Kristian Aakervik Rønning (<b>krisaro</b>) </br>
Milosz Antoni Wudarczyk (<b>miloszaw</b>) </br>
Kristian Amundsen Øhman-Norén (<b>kristaoh</b>) </br>
