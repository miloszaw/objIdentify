# Banana Age Identification

Program designed to identify age of bananas within a set margin

Utilizes a self-made dataset for training
# Creating the dataset
10 bananas each.</br>Take a picture every day of every banana for a week.</br>Every person assigned different conditions to diversify dataset.</br>Proposed Conditions: </br>
- Banana in fridge/cold environment (small batch of dataset, 3-5 bananas total?)

- "Normal" conditions (makes most sense for this to be the largest batch of our dataset)

- Rich sunlight

- Professional-grade light</br>

Proposed timeslot:</br>Easter Week - Mon 29.03 - Sun 04.04</br>This should create a rich and controlled dataset of 240-280 pictures

# Program Features
<b> semanticSegmentationCNN.m </b>
Using sematic segmentation with convoluted neural network the model tries to diffirentiate between a normal banana, a bad banana, background 
and then label the input image with the correct label corresponding to the pixel value.  
  
The script semanticSegmentationCNN.m provides to core functions readWriteImgFilesFromToFolder(path, operation) and cnn().  
readWriteImgFilesFromToFolder(path, operation) takes images from a folder specified in path, it resizes them to 224x224x3. 
Based on the operator variable it either writes these new resized images to disk in the folder "resizedDatasetBananas" 
with a uniqe number for each image and the day the picture was taken, or it shoves them into variable.  
  
cnn() is a convoluted neural network that performes semantic segmentation. Each image from resizedDatasetBananas has a ground trouth label, 
the gTruth labels are located in the folder "PixelLabelData_2". Training of the model is done on a CUDA gpu to speed up processing time, 
if CUDA gpu is unavailable on your system then change the 'ExecutionEnvironment' to cpu. The semanticseg method takes an image, it then 
runs the image trough the CNN and returns a semantic segmentation of the input image, the returned image is then overlayed with the gTruth labels.  
</br> </br>

Authored and developed by Group 5 for IDATG2206 - Computer Vision - Project - Spring 2020 </br>
Group members:</br>
Andreas Blakli (<b>andrbl</b>) </br>
Kristian Aakervik Rønning (<b>krisaro</b>) </br>
Milosz Antoni Wudarczyk (<b>miloszaw</b>) </br>
Kristian Amundsen Øhman-Norén (<b>kristaoh</b>) </br>
