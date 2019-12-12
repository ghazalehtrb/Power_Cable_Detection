# Power_Cable_Detection
This project was for bachelor thesis at Amirkabir University of Iran.

I used USF (University of South Florida) wire dataset to train the network. The data set contain wire images along with the coordinates of the starting and ending point of the wires. I created the groundtruth images from these coordinates myself. The python codes for creating the dataset is in trans.py file. To expand the dataset, the images were chopped and rotated along with their groundtruth.

I have used Hough Transform as a post processing step to abtain the coordinate of the detected wires.

For the original U-net network go to : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/.

## data folders setup

1. all_data1
1. └───train
    1. │─── image
    1. │─── mask

1. └───test
    1. │─── image
    1. │─── mask
    1. │─── preds
    
## Results

![GitHub Logo](/images/logo.png)
Format: ![Alt Text](url)
