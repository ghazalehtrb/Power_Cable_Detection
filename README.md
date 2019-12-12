# Power_Cable_Detection
This project was for bachelor thesis at Amirkabir University of Iran.

I used USF (University of South Florida) wire dataset to train the network. The data set contain wire images along with the coordinates of the starting and ending point of the wires. I created the groundtruth images from these coordinates myself. The python codes for creating the dataset is in trans.py file. To expand the dataset, the images were chopped and rotated along with their groundtruth.

For the original U-net network go to : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/.

## data folders setup

all_data1
└───train
    │─── image
    │─── mask

└───test
    │─── image
    │─── mask
    │─── preds
    
## Results

![GitHub Logo](/images/logo.png)
Format: ![Alt Text](url)
