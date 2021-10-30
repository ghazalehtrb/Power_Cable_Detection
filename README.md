# Power_Cable_Detection

This project was for my bachelor's dissertation.
The U-net encoader-decoder network is trained for semantic segmentation of power cables. 
I used USF (University of South Florida) wire dataset to train the network. The data set contains wire images along with the coordinates of the starting and ending points of the wires stored in a txt file. I generated the groundtruth images from these coordinates myself. The python codes for creating the dataset is in trans.py file. To expand the dataset, the images were chopped and rotated together with their groundtruth.

I have used Hough Transform as a post processing step to abtain the coordinate of the detected wires.

Please refer to https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ for the original U-net network.

## data folders setup

* all_data1
* └───train
    * │─── image
    * │─── mask

* └───test
    * │─── image
    * │─── mask
    * │─── preds
    
## Results

![Alt Text](https://github.com/ghazalehtrb/Power_Cable_Detection/blob/master/results/w.jpg)
![Alt Text](https://github.com/ghazalehtrb/Power_Cable_Detection/blob/master/results/ww.png)
![Alt Text](https://github.com/ghazalehtrb/Power_Cable_Detection/blob/master/results/wireee.jpg)
![Alt Text](https://github.com/ghazalehtrb/Power_Cable_Detection/blob/master/results/wire.png)
