# Power_Cable_Detection
This project was for bachelor thesis.

I used USF (University of South Florida) wire dataset to train the network. The data set contains wire images along with the coordinates of the starting and ending point of the wires stored in a txt file. I created the groundtruth images from these coordinates myself. The python codes for creating the dataset is in trans.py file. To expand the dataset, the images were chopped and rotated along with their groundtruth.

I have used Hough Transform as a post processing step to abtain the coordinate of the detected wires.

For the original U-net network go to : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/.

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
