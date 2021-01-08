# pytorch-retinanet


Pytorch  implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

This implementation is primarily designed to be easy to read and simple to modify.


## Installation

1) Clone this repo

2) Install the required packages:

```
apt-get install tk-dev python-tk
```

3) Install the python packages:
	
```
pip install pandas
pip install pycocotools
pip install opencv-python
pip install requests

```

## Training

The network can be trained using the `train.py` script.  
The dataset must be in the csv format.   
Each row of the csv is of the form: \[image_path,x1,y1,x2,y2,labelname\]  

For the training script, a config file must be provided. An example config file can be seen at config.txt in the repo.

```
python train.py --h 
```

For optimizing the anchors for the dataset, you can use this repo: https://github.com/curbFlow/anchor-optimization.
