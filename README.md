# Swin-MLP model for the diagnosis of Helicobacter pylori
## Requirements
```
# [Optional] If you are using CUDA 11.0 or newer, please install `torch==1.7.1+cu110`
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
``` 
## Pretrained params and Datasets
We provide
* pretrained parameters on real Helicobacter pylori diagnostic data.  
The parameters are in the 'params' folder.
* Real multi-level infection intensity diagnostic data of Helicobacter pylori obtained from hospitals  
Download [HPMutiClass.rar (17.00GB)](https://pan.baidu.com/s/1xSQU_2TeO2OXBcbLZhIOJQ) with code "a5i7".
## Visualize lesion areas
Please run 'visualize. py' to automatically search for positive images in each patient's endoscopic image group and visualize the lesion area.
## Train
If you need to train the model from scratch, please run 'train. py'.