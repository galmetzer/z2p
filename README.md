# Z2P: Instant Visualization of Point Clouds

<img src='https://galmetzer.github.io/assets/img/z2p.png' align="right" width=325>

### Eurographics 2022 [[Paper]](https://arxiv.org/abs/2105.14548) [[Demo]](https://huggingface.co/spaces/galmetzer/z2p)<br>
by [Gal Metzer](https://galmetzer.github.io/), [Rana Hanocka](https://www.cs.tau.ac.il/~hanocka/), [Raja Giryes](http://web.eng.tau.ac.il/~raja), [Niloy J. Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/) and [Daniel Cohen-Or](https://danielcohenor.com/)

# Getting Started

### Installation
The code relies on [PyTorch](https://pytorch.org/) version 1.8.1 and other packages specified in requirements.txt

#### Setup Conda Environment 
- Change cuda version in `install.sh`
- Run `install.sh` to create a conda environment and install everything
  
# Running 

## Data
The datasets used for the paper can be downloaded from google drive. <br>
Both links contain scripts to .zip files that contain the datasets, please extract before training. <br> 
Each set should take up about 16GB or disk space. 
If you wish to use the ***cache option*** at train time please space for around 350GB of disk space. <br>
This will save the 2D point cloud z-buffers to disk and allow for faster training. 


### Simple dataset
[Train](https://drive.google.com/file/d/1-cUSVSVOX2pwVoCn1qekYjHnYrZVBeDs/view?usp=sharing), 
[Test](https://drive.google.com/file/d/1YvsHuaGV_2KsgkinJtbER0zojhkcuZpK/view?usp=sharing)

This dataset only allows control over the color and light direction, and uses a simple diffuse material.

### Metal-Roughness dataset
[Train](https://drive.google.com/file/d/11K-Kd17QOPWm8p7DsQzVHjyBVLiYR2BY/view?usp=sharing), 
[Test](https://drive.google.com/file/d/1htACHARgjuzFu5kNdm-X9Sc4Ur-8vnzd/view?usp=sharing)

This dataset also augments the metallic and roughness of the shape, and allows control over them as well. 

## Training
First make sure the datasets are downloaded and extracted to disk.

There are two training scripts ``train_regular.sh`` and ``train_metal_roughness.sh``, 
corresponding to the dataset that should be used for training.<br>
Both scripts can be found in the ``scripts`` folder and require three inputs: trainset dir, testset dir, export dir.

For example:  
```
train_regular.sh /home/gal/datasets/renders_shade_abs /home/gal/datasets/renders_shade_abs_test /home/gal/exports/train_regular
```

```
train_metal_roughness.sh /home/gal/datasets/renders_mr /home/gal/datasets/renders_mr_test /home/gal/exports/train_mr
```

## Inference
Inference with the pre-trained demos is available in an interactive [demo app](https://huggingface.co/spaces/galmetzer/z2p), 
as well as with demo scripts in this repo. <br>
The ``scripts`` folder containes two inference scripts: 
* ``inference_goat.sh``
* ``inference_chair.sh``

The scripts require an ``export dir`` output, for example:  

```
scripts/inference_goat.sh /home/gal/exports/goat_results
```

The scripts use ``inference_pc.py`` which allows for more inference options like:
* ``--show_results`` enables showing the results with matplotlib instead of exporting them
* ``--checkpoint`` enables loading a trained checkpoint instead of the pretrained models pulled from drive
* ``--model_type`` toggle between ``regular`` and ``metal_roughness`` models 
* ``--rx``, ``--ry``, ``--rz`` rotate the pc before projecting
* ``--rgb`` control over the color parameter
* ``--light`` control over the lights parameters
* ``--metal``, ``--roughness`` control over the metal and roughness parameters

and more options accessible through ``python inference_pc.py --help``
 

# Citation
If you find this code useful, please consider citing our paper
```
@article{metzer2021z2p,
author={Metzer, Gal and Hanocka, Rana and Giryes, Raja and Mitra, Niloy J and Cohen-Or, Daniel},
title = {Z2P: Instant Visualization of Point Clouds},
journal = {Computer Graphics Forum},
volume = {41},
number = {2},
pages = {461-471},
doi = {https://doi.org/10.1111/cgf.14487},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14487},
year = {2022}
}
```

# Questions / Issues
If you have questions or issues running this code, please open an issue.
