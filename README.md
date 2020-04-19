# A Graph-Based Framework to Bridge Movies and Synopses
This repo holds the code and models for the movie-synopsis association framework
presented on ICCV2019:

Title: **A Graph-Based Framework to Bridge Movies and Synopses**,
ICCV19, Seoul, South Korea.

Authors: 
[Yu Xiong](http://www.xiongyu.me/home), 
[Qingqiu Huang](http://qqhuang.cn), 
[Lingfeng Guo](), 
[Hang Zhou](https://hangz-nju-cuhk.github.io/), 
[Bolei Zhou](http://bzhou.ie.cuhk.edu.hk/), 
[Dahua Lin](http://dahua.me/). 

Useful Links:
[[Paper]](http://xiongyu.me/src/conference/iccv19/0385.pdf)
[[Arxiv]]()
[[Project Page]](http://xiongyu.me/projects/moviesyn)
[[Dataset Website(Comming soon)]]()

## News
- [10/2019] We are still cleaning and expanding this dataset.

## TODO
- [ ] Update to a newer version of PyTorch
- [ ] Data preprocessing codes
- [ ] Release the whole dataset

## Getting Started
The following instructions will get the project set up on your local machine.

### I. Prerequisites
All the codes are tested on the following environments:
- Linux (Ubuntu 16.04)
- Python >= 3.6
- PyTorch >= 0.4.1
- CUDA 9.2

### II. Install Python Dependencies
Pip install the following dependencies:
```
mmcv >= 0.2.0
```

### III. Install Gurobi
Gurobi is a powerful optimization solver. 
We use gurobi for solving the graph matching problem. 
Follow the instructions below to install gurobi with ``Free Academic License``.
For other license or detailed official instructions,
please visit [here](https://www.gurobi.com/documentation/8.1/quickstart_mac/the_gurobi_python_interfac.html).

#### Download Gurobi
1. Login or register an account using academic email address at [here](https://www.gurobi.com).
2. Enter [this page](https://www.gurobi.com/academia/academic-program-and-licenses/) 
and download gurobi optimizer (linux, version>=8.1.0)
3. Issue an academic lisence for your local machine. 

#### Install Gurobipy
Take Gurobi-8.1.0 for example, enter the folder that store the gurobi source file, unzip it
```
tar -xvzf gurobi8.1.0_linux64.tar.gz
```
And then install gurobipy by
```
cd gurobi810/linux64
python setup.py install
```

#### FAQ
Q1. Python ``ImportError: libgurobi{version}.so: cannot open shared object file: No such file or directory``

A1. export gurobi lib to your ``LD_LIBIRARY_PATH`` by:
```
export GUROBI_HOME="/path/of/gurobi810(or_other_version)/linux64" # replace the path
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

### IV. Fetch the code
```
git clone git@github.com:ycxioooong/MovieSynopsisAssociation.git
```

### V. Prepare the data
#### Download
Download our MSA (Movie Synopsis Association) dataset from
[Google Drive]() or [Baidu Pan]().
Note that we only provide extrated features at this moment due to legal issues.
More information of our dataset will be updated at the 
[Dataset Website]().

Unzip the dataset by
```
unzip xxx.zip
```

#### Create soft link
```
cd /path/of/MovieSynopsisAssociation
ln -s /path/of/your/msa/dataset/directory data
```

### VI. Training
#### Train element embedding networks
First we train embedding networks for appearance feature and action feature respectively by
```
python tools/train.py config/CONFIG_OF_APPR_BASELINE.py --work_dir work_dir/WORK_DIR_NAME --validate
```


## Other Information
### Citation
```
@InProceedings{Xiong_2019_ICCV,
author = {Xiong, Yu and Huang, Qingqiu and Guo, Lingfeng and Zhou, Hang and Zhou, Bolei and Lin, Dahua},
title = {A Graph-Based Framework to Bridge Movies and Synopses},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

### Related Projects
This project is part of the ``Movie Understanding Project``. The other related movie projects are:

#### Actor Recognition
- [Cast In Movies](https://arxiv.org/pdf/1806.03084.pdf): Unifying Identification and Context Learning for Person Recognition. [[Project Page]](http://qqhuang.cn/projects/cvpr18-person-recognition/)
- [Cast Search](https://arxiv.org/pdf/1807.10510.pdf): Person Search in Videos with One Portrait
Through Visual and Temporal Links. [[Project Page]](http://qqhuang.cn/projects/eccv18-person-search/)

#### Trailer Analytics
- [Trailer to Movie](https://arxiv.org/pdf/1806.05341.pdf): From Trailers to Storylines: An Efficient Way to Learn from Movies

### Contact
```
Xiong Yu: xy017@ie.cuhk.edu.hk
```