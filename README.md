# A Graph-based Framework to Bridge Movies and Synopses
This repo holds the code and models for the movie-synopsis association framework
presented on ICCV2019.

**A Graph-based Framework to Bridge Movies and Synopses**

Authors: 
[Yu Xiong](http://www.xiongyu.me/home), 
[Qingqiu Huang](http://qqhuang.cn), 
Lingfeng Guo, 
[Hang Zhou](https://hangz-nju-cuhk.github.io/), 
[BoleiZhou](http://bzhou.ie.cuhk.edu.hk/), 
[Dahua Lin](http://dahua.me/). ICCV19, Seoul, South Korea.

[[Paper]]()
[[Supp]]()
[[Project Page]]()
[[Dataset Website]]()

## Getting Started
The following instructions will get the project set up on your local machine.

### Prerequisites
All the codes are tested on the following environments:
- Linux (Ubuntu 16.04)
- Python 3.6+
- Pytorch xxx

### Install Python Dependencies
Pip install the following dependencies:
```
mmcv >= 0.2.0
```

### Install Gurobi
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

#### Install Gurobi
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
Q. Python ``ImportError: libgurobi{version}.so: cannot open shared object file: No such file or directory``

A. export gurobi lib to your LD_LIBIRARY_PATH by:
```
export GUROBI_HOME="/path/of/gurobi810(or_other_version)/linux64" # replace the path
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

### Fetch the code
```
git clone git@github.com:ycxioooong/MovieSynopsisAssociation.git
```

### Prepare the data
#### Download
Download our MSA (Movie Synopsis Association) dataset from
[Google Drive]() or [Baidu Pan]().
Note that we only provide extrated features at this moment due to legal issues.
More information of our dataset will be updated at the 
[MovieNet Dataset Website]().

#### Create soft link
