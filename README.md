# LNS2+RL: Combining Multi-agent Reinforcement Learning with Large Neighborhood Search in Multi-agent Path Finding
This repository is the official implementation of [LNS2+RL: Combining Multi-agent Reinforcement Learning with Large Neighborhood Search in Multi-agent Path Finding]() 
The paper is currently under review. 
Part of the C++ code in this repository draws inspiration from [MAPF-LNS2: Fast Repairing for Multi-Agent Path Finding via Large Neighborhood Search](https://github.com/Jiaoyang-Li/MAPF-LNS2)


## Installation
This is a hybrid C++/Python project. 
The neural network is written in Python, and part of the code for the simulation environment and the algorithm is written in C++.
We use pybind11 to bind the two languages.

### Python 
For optimal speed of the algorithm we recommend using python=3.11.

```
conda create --name myenv python=3.11
conda activate myenv
pip install -r requirements.txt
```
Then install torch manually 
```
pip install torch==2.1.1+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### C++
The C++ code requires the external libraries BOOST (https://www.boost.org/) and Eigen (https://eigen.tuxfamily.org/). 
Here is an easy way of installing the required libraries on Ubuntu:
```
sudo apt update
```
   * Install the boost library
```
    sudo apt install libboost-all-dev
```
   * Install the Eigen library (used for linear algebra computing)
```
    sudo apt install libeigen3-dev
```

## Run
After installing the above libraries, you can run the code by following the instructions below.

### C++ Compile
No matter which branch you use, you must first compile the included C++ code separately using CMake. 
Begin by cd to the directory, then run

```
mkdir build
cd build
cmake ..
make 
```
The directory name for the C++ code that needs to be compiled is 'lns2' in both the main and 'second\_stage' branches. 
In the 'LNS2\_RL\_eval' branch, the directory names are 'lns2' and 'mapf_env'.

### Training
The complete training of the MARL model consists of two stages.
The code in the main branch is used for the first training stage, and the code in the "second\_stage" branch is used for the second training stage.

To begin the first training stage, cd to the directory of the downloaded main branch code and then run:
```
CUDA_VISIBLE_DEVICES=gpu_ids python driver.py
```
The model generated at this stage will be saved to the *./models* directory.

After the first training stage is complete, modify the *driver.py* file in the code downloaded from the 'second\_stage' branch. 
Change the variable named *restore_path* on line 23 to the path of the last model saved from the first training stage (typically named *./final*).
Then start the second training stage by running

```
CUDA_VISIBLE_DEVICES=gpu_ids python driver.py
```
The model finally saved in the second training stage is the model used in LNS2+RL

### Evaluation
Use the code in the 'LNS2\_RL\_eval' branch to evaluate the performance of LNS2+RL.
To evaluate LNS2+RL on a specific task set, you need to first generate or download the task set and then modify the variable named *FOLDER\_NAME* on line 32 of the *multi\_eval.py* file to the folder name of the task set.
The variable named *model\_path* on line 53 of the *multi\_eval.py* file also needs to be modified to the path of the saved MARL model.
Finally, start the evaluation by run 
```
CUDA_VISIBLE_DEVICES=gpu_ids python multi_eval.py
```
This multi-process evaluation code will print the test results in the terminal.
If wandb=True is set, these print results can be found in the wandb logs file.

We provide an example task set "maps_60_10_10_0.175" in this repo.
More task sets we evaluated in the paper and the fully trained MARL model can be download from [https://www.dropbox.com/scl/fo/bmn29rfzeb84ipgs81kqe/ADPMx_VNpDAdU_GEsNo9xnM?rlkey=i2d8gt4n1dfntt938s7asoq8a&st=bfz5revv&dl=0](https://www.dropbox.com/scl/fo/bmn29rfzeb84ipgs81kqe/ADPMx_VNpDAdU_GEsNo9xnM?rlkey=i2d8gt4n1dfntt938s7asoq8a&st=bfz5revv&dl=0)


# license 
The code is released under the MIT License.
