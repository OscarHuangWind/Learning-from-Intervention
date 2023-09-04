# Learmomg from Heterogeneous Intervention: Uncertainty-aware Human-guided Reinforcement Learning for Autonomous Driving
:dizzy: **A novel learning from heterogeneous intervention (LfHI) approach that considers the heterogeneity and intrinsic uncertainty of natural human behaviors in the context of the HiL-RL framework for learning robust driving policy under multi-human interventions and corresponding uncertainties.**

:wrench: Realized in SMARTS simulator with Ubuntu 20.04 and Pytorch. 

# Framework

<p align="center">
<img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/LfHI.png">
</p>

# User Guide

## Create a new virtual environment with dependencies.
```
conda env create -f environment.yml
```

## Activate virtual environment.
```
conda activate unahug
```

## Install Pytorch
Select the correct version based on your cuda version and device (cpu/gpu):
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Install the SMARTS.
```
# Download SMARTS
git clone https://github.com/huawei-noah/SMARTS.git
cd <path/to/SMARTS>

# Install the system requirements.
bash utils/setup/install_deps.sh

# Install smarts.
pip install -e '.[camera_obs,test,train]'

# Install extra dependencies.
pip install -e .[extras]
```

## Clone the repository.
cd to your workspace and clone the repo.
```
git clone https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation.git
```

## Build the scenario in case of adding new scenarios (No need for LeftTurn, RampMerge, and T-Intersection).
```
cd <path/to/UnaHug-RL>
scl scenario build --clean Scenario/Roundout/
```

## Visulazation
Type the following command in the terminal:
```
scl envision start
```
Then go to http://localhost:8081/

## DRL Training
```
python main.py
```

## Human Demonstration (collect human driver data through Logitech G29)
```
python demonstration.py
```

## Learn N-human Digital Driver
```
python N-human_policy_learning.py
```

## UnaHug-RL Training
Modify name of the mode in **main.py** file to "UnaHug", and run:
```
python main.py
```

## Parameters
Feel free to play with the parameters in **config.yaml**. 




