# Learmomg from Heterogeneous Intervention: Uncertainty-aware Human-guided Reinforcement Learning for Autonomous Driving
:dizzy: A novel **_learning from heterogeneous intervention (LfHI)_** approach that considers the **_heterogeneity and intrinsic uncertainty_** of natural human behaviors in the context of the **_Human-in-the-loop-RL_** framework for learning robust **_autonomous driving_** policy under heterogeneous policies from multi-human interventions and corresponding uncertainties.

:wrench: Realized in SMARTS simulator with Ubuntu 20.04 and Pytorch. 

# Framework

<p align="center">
<img src="https://github.com/OscarHuangWind/Learning-from-Intervention/blob/master/presentation/LfHI.png">
</p>

# User Guide

## Clone the repository.
cd to your workspace and clone the repo.
```
git clone https://github.com/OscarHuangWind/Learning-from-Intervention.git
```

## Create a new virtual environment with dependencies.
```
cd ~/$your workspace/Learning-from-Intervention
conda env create -f environment.yml
```
You can modify your virtual environment name and dependencies in **environment.yml** file.

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

## Build the scenario in case of adding new scenarios
(:heavy_exclamation_mark:No need for LeftTurn, RampMerge, and T-Intersection scenarios:heavy_exclamation_mark:)

e.g. suppose to build Roundout scenario.
```
cd <path/to/UnaHug-RL>
scl scenario build --clean Scenario/Roundout/
```
## DRL Training
```
python main.py
```
## Visualization
Type the following command in the terminal:
```
scl envision start
```
Then go to http://localhost:8081/

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




