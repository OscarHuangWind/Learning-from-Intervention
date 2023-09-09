# LfHI: UnaHug-RL for Autonomous Driving
### :page_with_curl: Learning from Heterogeneous Intervention: Uncertainty-aware Human-guided Reinforcement Learning for Autonomous Driving
:dizzy: This work proposes a novel **_learning from heterogeneous intervention (LfHI)_** approach that considers the **_heterogeneity and intrinsic uncertainty_** of human behaviors in the context of the **_Human-in-the-loop-RL framework_**.  

:red_car: LfHI aims to learn a robust **_uncertainty-aware autonomous driving policy_** through heterogeneous behaviors from **_multi-human concurrent interventions_**.

:wrench: Realized in SMARTS simulator with Ubuntu 20.04 and Pytorch. 

# General Info
- :2nd_place_medal: This work has won the **2nd place in Alibaba Future Car Innovation Challenge (Algorithm Track) 2022**.
- :page_with_curl: [**Alibaba Future Car Innovation Challenge (Algorithm Track) 2022**](https://tianchi.aliyun.com/competition/entrance/531996/rankingList)

# Video: Multi-Human Demonstration
Distinct genders, age groups, and driving proficiency.

https://github.com/OscarHuangWind/Learning-from-Intervention/assets/41904672/c6259c23-4a88-4283-9aa0-579e0b1ae2c9

# Video: Unprotected Left Turn
https://github.com/OscarHuangWind/Learning-from-Intervention/assets/41904672/8227ce00-73e5-4037-9192-5a837e5c2500

# Framework

<p align="center">
<img src="https://github.com/OscarHuangWind/Learning-from-Intervention/blob/master/presentation/LfHI_framework.png">
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




