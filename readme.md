# ADP

This repository contains an implemention of the Adaptive Dopamine Policy inspired from the work of Decision-Pretrained Transformer (DPT) from the paper [Supervised Pretraining Can Learn In-Context Reinforcement Learning](https://arxiv.org/abs/2306.14892).
This repo supports pretraining and evaluating in the following settings:
- Bandits

## Instructions for Setting Up the Environment


To create a new conda environment, open your terminal and run the following command:

```bash
conda create --name adp python=3.9.15
```

```bash
torch==1.13.0
torchvision==0.14.0
```
For example, you might run:

```bash
conda install pytorch=1.13.0 torchvision=0.14.0 cudatoolkit=11.7 -c pytorch -c nvidia
```

The remaining requirements are fairly standard and are listed in the `requirements.txt`. These can be installed by running

```bash
pip install -r requirements.txt
```


## Running Experiments

Each experiment has three phases: (1) pretraining data collection (2) pretraining (3) evaluation of the in-context algorithm. There is a  file `run_bandit.sh`, that show example usage to run these. Training in all settings can take several hours, so it may be prudent to start with smaller problems (e.g. fewer arms, reduced time horizon, etc.). 
