# DQN CartPole-v1 Agent

This project is for assignment 2 of Reinforcement learning course at Leiden University.

## 📌 Overview

- Uses a deep neural network to approximate Q-values
- Implements an experience replay buffer
- Trains the agent to balance a pole on a moving cart
- Visualizes training progress using matplotlib

## 🚀 Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```


### How To Run

Run the script 


```base
python a21.py
```

### Folders

Folders name include gamma is for gamma ablation.
Folders name include lr is for learning rate ablation.


### Output

You will see csv files for different process, REINFORCE, AC, A2C and a graph for comparison

### Notes

This implementation uses SyncVectorEnv for parallel environments.

You can tweak hyperparameters (e.g. learning rate, gamma value) in the code to improve training.