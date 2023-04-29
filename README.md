# CS_Final_Project
Using reinforcement learning to play the Atari Atlantis game

## File Breakdown
#### DDQN_Atlantis
Contains all relevant files used in implementing DDQN and training
* train.py - master file to train the agent NOTE: save locations must be changed (log_dir variable)
* DDQNetwork - contains DDQN class to create and traind DDQN models
* Environment.py - creates OpenAI gym environment class and associated functions
* Policy.py - creates policy class, used to select actions
* visualization.ipynb - Used to create graphs and visualize training results

#### OldVersions_Ignore
Contains previous iterations of implementing the project. These are irrelevant
