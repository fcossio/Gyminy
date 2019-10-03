# Gyminy
Framework to "easily" implement reinforcement learning for robotic humanoids<br/>
Gym + Roboschool + Stable_baselines + Nao URDF
<br/>
<br/>
Currently is under development and only works for the Nao robot by Softbank Robotics.
There is a fork of OpenAI's Roboschool repo that can simulate the Nao in a similar way that "Atlas Forward Walk" environment works. As of Oct 3rd 2019, we have not found a good reward function to train the agent.<-HELP WANTED
<br/>
## Instalation
To use the roboschool fork you need to use conda.
```
https://github.com/fcossio/Gyminy.git
cd Gyminy/RoboschoolFork
conda env create -f environment.yml
conda activate rbs
pip install tensorflow
pip install roboschool
pip install stable_baselines
pip install -r requirements.txt
pip install -e .
```

You can use the gym enviroment by
```python
import gym, roboschool, roboschoolfork_nao
gym.make("NaoLLC-v1")
```
##### _by: Alfonso Brown, and Fernando Cossio._ <br/>
2019
