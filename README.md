# Gyminy 🦿🦿

Teaching a Humanoid Robot to Walk in a Reinforcment Learning Framework

Our ingredients 🧑‍🍳: Gym + Roboschool + Stable_baselines + Nao URDF 

![Nao Walks](https://user-images.githubusercontent.com/39391180/185484973-deb34d87-2726-462e-af19-fa464d0ce5ce.gif)

## Google docs

[Final presentation slides](https://docs.google.com/presentation/d/1L1BGWqgA0MRw4RYff0OiOh2cwVkDFJicj5Y1E3k-QVw/edit?usp=sharing)

[Compilation of failures](https://drive.google.com/file/d/1tDZwt6tiwnU02cslxOfT5Amw7LAqPBey/view)

## Blog
[The robot diaries](https://medium.com/@fcossio/the-robot-diary-a-reinforcement-learning-journey-b8453ff8e16b)

## Poster
<img src="./poster final gyminy.svg">

Framework to "easily" implement reinforcement learning for robotic humanoids<br/>


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
sudo apt install libopenmpi-dev
pip install -r requirements.txt
pip install -e .
apt-get install libpcre3-dev
```

You can use the gym enviroment by
```python
import gym, roboschool, roboschoolfork_nao
gym.make("NaoLLC-v1")
```
##### _by: Alfonso Brown, and Fernando Cossio._ <br/>
2019
